"""Recursive forecasts from every observed starting day of one complete fire."""
from pathlib import Path
import csv
import gc
import hashlib
import json
import time

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
from torch.nn import functional as F

from checkpointing import load_forecaster, spatial_checkpoint_path
from model import MemoryAdapter
from evaluate_baselines import encode_day, predict_cached, reset_memory, score
from prepare_data import has_fire_observation
import wsts_baseline as wsts

ROOT = Path(__file__).resolve().parent
FIRE = '24462753'  # First comparable validation fire; chosen before inspecting rollout scores.
OUT = ROOT / 'data/evaluation/recursive_fire_spatial' / FIRE
MODES = ('probability', 'binary')
KEYS = ('active', 'new', 'burned')


def copy_memory(net):
    if net.memory is None:
        return None
    if isinstance(net.memory, MemoryAdapter):
        cells = net.memory.cells
        return cells.voltage.clone(), cells.spikes.clone(), net.memory.state_shape
    m = net.memory.memory
    return m.voltage.clone(), m.spikes.clone(), {k: v.clone() for k, v in m.y.items()}


def restore_memory(net, saved):
    if net.memory is None:
        return
    if isinstance(net.memory, MemoryAdapter):
        net.memory.cells.voltage, net.memory.cells.spikes = saved[0].clone(), saved[1].clone()
        net.memory.state_shape = saved[2]
        return
    m = net.memory.memory
    m.voltage, m.spikes = saved[0].clone(), saved[1].clone()
    m.y = {k: v.clone() for k, v in saved[2].items()}


def feedback(value, mode):
    return (value >= 0.5).astype(np.float32) if mode == 'binary' else value.copy()


def predict_wsts_native(net, history):
    x = torch.from_numpy(np.stack(history)).cuda()[None]
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32))
    return net(x).sigmoid()[0, 0, :h, :w].cpu().numpy()


def align(probability, source, reference):
    result = np.full(reference['shape'], np.nan, dtype=np.float32)
    reproject(probability, result, src_transform=source['transform'], src_crs=source['crs'],
              dst_transform=reference['transform'], dst_crs=reference['crs'],
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    return result[None]


@torch.inference_mode()
def run(checkpoints=None, output_dir=None):
    OUT = Path(output_dir) if output_dir else globals()['OUT']
    model_checkpoints = ({Path(p).stem: Path(p).resolve() for p in checkpoints}
                         if checkpoints else {'ours': spatial_checkpoint_path()})
    models = tuple(model_checkpoints) if checkpoints else ('persistence', 'wsts_proxy', 'ours')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.20)
    torch.backends.cudnn.benchmark = False
    paths = sorted((ROOT/'data'/FIRE).glob('*.npz'))
    assert FIRE in (ROOT/'val_fires.txt').read_text().splitlines()
    days = []
    for path in paths:
        with np.load(path) as z:
            days.append({k: z[k] for k in ('image', 'date', *KEYS)})
    dates = [int(d['date']) for d in days]
    assert np.all(np.diff(dates) == 1), 'This experiment requires a consecutive daily sequence'
    source_paths = [wsts.RAW/FIRE/f'{p.stem}.tif' for p in paths[:-1]]
    ref_path = next((ROOT/'data/raw/ts-satfire'/FIRE/'VIIRS_Day').glob('*.tif'))
    with rasterio.open(source_paths[0]) as src, rasterio.open(ref_path) as ref:
        source = dict(shape=src.shape, transform=src.transform, crs=src.crs)
        reference = dict(shape=ref.shape, transform=ref.transform, crs=ref.crs)
    mask = np.isfinite(align(np.ones(source['shape'], dtype=np.float32), source, reference))
    source_inputs = [wsts.load_input(p) for p in source_paths]
    rows, checks = [], []
    started = time.monotonic()
    checkpoint = next(iter(model_checkpoints.values()))
    for name in models:
        net = None
        observed_cache = []
        if name in model_checkpoints:
            checkpoint = model_checkpoints[name]
            net = load_forecaster(checkpoint, 'cuda')
            observed_cache = [encode_day(net, d['image']) for d in days[:-1]]
            reset_memory(net)
        elif name == 'wsts_proxy':
            net = wsts.load_model()
        spatial = name in model_checkpoints and isinstance(net.memory, MemoryAdapter)
        for origin in range(len(days)-1):
            # Prefix contains real observations only up to issuance. The memory
            # snapshot is warmed exclusively using earlier observed prefixes.
            if name in model_checkpoints:
                warm_state = copy_memory(net)
            image = days[origin]['image']
            observed = spatial and has_fire_observation(np.isfinite(image))
            maps = torch.from_numpy(image[27:29])[None] if observed else None
            first_mode = None
            for mode in MODES:
                if name in model_checkpoints:
                    restore_memory(net, warm_state)
                    history = observed_cache[:origin+1].copy()
                elif name == 'wsts_proxy':
                    history = [x.copy() for x in source_inputs[:origin+1]]
                    # WSTS predicts no acquisition hour. Persist the mean hour
                    # of the latest past day with detections as an explicit surrogate.
                    hour = float(wsts.MANIFEST['means'][22])
                    for x in reversed(history):
                        detected = x[39] > 0
                        if detected.any():
                            hour = float((x[38][detected]*wsts.MANIFEST['stds'][22]
                                          +wsts.MANIFEST['means'][22]).mean())
                            break
                history_dates = dates[:origin+1].copy()
                previous = days[origin]['burned'].astype(np.float32)
                for target_index in range(origin+1, len(days)):
                    # Ground-truth target enters scoring below, never the prediction path.
                    if name in model_checkpoints:
                        write = not spatial or (target_index == origin+1 and observed)
                        prediction = predict_cached(net, history, history_dates,
                                                    observed_fire_maps=maps, update_memory=write)
                        if target_index == origin+1:
                            next_warm_state = copy_memory(net)
                    elif name == 'wsts_proxy':
                        native = predict_wsts_native(net, history)
                        active = align(native, source, reference)
                        prediction = wsts.burned_proxy(active, previous)
                    else:
                        prediction = dict(active=days[origin]['active'].astype(np.float32),
                                          burned=previous, new=np.zeros_like(previous))
                    if target_index == origin+1:
                        if first_mode is None:
                            first_mode = {k: v.copy() for k,v in prediction.items()}
                        else:
                            errors = {k:float(np.max(np.abs(v[mask]-first_mode[k][mask])))
                                      for k,v in prediction.items()}
                            assert max(errors.values()) == 0, errors
                            checks.append(dict(model=name,start_day=origin+1,first_step_mode_errors=errors))
                    for key in KEYS:
                        p = prediction[key][mask]
                        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                        s = score([p], [days[target_index][key][mask]])
                        denominator = s['tp']+s['fp']+s['fn']
                        rows.append(dict(model=name, feedback=mode, start_day=origin+1,
                                         start_date=paths[origin].stem, target_day=target_index+1,
                                         target_date=paths[target_index].stem,
                                         horizon=target_index-origin, output=key,
                                         iou=s['tp']/denominator if denominator else None, **s))
                    if target_index == len(days)-1:
                        break
                    previous = feedback(prediction['burned'], mode)
                    if name in model_checkpoints:
                        generated = days[origin]['image'].copy()
                        generated[27:28] = feedback(prediction['active'], mode)
                        generated[28:29] = previous
                        # All non-fire channels remain fixed at issuance. Never load
                        # future imagery, future observed weather, or future labels here.
                        history.append(encode_day(net, generated))
                        history_dates.append(dates[origin] + target_index-origin)
                        np.testing.assert_array_equal(generated[:27], days[origin]['image'][:27])
                    elif name == 'wsts_proxy':
                        generated = source_inputs[origin].copy()
                        fire_input = feedback(native, mode)
                        generated[39] = fire_input
                        generated[38] = (hour*fire_input-wsts.MANIFEST['means'][22])/wsts.MANIFEST['stds'][22]
                        history.append(generated)
                        np.testing.assert_array_equal(generated[:38], source_inputs[origin][:38])
                if name != 'persistence':
                    del history
                print(name, mode, 'start',origin+1,'/',len(days)-1,
                      'elapsed',round(time.monotonic()-started),'s',flush=True)
            if name in model_checkpoints:
                # Discard simulated future state before moving to the next origin.
                restore_memory(net, next_warm_state)
            (OUT/'partial.json').write_text(json.dumps(rows))
        del observed_cache, net
        gc.collect(); torch.cuda.empty_cache()
    report = dict(fire=FIRE, first_date=paths[0].stem,last_date=paths[-1].stem,days=len(days),
                  scored_pixels=int(mask.sum()),full_pixels=mask.size,coverage=float(mask.mean()),
                  checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  wsts_checkpoint_sha256=hashlib.sha256((wsts.BASE/'wsts_fold1.pt').read_bytes()).hexdigest(),
                  protocol={
                      'observed_history':'All real days 1..start_day; append only model-generated fire channels thereafter. Histories grow in both neural models.',
                      'exogenous_inputs':'All non-fire channels frozen at the last observed day of each origin, including imagery and weather/forecast channels. No target-day observations fed into forecasts.',
                      'ours':'Feed predicted active and cumulative-burned maps into channels 27/28 of the encoder input. Spatial memory writes available observations at issuance and is held through each rollout. Legacy checkpoints use their dense recurrence during each rollout. Restore the observed prefix for each independent origin/mode.',
                      'wsts':'Native source-grid active probabilities fed into channel 39. Acquisition-time surrogate in channel 38 uses the latest past nonempty detection-day mean hour, multiplied by feedback fire values, then standardized. Burned maps are deterministic accumulation proxies.',
                      'binary':'Threshold only feedback maps at 0.5. Evaluate the raw forecast probabilities for AP. IoU always thresholds forecast output at 0.5.',
                      'probability':'Feed continuous active and cumulative-burned probabilities back.',
                      'persistence':'Keep the origin active and cumulative-burned maps constant; new burned area zero. Both feedback modes are identical.',
                      'metrics':'Exact spatial AP for each target date, undefined when that target has no positives; IoU at 0.5, undefined when union empty. Same common native TS-SatFire pixels for all models.',
                      'scope':'One complete fire sequence, with WSTS shared spatial coverage only. Native spatial resolution; no cropping or downscaling of original TS-SatFire input.',
                      'caveat':'These models were trained for one-day forecasting, not this recursive frozen-input setting. WSTS was trained on five observations; growing histories and synthesized feedback are out of training distribution.'},
                  evaluated_checkpoints={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                         for name, path in model_checkpoints.items()},
                  checks=checks,rows=rows,elapsed_seconds=time.monotonic()-started)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    with (OUT/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.checkpoints, args.output_dir)
