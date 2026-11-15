from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import time
import zlib
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'checkpoints' / 'baselines'
RAW = ROOT / 'data' / 'baselines' / 'wsts_inputs'
MANIFEST = json.loads((ROOT / 'wsts_manifest.json').read_text())


def download_inputs():
    def read_range(start, count):
        end = start + count - 1
        url = MANIFEST['archive_url'] + f'?range_start={start}&range_end={end}'
        request = Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urlopen(request, timeout=60) as response:
            expected = f"bytes {start}-{end}/{MANIFEST['archive_bytes']}"
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise IOError('Server did not honor the byte range; refusing full archive download')
            data = response.read(count + 1)
        if len(data) != count:
            raise IOError('Incomplete byte range')
        return data

    def fetch(entry):
        path = RAW / entry['fire'] / f"{entry['date']}.tif"
        if path.exists() and path.stat().st_size == entry['size']:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                header = read_range(entry['offset'], 30)
                if header[:4] != b'PK\x03\x04':
                    raise IOError('Invalid ZIP header')
                method = struct.unpack_from('<H', header, 8)[0]
                name_length, extra_lenworksgth = struct.unpack_from('<HH', header, 26)
                data = read_range(entry['offset'] + 30 + name_length + extra_length,
                                  entry['compressed'])
                data = zlib.decompress(data, -15) if method == 8 else data
                if len(data) != entry['size']:
                    raise IOError('Incomplete TIFF')
                temporary = path.with_suffix('.part')
                temporary.write_bytes(data)
                with rasterio.open(temporary) as src:
                    if src.count != 23 or src.crs is None:
                        raise ValueError('Expected georeferenced 23-band WSTS input')
                    src.read()  # Check compressed TIFF blocks before accepting the file.
                temporary.replace(path)
                print(entry['fire'], entry['date'], flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, MANIFEST['files']))


def load_input(path):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
    x[-1] = np.floor(np.nan_to_num(x[-1], nan=0) / 100)
    active = (x[-1:] > 0).astype(np.float32)
    x[[7, 13, 19]] = np.sin(np.deg2rad(x[[7, 13, 19]]))
    means = np.asarray(MANIFEST['means'], dtype=np.float32)[:, None, None]
    stds = np.asarray(MANIFEST['stds'], dtype=np.float32)[:, None, None]
    x = np.nan_to_num((x - means) / stds, nan=0)
    # Exactly reproduce the authors' one-hot ordering and missing-class behavior.
    landcover = np.eye(17, dtype=np.float32)[x[16].astype(int) - 1].transpose(2, 0, 1)
    return np.concatenate([x[:16], landcover, x[17:], active])


class WSTSForecast(nn.Module):

    def __init__(self, LTAE2d, TemporalAggregator):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=40, classes=1)
        self.ltae = LTAE2d(in_channels=512, n_head=16, d_k=4, mlp=[256, 512], dropout=0.2, d_model=256, T=1000, return_att=True, positional_encoding=True)
        self.temporal_aggregator = TemporalAggregator(mode='att_group')

    def forward(self, x, positions=None):
        stages = [self.model.encoder(x[:, t]) for t in range(x.shape[1])]
        positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        last, attention = self.ltae(torch.stack([s[-1] for s in stages], dim=1), batch_positions=positions)
        attention = attention.contiguous()
        skips = [self.temporal_aggregator(torch.stack([s[i] for s in stages], dim=1), attn_mask=attention) for i in range(1, len(stages[0]) - 1)]
        decoded = self.model.decoder(stages[0][0], *skips, last)
        return self.model.segmentation_head(decoded)


def load_model(device='cuda'):
    from utae.ltae import LTAE2d
    from utae.utae import Temporal_Aggregator
    net = WSTSForecast(LTAE2d, Temporal_Aggregator)
    net.load_state_dict(torch.load(BASE / 'wsts_fold1.pt', map_location='cpu', weights_only=True))
    return net.to(device).eval()


@torch.inference_mode()
def predict(net, inputs, source_path, reference_path):
    # No cropping or downscaling: only pad the full source to a multiple of 32.
    x = torch.from_numpy(np.stack(inputs)).to(next(net.parameters()).device)[None]
    height, width = x.shape[-2:]
    x = F.pad(x, (0, (-width) % 32, 0, (-height) % 32))
    probability = net(x).sigmoid()[0, 0, :height, :width].cpu().numpy()
    with rasterio.open(source_path) as src, rasterio.open(reference_path) as dst:
        result = np.full(dst.shape, np.nan, dtype=np.float32)
        reproject(probability, result, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst.transform, dst_crs=dst.crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return result[None]


def burned_proxy(active, previous_burned):
    """Derived proxy, not a learned burned-area forecast. No target-day data used."""
    new = (1 - previous_burned) * active
    return {'active': active, 'new': new, 'burned': previous_burned + new}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download_inputs()
