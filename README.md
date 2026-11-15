# FireForecaster

## Repository files

- `train.ipynb`, `test.ipynb`: training and evaluation entry points.
- `model.py`, `checkpointing.py`: architecture and checkpoint loading.
- `prepare_data.py`, `download.py`: dataset preparation and download utilities.
- `evaluate_baselines.py`, `recursive_fire.py`, `wsts_baseline.py`: evaluation helpers.
- `data/`, `checkpoints/`, and root `.pt` files: datasets and existing model weights.
- `diagnostics/`: experiment scripts, archived logs, the results PDF, and one-off files.
- `diagnostics/backups/runpod-2026-11-15/`: verified Pod backup, including epoch 41.

The extra download list is `diagnostics/additional_train_fires.txt`. Run the
archived single-fire plotter with `python diagnostics/plot_recursive.py`.
Diagnostics and datasets remain excluded from Git.

## Install

Use the Python environment that you will use for training, with GPU-compatible
PyTorch installed. From this repository run:

```bash
python setup_project.py
```

The setup command installs requirements, downloads five checksum-verified files
from a fixed U-TAE upstream commit into a temporary directory, builds a small
wheel, and installs it into the active Python environment. Temporary source files
are removed automatically. It does not use a clone in your home directory.
The first setup needs internet access. Subsequent runs reuse the installed U-TAE
version. Run setup again in each new Python environment.

```python
from utae import UTAE
from model import FireHistoryEncoder
```

`requirements.txt` contains the PyPI dependencies; the complete setup command
above also installs the U-TAE package. Compatible existing PyTorch installations
are retained. For a fresh environment, select the appropriate GPU PyTorch build
before setup. Minimum dependency versions are specified, not a complete lockfile.

## Model

`FireHistoryEncoder` uses U-TAE's spatial encoder and temporal attention only.
It returns a 128-channel temporal bottleneck and three temporally aggregated
skip maps (64, 32, 32 channels, coarse to fine). Its built-in decoder is discarded.
Padded observations are excluded from temporal normalization as well as attention.

`UNetDecoder` upsamples to each skip map's actual dimensions and returns
`[batch, 32, height, width]` features. The output heads predict active fire,
cumulative burned area, and newly burned area.

```python
from checkpointing import load_forecaster, spatial_checkpoint_path

model = load_forecaster(spatial_checkpoint_path(), "cuda")
model.memory.reset_memory()  # Start of an independent fire sequence
active, burned, new = model(history, dates, update_memory=observation_available)
```

`history` is `[B,T,C,H,W]`, with H and W at least 16. `dates` is `[B,T]`.
In the prepared TS-SatFire inputs, channels 27 and 28 contain active-fire and
cumulative-burned-area maps, respectively.

`MemoryAdapter` stores voltage and spikes for 24 neurons at every encoder-grid
location. To update, it combines the bottleneck with the last frame's fire maps.
Four ALPACA `Memory.step` calls integrate that input and recurrent current
from neighboring cells through a shared 3×3 convolution. The decoder gets a
location-aligned readout of the resulting voltages and spikes.

Pass `update_memory=False` to use the held state during observation gaps and
recursive forecasts with generated inputs. By default the forward call writes.
Reset between independent fires, and call `model.memory.detach_memory()` at
truncated-backpropagation boundaries. The training and evaluation pipelines
permit writes on real input days with some finite day or night VIIRS I4 data
and valid pixels in both fire maps. This is a basic availability check for the
prepared inputs. The caller can pass a more specific observation-quality flag.
Memory dynamics run in FP32. ALPACA must be installed in the training environment.

`load_forecaster` also accepts existing dense ALPACA checkpoints. These load
with `LegacyMemoryAdapter`. To construct a fresh spatial model directly, use
`FireForecast(58, MemoryAdapter())`. To disable the adapter, use `FireForecast(58)`.

## Upstream attribution

U-TAE source: https://github.com/VSainteuf/utae-paps
Pinned commit: `987874e27a98bb399277f2d55eb010744d26f881`.
Copyright (c) 2021 VSainteuf, MIT license. The installed package includes the
license and provenance. Packaging changes are limited to relative imports and
an export of `UTAE`; model calculations are unchanged.

The locally built distribution is named `fireforecaster-utae`; it is not an
official upstream PyPI release. The import name is `utae`.

Reference: Sainte Fare Garnot and Landrieu, Panoptic Segmentation of Satellite
Image Time Series with Convolutional Temporal Attention Networks, ICCV 2021.
https://arxiv.org/abs/2107.07933

## Baseline evaluation

Run `test.ipynb` in the GPU environment for the day-1 benchmark. It forecasts 32
days recursively on five held-out fires, scores available target dates, and
saves the results and four-panel chart to
`data/evaluation/test_epoch050_issue_day1_horizon32/`. It uses the verified
cohort and WSTS inputs under `data/evaluation/test_epoch050/`, the Epoch 50
and `baseline_1.pt` FireForecaster checkpoints, and the WSTS checkpoint at
`data/baselines/wsts_fold1.pt`. Install the optional WSTS dependencies with
`python -m pip install -r requirements-baselines.txt`.

The earlier one-day validation comparison runs through `evaluate_baselines.py`.
Pass `checkpoint=Path("baseline_1.pt")` to `evaluate()` to use the historical
checkpoint in that comparison.
The one-time `python wsts_baseline.py --download` command retrieves only 55
original WSTS daily rasters (189 MB transferred), skips completed files, and
refuses servers that ignore byte-range requests. Inputs are already downloaded
on this machine.

The comparison uses the intersection of the frozen validation list and the three
2020 fires in `wsts_manifest.json`: 43 one-day forecasts from day 6 onward. All
models are scored on exactly the same native TS-SatFire pixels and dates. WSTS
archives cover 26–31% of these maps. Spatial-model subset results are written
to `data/evaluation/wsts_overlap_spatial/`. The input maps and targets use
native resolution. WSTS uses its archived inputs, and FireForecaster uses
TS-SatFire measurements.

`wsts_proxy` learns active-fire forecasts only. Its extra outputs are explicitly
derived: `new = (1 - previous_burned) * predicted_active`,
`burned = previous_burned + new`. The previous burned map is observed through
issuance. These are proxies, not native WSTS burned-area outputs. DailyWildfireCAN
is excluded because required Canadian layers do not cover these US fires; no
compatible independent burned-area checkpoint has yet been substituted.

AP is averaged per fire; IoU pools pixel counts at threshold 0.5. All history
lengths use the same target dates. Our memory is warmed from day 2 and reset
between fires and independent history-policy runs. The WSTS checkpoint was
trained with five observations, so its other lengths test extrapolation.
No new reserved test fires are used.

WSTS code and statistics: https://github.com/slahrichi/WildfireSpreadTS
commit `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`. The fold-1 checkpoint
trained on 2018/2019, validated on 2021, and tested on 2020. The baseline reuses
the installed U-TAE dependency, which includes its MIT license; no duplicate
backbone source is needed. Original input archive:
https://doi.org/10.5281/zenodo.8006177.

Released WSTS weights: https://huggingface.co/saadlahrichi/WSTSPlus/blob/main/trained_model_weights/Res18UTAE_T5/All/fold1_testAP0.420.pth
Save this file locally as `data/baselines/wsts_fold1.pt`.

## Recursive training

`train.ipynb` uses growing observed history and recursively predicts up to 16
days. At each origin, the spatial SNN writes the available observation and
holds its state while subsequent predicted fire maps feed back into the encoder.
The notebook resets the state between independent fires.

The training split simulates one- and three-day observation outages. Each
eligible day has a 15% chance of starting a gap after a usable observation.
The gap clears day and night VIIRS channels, both fire maps, and their input
validity masks. The SNN holds its most recent state during the gap. Dates,
environmental inputs, target labels, and validation inputs retain their original
values. Set `synthetic_gap_probability=0` to disable the augmentation.

The notebook starts fine-tuning at epoch 42 from
`data/checkpoints/spatial/epoch_041.pt`. This checkpoint combines the encoder,
decoder, and forecast heads from the completed epoch-41 run with the trained
seed-37 ALPACA spatial adapter from the observation-gap experiment. The first
spatial epoch starts with a fresh optimizer and validation history. Later
runs resume the model and optimizer from `data/checkpoints/spatial/latest.pt`.
Copy `epoch_041.pt` to the same path to start this fine-tune on another machine.

The notebook uses `low_vram=True`. The spatial encoder processes one full-size
day at a time and replays cached CNN features for backward. Temporal attention
sees the entire available history.

## Forecast app

A simple Streamlit app using the existing **epoch 50** PIPS checkpoint. Upload raw TS-SatFire TIFFs, click the map to add points of interest, run probability-feedback forecasts, inspect each day, and get a plain-language protection-planning recommendation.

### Run locally

From FireForecaster:

```bash
./frontend/app.py
```

Open http://127.0.0.1:8501. The local app environment inherits your existing GPU PyTorch environment. It does not change or retrain the model. Leave other GPU-heavy jobs to finish first; the app refuses inference with less than 6 GB free.

For a fresh machine, install the parent project's model dependencies (including its U-TAE package and compatible ALPACA version), copy `data/checkpoints/spatial/epoch_050.pt`, then:

```bash
python -m venv --system-site-packages frontend/.venv
frontend/.venv/bin/python -m pip install -r requirements.txt
./frontend/app.py
```

Create the venv using the Python environment that can already import `torch`, `utae`, `alpaca.moc`, and `rasterio`. Uploading a dataset never loads Python code or an uploaded model checkpoint. This is a local single-operator app, not an authenticated public deployment.

### Input and demonstration

Upload a ZIP or folder containing one fire. The separately packaged example is `frontend/demo/thomas_fire_first5.zip`; it is loaded through the same upload control. There are exactly five dates, December 4–8, 2017, and 15 original TIFFs: daytime, nighttime and environmental data. The raw folder is native resolution (596 × 594); no crop, downscale, future labels or NPZ conversion is included.

The Thomas Fire was chosen because these dates precede additional spread toward communities around Carpinteria and Montecito. Its four approximate demo points represent Carpinteria, Montecito, Ojai and the eastern shore of Lake Casitas. Editable values (1–10) and 2 km buffers are illustrative assumptions, not official protection priorities or asset valuations. The fire belongs to the existing validation split, so this demo is not evidence of unseen-fire performance. [Historical context from the City of Carpinteria](https://carpinteriaca.gov/wp-content/uploads/2026/03/6_Safety_CLUP-GP_Revised_Feb2026.pdf).

```text
fire/
  VIIRS_Day/YYYY-MM-DD_VIIRS_Day.tif
  VIIRS_Night/YYYY-MM-DD_VIIRS_Night.tif
  FirePred/YYYY-MM-DD_FirePred.tif
  points.geojson                         optional
```

Supply one fire with consecutive dates. Day TIFFs require eight original TS-SatFire bands. Missing night/auxiliary files use the same missing-value and validity-mask processing as training. Native alignment, scaling, observed active-fire and cumulative burned construction are reused directly from `prepare_data.py`. GeoJSON point coordinates are longitude, latitude in WGS84. Buffer distances use great-circle distances rather than assuming every geographic-grid pixel is exactly 375 m square.

### Inference and decisions

- Strictly loads `epoch_050.pt` and checks its epoch metadata. No use of the moving `latest.pt`.
- Each real observation updates spatial SNN state once, in order. Memory is held during synthetic forecast steps and reset between runs. This matches the trained epoch 50 rollout semantics.
- CNN features are cached; a bounded GPU feature cache spills to host RAM. All native pixels and growing history are retained.
- Each predicted active/burned map feeds the next input **as probabilities**. Environmental and image channels stay at issuance values; no future observation is accessed. Predicted fire-channel validity flags follow training.
- Calls the original output module without overriding its initial burned head, exactly as training does. Its cumulative-burned output is not additionally forced monotonic across recursive calls; the app does not silently change the model.
- The monitor shows active, cumulative burned and newly burned outputs. The backdrop is a muted grayscale rendering of the last observed VIIRS context, not predicted imagery. A square-root color mapping emphasizes lower and intermediate scores; legend labels remain original model scores. Playback uses a browser-local canvas: slider input updates the overlay without rerunning Streamlit, clearing the displayed image while loading, or resetting zoom/pan. Only display rasters are quantized to 8 bits (hover scores are approximate); inference and decisions retain float32 predictions. The initial viewport focuses on the locations; calculations do not crop.
- A point's exposure is `max(max(active in buffer), max(burned in buffer over pixels unburned at issuance))`. This is a conservative max-pixel score, **not a calibrated probability that an asset will be hit**. Existing burned pixels alone do not trigger future exposure; active fire on burned land still counts.
- Priority is `max_over_leads(value * exposure / (1 + lead))`. Higher user value, higher model scores and shorter lead increase priority. The plain-language recommendation selects the highest-priority point that crosses the selected exposure threshold. If no point crosses, it recommends monitoring instead.
- The default stops after the first forecast day on which any point crosses the threshold (default 0.5 is a demo setting, not an operational standard). The ranking is limited to days actually simulated. Disable this option to compare the full selected horizon, up to 64 days. Longer horizons exceed some training rollouts and should be treated as exploratory.
- Output is an operator-review recommendation about protection planning. The model does not evaluate suppression interventions, crew availability or evacuation-route safety, so it cannot establish a globally optimal tactical action. Template-based wording needs no LLM/API key.

Download the decision JSON for dates, settings, points, scores and wording. Upload temporary files are removed after preprocessing. Model state is not shared between runs. Raw demo files and its ZIP are ignored by Git.

`frontend/package_demo.py` regenerates the package from the existing local raw source and records checksums. It copies only the first five dates.

### Frontend source

The entire frontend is in executable `frontend/app.py`, including the interactive map. Run `./frontend/app.py` or `python frontend/app.py` from the project root. It starts Streamlit using the local `.venv` when present. `frontend/engine.py` remains the separate inference backend. No shell launcher, separate HTML file, or frontend build is needed.

Choose **Add point**, click the map, then name the location and save. Coordinates are calculated automatically. Saving preserves the map view and updates the location table and any existing forecast ranking. Use **Edit points of interest** for table edits.
