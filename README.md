# FireForecaster

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

`UNetDecoder` is a separate, trainable decoder. It upsamples to each skip map's
actual dimensions and returns `[batch, 32, height, width]` features, including for
odd and rectangular inputs. Add the active-fire and burn-growth heads to these
features; those heads and the training loop remain to be implemented.

```python
from model import FireForecast, MemoryAdapter

adapter = MemoryAdapter(
    high=1.0, low=-1.0, dims=3, size=4096, device="cuda",
    shape={"encoder": 512}, steps=8,
)
model = FireForecast(input_channels=12, memory_adapter=adapter).cuda()
adapter.reset_memory()  # Start of an independent fire sequence
features = model(history, dates)
```

`history` is `[B,T,C,H,W]`, with H and W at least 16. `dates` is `[B,T]`.
ALPACA's current memory supports B=1. The encoder/decoder alone supports larger
batches: use `FireForecast(input_channels=12)` without an adapter.

The memory adapter compresses the 128-channel bottleneck into 512 values, runs
feedback-controlled memory steps, and fuses the resulting readout back into the
bottleneck before decoding. State persists across calls. Reset between independent
fires; call `adapter.detach_memory()` at truncated-backpropagation boundaries.
Memory dynamics currently use FP32; mixed-precision training has not been validated.
ALPACA's development package must also be installed in the training environment.

GPU checks covered odd-size reconstruction, temporal padding, decoder gradients
through all skips, and an eight-turn 4096-neuron memory forward pass. Full-network
backward remains unverified; the earlier environment check found a missing C
compiler required by the installed PyTorch/Triton backward path.

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

Run `test.ipynb` in the GPU environment. Install its small optional dependencies
with `python -m pip install -r requirements-baselines.txt`. It loads local
`baseline_1.pt` and `data/baselines/wsts_fold1.pt`; inference does not download weights.
The one-time `python wsts_baseline.py --download` command retrieves only 55
original WSTS daily rasters (189 MB transferred), skips completed files, and
refuses servers that ignore byte-range requests. Inputs are already downloaded
on this machine.

The comparison uses the intersection of the frozen validation list and the three
2020 fires in `wsts_manifest.json`: 43 one-day forecasts from day 6 onward. All
models are scored on exactly the same native TS-SatFire pixels and dates. WSTS
archives cover 26–31% of these maps, so results in
`data/evaluation/wsts_overlap/` are a separate subset comparison and do not
replace the full 11-fire history-length results. Neither input maps nor targets
are downscaled. WSTS uses original archived input measurements; our model uses
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
trained on 2018/2019, validated on 2021, and tested on 2020. Backbone source
is included in `wsts_backbones/` with its MIT license. Original input archive:
https://doi.org/10.5281/zenodo.8006177.

Released WSTS weights: https://huggingface.co/saadlahrichi/WSTSPlus/blob/main/trained_model_weights/Res18UTAE_T5/All/fold1_testAP0.420.pth
Save this file locally as `data/baselines/wsts_fold1.pt`.

## Lower-VRAM training

`train.ipynb` enables `low_vram=True`. The spatial encoder processes one full-size
day at a time, checkpoints its pure CNN operations, and stores saved tensors in
system RAM with PyTorch `save_on_cpu`. Temporal attention still sees the entire
history. Stateful ALPACA memory and BatchNorm are evaluated once per forecast.
The low-VRAM loader avoids worker prefetch copies of large histories.

This trades extra computation and CPU/GPU transfers for lower VRAM use. Existing
model and optimizer checkpoints remain compatible. Tests of the actual notebook
step with 63 and 64 history days at 596x595 passed forward, backward and optimizer
updates, peaking at about 5.2 GiB allocated VRAM, 5.7 GiB reserved VRAM and 19.1 GiB
process RAM. Results are in `data/evaluation/low_vram/verification.json`. Reload the
updated notebook and model before restarting training from its existing checkpoint.

This run still trains on growing observed histories. The planned third training
run (second fine-tune) will introduce short recursive rollouts; it is not enabled
by this memory optimization.
