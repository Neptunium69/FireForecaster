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
