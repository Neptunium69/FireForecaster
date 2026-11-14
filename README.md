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

`FireHistoryEncoder` accepts `[batch, time, channels, height, width]` history and
`[batch, time]` day offsets. It returns `[batch, 32, height, width]` features.
Forecast heads and training remain to be implemented.

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
