# FireForecaster

## Introduction

FireForecaster predicts how a wildfire may spread and recommends which user-selected locations to prioritize for protection planning. It was developed for the Veritas AI hackathon's AI + Climate track using the [TS-SatFire dataset](https://doi.org/10.1038/s41597-025-06271-3).

The model recursively predicts active fire, cumulative burned area, and newly burned area from the available observation history. Each predicted active-fire and burned-area map becomes part of the next day's input. The app combines these forecasts with custom points of interest and protection values to produce a ranked list and a plain-language recommendation.

The main entry points are:

| File | Purpose |
| --- | --- |
| `model.py` | U-TAE encoder, spatial SNN memory, decoder, and output heads |
| `train.ipynb` | Training and recursive fine-tuning |
| `test.ipynb` | Recursive evaluation against persistence and a pretrained WSTS baseline |
| `prepare_data.py` | Convert raw TS-SatFire TIFFs into daily NPZ files |
| `checkpointing.py` | Load current and legacy model checkpoints |
| `frontend/app.py` | Executable Streamlit frontend |
| `frontend/engine.py` | Forecast inference and deterministic protection priorities |

## Installation

Use Python 3.11 or newer (Python 3.12 is used in development), Git, and an NVIDIA GPU with a compatible driver. The frontend requires CUDA and checks for at least 6 GiB of free GPU memory; actual requirements grow with image size and history length.

Run these commands from the repository root. Create an environment unless you already have a suitable GPU Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build first. For example, the project has used PyTorch 2.8 with CUDA 12.8:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Use the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) if your GPU or driver needs a different build. Keep an existing compatible CUDA installation rather than replacing it unnecessarily.

Then install the project dependencies:

```bash
python setup_project.py
```

This installs the model, frontend, and WSTS benchmark dependencies from `requirements.txt`. It also packages U-TAE from the fixed upstream commit `987874e27a98bb399277f2d55eb010744d26f881`, verifies the downloaded files, and installs the small `fireforecaster-utae` wheel. Temporary source files are removed; no source checkout in your home directory is required.

**ALPACA must use development commit `8db2079992720ed3d7382d7a468cd22d46c1e0e9`, not an unpinned `dev` branch or a stable release.** This revision's `Memory` implementation matches the installed development environment. The exact Git dependency is recorded in `requirements.txt`. To install it separately:

```bash
python -m pip install "alpaca-robotics @ git+https://github.com/Neptunium69/ALPACA.git@8db2079992720ed3d7382d7a468cd22d46c1e0e9"
```

For notebooks and evaluation:

```bash
python -m pip install ipykernel matplotlib
python -m ipykernel install --user --name fireforecaster --display-name "FireForecaster"
```

Select the **FireForecaster** kernel in your notebook editor. Check the main imports and GPU availability:

```bash
python -c "import torch; from utae import UTAE; from alpaca.moc import Memory; from model import FireForecast; print('CUDA available:', torch.cuda.is_available())"
```

### Data and checkpoints

Installation does not download datasets or model weights. The app expects `checkpoints/spatial/epoch_050.pt`. Place that checkpoint there before running a forecast.

For training, place raw TS-SatFire fire folders under `data/raw/ts-satfire/` and run:

```bash
python prepare_data.py
```

This preserves native resolution and writes daily NPZ files under `data/<fire>/`. Each file contains 29 data channels, its date, and active, cumulative-burned, and newly-burned targets. Missing measurements are stored as NaNs. The loaders create 29 validity-mask channels, giving the network 58 input channels. Burned targets accumulate active-fire detections and available burned-area perimeters; newly burned targets are their day-to-day difference.

`train.ipynb` reads the fixed fire lists in `train_fires.txt` and `val_fires.txt` and resumes from `checkpoints/spatial/latest.pt` when available. Check its configuration before starting a run.

`test.ipynb` evaluates 32-day probability-feedback forecasts from the first observed day of five matched test fires. It requires the local cohort and WSTS inputs under `data/evaluation/test_epoch050/`, raw test data under `data/raw_test/ts-satfire/`, and the checkpoints named in the notebook. It writes `scores.csv` and `day1_32day_metric_grid.png` in the working directory. WSTS weights are available from [WSTSPlus](https://huggingface.co/saadlahrichi/WSTSPlus/blob/main/trained_model_weights/Res18UTAE_T5/All/fold1_testAP0.420.pth); save them as `checkpoints/baselines/wsts_fold1.pt`.

## Architecture

```text
Observation history + validity masks + dates
                    |
         U-TAE spatial encoder
          and temporal attention
                    |
       Spatial SNN memory readout
                    |
          Separate U-Net decoder
                    |
              Output heads
       active / burned / newly burned
                    |
       Probability feedback for next day
```

1. **U-TAE encoder:** Extracts spatial features from each frame and uses temporal attention to aggregate the available history. It returns a 128-channel bottleneck and three skip maps. U-TAE's original decoder is discarded.
2. **Spatial SNN memory:** The ALPACA-based `MemoryAdapter` stores voltage and spikes for 24 neurons at each bottleneck-grid location. Four neuron steps integrate the encoded observation and fire maps, with a learned 3×3 convolution exchanging recurrent signals between neighbors. The memory readout modifies the bottleneck before decoding.
3. **U-Net decoder:** Upsamples the modified bottleneck and combines it with the encoder's skip maps to produce 32 full-resolution feature channels.
4. **Output heads:** Estimate existing burned area, active fire, and conditional burn growth. Newly burned area is `(1 - existing_burned) * growth`; cumulative burned area is `existing_burned + newly_burned`.

Inputs are `history: [B, T, 58, H, W]` and `dates: [B, T]`. Each returned map has shape `[B, 1, H, W]`. `B` is batch size, `T` is history length, and `H, W` are image dimensions. Channels 27 and 28 hold the active-fire and cumulative-burned inputs. Height and width must each be at least 16.

```python
from checkpointing import load_forecaster

model = load_forecaster("checkpoints/spatial/epoch_050.pt", "cuda")
model.memory.reset_memory()
active, burned, new = model(history, dates, update_memory=True)
```

Write memory for usable real observations. During recursive prediction or observation gaps, pass `update_memory=False` to read the held state. Reset between independent fires. The app feeds active and burned probabilities back into a growing history while keeping imagery and environmental inputs fixed at issuance. The existing-burned head is evaluated again each step, so cumulative predictions are not explicitly forced to be monotonic across separate calls.

## Frontend

Launch the app from the same installed environment:

```bash
python -m streamlit run frontend/app.py --server.address=127.0.0.1
```

Open [localhost:8501](http://127.0.0.1:8501). You can also run `python frontend/app.py`; this launcher uses `frontend/.venv` if present, otherwise the current Python interpreter. There is no JavaScript build step.

Upload a ZIP or folder containing one fire with consecutive observation dates:

```text
fire/
  VIIRS_Day/YYYY-MM-DD_VIIRS_Day.tif
  VIIRS_Night/YYYY-MM-DD_VIIRS_Night.tif
  FirePred/YYYY-MM-DD_FirePred.tif
  points.geojson                         optional
```

Daytime TIFFs must contain the eight original TS-SatFire bands. Missing nighttime or environmental files use the same missing-value handling as training. Optional GeoJSON points use WGS84 longitude and latitude. The app reads raw TIFFs directly and preserves native resolution.

1. Upload the observations and click **Load upload**.
2. Choose **Add point**, click the map, and enter a name, protection value, and buffer radius.
3. Choose the forecast horizon and exposure threshold, then click **Run forecast**.
4. Use the day slider and layer selector to inspect predictions. Download the decision report for the ranked locations and recommendation.

The map uses a nonlinear color scale to make expansion easier to see. Changing days preserves the map view. Forecasts use the fixed epoch-50 checkpoint and probability feedback. The optional stop setting ends the rollout when a point first crosses the exposure threshold; otherwise the app supports up to 64 days.

Protection ranking is deterministic:

```text
exposure = max(active score in buffer,
               burned score over pixels unburned at issuance in buffer)
priority = max over forecast days of (protection value × exposure / (1 + lead day))
```

The recommendation selects the highest-priority location among those crossing the chosen threshold. If none crosses, it recommends continued monitoring. These are model scores and planning priorities, not calibrated asset-impact probabilities or crew assignments. Recommendations use templates and require no LLM or API key.

The separately packaged demonstration is `frontend/demo/thomas_fire_first5.zip`, containing the Thomas Fire's first five supplied dates, December 4–8, 2017. Load it through the normal upload control. It is a validation-fire demonstration, not an unseen test. Data and the demo ZIP are excluded from Git.

## Licensing

Copyright 2026 FireForecaster contributors.

The project's original source code and documentation are licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).

Third-party components retain their own licenses:

- **ALPACA:** [Apache 2.0](https://github.com/Neptunium69/ALPACA/blob/8db2079992720ed3d7382d7a468cd22d46c1e0e9/LICENSE), pinned to the development commit documented above.
- **U-TAE:** [MIT](https://github.com/VSainteuf/utae-paps/blob/987874e27a98bb399277f2d55eb010744d26f881/LICENSE), copyright 2021 VSainteuf. The installed wheel includes the upstream license and provenance. Local packaging uses relative imports, exports `UTAE`, and replaces repeated attention-query copies with tensor expansion.
- Other installed dependencies retain their accompanying license notices.

This source-code license does not relicense TS-SatFire or WSTS data, externally supplied pretrained checkpoints, or third-party images and videos. Follow the terms from their original providers when using or redistributing those assets. In particular, the WSTS baseline weights and the wildfire video are not covered by this project's Apache grant.

Upstream references: [U-TAE paper](https://arxiv.org/abs/2107.07933), [ALPACA](https://github.com/Neptunium69/ALPACA), [TS-SatFire](https://doi.org/10.1038/s41597-025-06271-3), [WildfireSpreadTS data](https://doi.org/10.5281/zenodo.8006177), and [WSTSPlus weights](https://huggingface.co/saadlahrichi/WSTSPlus).
