"""Load FireForecaster checkpoints and initialize the spatial adapter."""

from pathlib import Path

import torch

from model import FireForecast, LegacyMemoryAdapter, MemoryAdapter


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
SPATIAL_CHECKPOINT_DIR = CHECKPOINT_DIR / "spatial"


def spatial_checkpoint_path():
    for name in ("latest.pt", "best.pt", "initial.pt"):
        path = SPATIAL_CHECKPOINT_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No spatial checkpoint in {SPATIAL_CHECKPOINT_DIR}. Run train.ipynb first.")


def load_forecaster(path, device, *, use_memory=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    if not use_memory:
        weights = {key: value for key, value in weights.items() if not key.startswith("memory.")}
        memory = None
    elif "memory.drive.0.weight" in weights:
        memory = MemoryAdapter()
    elif "memory.memory.weights" in weights:
        memory = LegacyMemoryAdapter(
            high=1, low=-1, dims=3, size=4096, device="cpu", shape={"encoder": 512},
            input_gain=checkpoint.get("snn_input_gain", 8.0), grid_size=checkpoint.get("grid_size"),
        )
    else:
        memory = None
    model = FireForecast(58, memory)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def initialize_spatial(model, legacy_checkpoint, spatial_adapter=None):
    """Use the legacy backbone and either a trained or fresh spatial adapter."""
    weights = legacy_checkpoint.get("model", legacy_checkpoint)
    if spatial_adapter is None:
        prefix = "memory.memory_decoder."
        model.memory.fusion.load_state_dict({
            name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)
        }, strict=True)
    else:
        model.memory.load_state_dict(spatial_adapter, strict=True)
    weights = {name: value for name, value in weights.items() if not name.startswith("memory.")}
    weights.update({f"memory.{name}": value for name, value in model.memory.state_dict().items()})
    model.load_state_dict(weights, strict=True)
