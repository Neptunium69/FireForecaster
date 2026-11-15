"""Native-resolution TS-SatFire inference and transparent POI prioritization."""
from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import threading
from time import perf_counter
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates

ROOT = Path(os.environ.get("FIREFORECAST_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from prepare_data import FireDataset, has_fire_observation

CHECKPOINT = ROOT / "checkpoints/spatial/epoch_050.pt"
GPU_LOCK = threading.Lock()


def save_uploads(uploads, destination):
    """Accept a ZIP or folder upload without trusting client paths or extracting code."""
    destination = Path(destination)
    seen = set()
    total = 0

    def save(name, size, read):
        nonlocal total
        name = name.replace("\\", "/").split("/")[-1]
        if name == "points.geojson":
            relative = Path(name)
        elif name.endswith(".tif") and len(name) >= 14:
            kind = next((k for k in ("VIIRS_Day", "VIIRS_Night", "FirePred")
                         if name.endswith(f"_{k}.tif")), None)
            if kind is None:
                return
            # ISO dates and a known suffix only; no uploaded path is used.
            from datetime import date
            date.fromisoformat(name[:10])
            if name != f"{name[:10]}_{kind}.tif":
                raise ValueError(f"Unexpected TIFF filename: {name}")
            relative = Path(kind) / name
        else:
            return
        if relative in seen:
            raise ValueError("Upload one fire only: duplicate dates or filenames were found.")
        total += size
        if size > 512 * 1024**2 or total > 3 * 1024**3:
            raise ValueError("Upload exceeds the 3 GB session limit or 512 MB per-file limit.")
        seen.add(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read())

    for upload in uploads:
        if upload.name.lower().endswith(".zip"):
            with zipfile.ZipFile(upload) as archive:
                for item in archive.infolist():
                    if not item.is_dir():
                        save(item.filename, item.file_size, lambda i=item: archive.read(i))
        else:
            save(upload.name, upload.size, upload.getvalue)
    if not (destination / "VIIRS_Day").exists():
        raise ValueError("No YYYY-MM-DD_VIIRS_Day.tif files were found.")


def load_fire(folder):
    folder = Path(folder)
    files = sorted((folder / "VIIRS_Day").glob("*.tif"))
    if not 1 <= len(files) <= 64:
        raise ValueError("Supply 1–64 consecutive daily VIIRS_Day TIFFs from one fire.")
    for path in files:
        with rasterio.open(path) as src:
            if src.count != 8 or src.crs is None or min(src.shape) < 16:
                raise ValueError(f"{path.name}: expected eight georeferenced TS-SatFire day bands.")
            if max(src.shape) > 1536:
                raise ValueError("This local demo supports native rasters up to 1536 pixels per side.")
    data = FireDataset(folder)
    images = [data.read_day(i).astype(np.float32) for i in range(len(files))]
    if not any(has_fire_observation(np.isfinite(image)) for image in images):
        raise ValueError("No usable fire observation in the uploaded history.")
    rgb = images[-1][[1, 0, 2]].transpose(1, 2, 0)
    for i in range(3):
        valid = rgb[..., i][np.isfinite(rgb[..., i])]
        low, high = np.percentile(valid, (2, 98)) if valid.size else (0, 1)
        rgb[..., i] = np.clip((rgb[..., i] - low) / max(float(high - low), 1e-6), 0, 1)
    return {
        "images": images, "dates": [str(x) for x in data.dates],
        "active": data.active[-1].astype(np.float32),
        "burned": data.burned[-1].astype(np.float32),
        "rgb": (np.nan_to_num(rgb) * 255).astype(np.uint8),
        "crs": data.crs, "transform": data.transform,
        "shape": (data.height, data.width), "name": folder.name,
        "usable": [bool(has_fire_observation(np.isfinite(x))) for x in images],
    }


def read_points(path):
    if not Path(path).exists():
        return []
    collection = json.loads(Path(path).read_text())
    points = []
    for feature in collection["features"]:
        if feature["geometry"]["type"] != "Point":
            raise ValueError("Points of interest must be GeoJSON Point features in WGS84.")
        lon, lat = feature["geometry"]["coordinates"][:2]
        p = feature["properties"]
        points.append(dict(name=str(p["name"]), latitude=float(lat), longitude=float(lon),
                           value=float(p.get("value", 5)), radius_km=float(p.get("radius_km", 2))))
    return points


def point_masks(fire, points):
    """Use great-circle distance on the native grid, including projected source grids."""
    height, width = fire["shape"]
    rr, cc = np.indices((height, width))
    affine = fire["transform"]
    xx = affine.a * (cc + .5) + affine.b * (rr + .5) + affine.c
    yy = affine.d * (cc + .5) + affine.e * (rr + .5) + affine.f
    lon, lat = transform_coordinates(fire["crs"], "EPSG:4326", xx.ravel(), yy.ravel())
    lon, lat = np.radians(np.array(lon).reshape(rr.shape)), np.radians(np.array(lat).reshape(rr.shape))
    masks = []
    names = set()
    for p in points:
        if not p["name"].strip() or p["name"] in names:
            raise ValueError("Give every point a unique, nonempty name.")
        names.add(p["name"])
        if not (1 <= p["value"] <= 10 and .1 <= p["radius_km"] <= 20
                and -90 <= p["latitude"] <= 90 and -180 <= p["longitude"] <= 180):
            raise ValueError(f"Invalid coordinates, value or radius for {p['name']}.")
        x, y = transform_coordinates("EPSG:4326", fire["crs"], [p["longitude"]], [p["latitude"]])
        row, col = rasterio.transform.rowcol(affine, x[0], y[0])
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{p['name']} is outside the uploaded raster.")
        plat, plon = np.radians([p["latitude"], p["longitude"]])
        hav = np.sin((lat - plat) / 2)**2 + np.cos(lat)*np.cos(plat)*np.sin((lon - plon)/2)**2
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        mask = distance <= p["radius_km"]
        mask[row, col] = True  # Subpixel buffers still contain the selected pixel.
        masks.append(mask)
    return masks


def rank_points(fire, points, masks, forecasts, threshold):
    rows = []
    for point, mask in zip(points, masks, strict=True):
        unburned = mask & (fire["burned"] < .5)
        exposure = []
        for frame in forecasts:
            # A max-pixel score, NOT P(any asset is hit). Burned pixels at issuance
            # do not count as newly threatened; active fire still counts as exposure.
            active = float(frame["active"][mask].max())
            burned = float(frame["burned"][unburned].max()) if unburned.any() else 0.
            exposure.append(max(active, burned))
        arrival = next((i + 1 for i, value in enumerate(exposure) if value >= threshold), None)
        priority = max((point["value"] * value / (1 + i + 1)
                        for i, value in enumerate(exposure)), default=0.)
        rows.append({"name": point["name"], "value": point["value"],
                     "priority": round(priority, 4), "first_exposure_day": arrival,
                     "peak_exposure": round(max(exposure, default=0.), 4),
                     "observed_active": bool((fire["active"][mask] > 0).any()),
                     "already_burned_fraction": round(float(fire["burned"][mask].mean()), 4)})
    return sorted(rows, key=lambda row: (-row["priority"], row["name"]))


def recommendation(rows, forecast_days, stopped):
    if not rows:
        return "Add points of interest to receive a protection recommendation."
    if not forecast_days:
        return "Run a forecast to compare protection priorities."
    reached = [p for p in rows if p["first_exposure_day"] is not None]
    if not reached:
        return (f"Continue monitoring. No selected point crosses the exposure threshold within the "
                f"{forecast_days} simulated days. Watch {rows[0]['name']} first: it has the highest "
                "value-weighted exposure score in this run. This does not establish that other areas are safe.")
    top = reached[0]
    text = (f"Prioritize protection planning for {top['name']}. Its buffer first crosses the selected "
            f"exposure threshold on forecast day {top['first_exposure_day']}, and its assigned value "
            f"of {top['value']:g}/10 gives it the highest priority among threshold-crossing locations. "
            "Verify local conditions and access, then review its protection plan with the incident lead.")
    if len(reached) > 1:
        text += " Next, assess " + ", then ".join(p["name"] for p in reached[1:3]) + "."
    if top["observed_active"]:
        text += " Active fire is already present in this buffer in the latest observation."
    if stopped:
        text += " The run stopped at the first threshold crossing; priorities cover only that simulated horizon."
    return text


def run_forecast(fire, points, days, threshold=.5, stop_on_hit=True, progress=None):
    import torch
    from checkpointing import load_forecaster
    from model import RolloutSpatialCache

    if not 1 <= days <= 64 or not 0 < threshold < 1:
        raise ValueError("Choose 1–64 forecast days and a threshold between zero and one.")
    masks = point_masks(fire, points)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python process. Run the app in your GPU environment.")
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        raise RuntimeError("Less than 6 GB of GPU memory is free. Wait for the current benchmark or training run.")
    if not GPU_LOCK.acquire(blocking=False):
        raise RuntimeError("Another app forecast is running. Wait for it to finish.")
    net = cache = None
    started = perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        header = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if header.get("epoch") != 50:
            raise ValueError("The configured checkpoint is not epoch 50.")
        del header
        net = load_forecaster(CHECKPOINT, "cuda")
        net.encoder.low_vram = True
        net.encoder.gpu_cache_bytes = 2 * 1024**3
        net.memory.reset_memory()
        count = len(fire["images"])
        h, w = fire["shape"]
        # Host storage bounds GPU use and preserves the complete native history.
        history = torch.empty((1, count + days, 58, h, w), dtype=torch.float32)
        for i, image in enumerate(fire["images"]):
            valid = np.isfinite(image)
            history[0, i] = torch.from_numpy(np.concatenate((np.where(valid, image, 0), valid)).astype(np.float32))
        outputs = []
        stopped = False
        with torch.inference_mode():
            cache = RolloutSpatialCache(net.encoder, batch_size=1)
            # Write each actual observation once, in order. No synthetic memory writes.
            for i in range(count - 1):
                if fire["usable"][i]:
                    length = i + 1
                    dates = torch.arange(1-length, 1, device="cuda", dtype=torch.float32)[None]
                    bottleneck, _ = net.encoder(history[:, :length], dates, spatial_cache=cache)
                    net.apply_memory(bottleneck, history[:, i, 27:29], update_memory=True)
                if progress:
                    progress(0, days, f"Reading observed day {i+1}/{count}")
            for lead in range(1, days + 1):
                length = count + lead - 1
                dates = torch.arange(1-length, 1, device="cuda", dtype=torch.float32)[None]
                active, burned, new = net(history[:, :length], dates, spatial_cache=cache,
                                          update_memory=lead == 1 and fire["usable"][-1])
                output = {name: value[0, 0].cpu().numpy().copy()
                          for name, value in zip(("active", "burned", "new"), (active, burned, new), strict=True)}
                if any(not np.isfinite(x).all() for x in output.values()):
                    raise RuntimeError("The model produced non-finite predictions.")
                output["date"] = str(np.datetime64(fire["dates"][-1]) + np.timedelta64(lead, "D"))
                outputs.append(output)
                ranking = rank_points(fire, points, masks, outputs, threshold)
                if progress:
                    progress(lead, days, f"Predicted {output['date']} (day +{lead})")
                if stop_on_hit and any(row["first_exposure_day"] is not None for row in ranking):
                    stopped = True
                    break
                if lead < days:
                    history[:, length].copy_(history[:, count-1])
                    history[0, length, 27] = torch.from_numpy(output["active"])
                    history[0, length, 28] = torch.from_numpy(output["burned"])
                    history[0, length, 56:58] = 1
        return {"forecasts": outputs, "ranking": ranking, "stopped": stopped,
                "recommendation": recommendation(ranking, len(outputs), stopped),
                "epoch": 50, "requested_days": days, "threshold": threshold,
                "points": points, "feedback": "probabilities", "stop_on_hit": stop_on_hit,
                "runtime_seconds": round(perf_counter() - started, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    finally:
        if net is not None:
            net.memory.reset_memory()
        if cache is not None:
            cache.clear()
        del net, cache
        gc.collect()
        torch.cuda.empty_cache()
        GPU_LOCK.release()
