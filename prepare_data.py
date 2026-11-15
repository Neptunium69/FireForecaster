from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

root=Path(__file__).resolve().parent
input_dir=root / "data/raw/ts-satfire"
output_dir=root / "data"
history_days=5

aux_bands={
    "NDVI_last": 10000, "EVI2_last": 10000, "total precipitation": 100,
    "wind speed": 30, "wind direction": 360, "minimum temperature": 350,
    "maximum temperature": 350, "energy release component": 100,
    "specific humidity": 0.03, "slope": 90, "aspect": 360,
    "elevation": 5000, "pdsi": 10, "LC_Type1": 17,
    "total_precipitation_surface_sum": 0.1, "forecast wind speed": 30,
    "forecast wind direction": 360, "forecast temperature": 50,
    "forecast specific humidity": 0.03,
}
scales=np.array([100, 100, 100, 400, 400, 100, 400, 400,
                 *aux_bands.values()], dtype=np.float32)[:, None, None]


def has_fire_observation(valid):
    """Check the input validity mask for VIIRS I4 and both fire-map channels."""
    return (valid[3].any() or valid[6].any()) and valid[27].any() and valid[28].any()


class FireDataset:
    def __init__(self, folder, *, require_daily=True):
        self.folder=folder
        self.files=sorted((folder / "VIIRS_Day").glob("*.tif"))
        self.dates=np.array([p.name[:10] for p in self.files], dtype="datetime64[D]")
        if require_daily and not np.all(np.diff(self.dates)==np.timedelta64(1, "D")):
            raise ValueError(f"{folder.name}: split missing-day gaps into separate sequences.")
        with rasterio.open(self.files[0]) as src:
            self.crs=src.crs
            self.transform=src.transform
            self.height, self.width=src.shape

        # TS-SatFire day bands 7/8 contain AF detections / available BA perimeters.
        # Cache only small binary maps; spectral/weather TIFFs are read on demand.
        self.active=[]
        self.burned=[]
        cumulative=np.zeros((self.height, self.width), dtype=bool)
        for path in self.files:
            masks, _=self.read_tiff(path, [7, 8], masks=True)
            active, perimeter=masks>0
            cumulative=cumulative | active | perimeter
            self.active.append(active)
            self.burned.append(cumulative)

    def read_tiff(self, path, indexes=None, masks=False):
        with rasterio.open(path) as src:
            # Three downloaded night TIFFs have five bands; the last two are I4/I5.
            if indexes=="night":
                indexes=list(range(src.count-1, src.count+1))
            indexes=indexes or list(src.indexes)
            data=src.read(indexes, masked=True).astype(np.float32).filled(np.nan)
            if masks:
                data=(np.isfinite(data) & (data>0)).astype(np.float32)
            else:
                data[~np.isfinite(data)]=np.nan
            names=[src.descriptions[i-1] for i in indexes]
            if src.crs==self.crs and src.transform==self.transform and src.shape==(self.height, self.width):
                return data, names
            aligned=np.full((len(indexes), self.height, self.width), np.nan, dtype=np.float32)
            reproject(data, aligned, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=self.transform, dst_crs=self.crs,
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.max if masks else Resampling.nearest)
            return aligned, names

    def read_day(self, index):
        path=self.files[index]
        date=path.name[:10]
        day, _=self.read_tiff(path, [1, 2, 3, 4, 5, 6])
        night_path=self.folder / "VIIRS_Night" / f"{date}_VIIRS_Night.tif"
        night=(self.read_tiff(night_path, "night")[0] if night_path.exists()
               else np.full((2, self.height, self.width), np.nan, dtype=np.float32))
        weather=np.full((19, self.height, self.width), np.nan, dtype=np.float32)
        weather_path=self.folder / "FirePred" / f"{date}_FirePred.tif"
        if weather_path.exists():
            values, names=self.read_tiff(weather_path)
            names=[name.replace("total_precipitation_surface_last",
                                "total_precipitation_surface_sum") for name in names]
            for i, name in enumerate(aux_bands):
                if name in names:
                    weather[i]=values[names.index(name)]
        data=np.concatenate([day, night, weather])
        data=data/scales
        # Only observations from this history day are supplied as fire-state inputs.
        data=np.concatenate([data,
                             self.active[index][None], self.burned[index][None]])
        return data


def convert_fire(folder):
    files=sorted((folder / "VIIRS_Day").glob("*.tif"))
    output=output_dir / folder.name
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(files[0]) as src:
        shape=src.shape
    dataset=None
    for i, path in enumerate(files):
        destination=output / f"{path.name[:10]}.npz"
        if destination.exists():
            with np.load(destination) as saved:
                if "image" in saved and "native_shape" in saved and np.array_equal(saved["native_shape"], shape):
                    continue
        if dataset is None:
            dataset=FireDataset(folder)
        newly_burned=(dataset.burned[i] & ~dataset.burned[i-1]) if i else dataset.burned[i]
        temporary=destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,
                                image=dataset.read_day(i),
                                date=dataset.dates[i].astype(np.int64),
                                active=dataset.active[i][None],
                                burned=dataset.burned[i][None],
                                new=newly_burned[None],
                                native_shape=np.array(shape))
        temporary.replace(destination)
    print(f"{folder.name}: {len(files)} days at {shape[0]}x{shape[1]}", flush=True)
    return len(files)


if __name__=="__main__":
    folders=sorted(folder for folder in input_dir.iterdir()
                   if len(list((folder / "VIIRS_Day").glob("*.tif")))>history_days)
    with ThreadPoolExecutor(max_workers=4) as pool:
        total=sum(pool.map(convert_fire, folders))
    print(f"Finished: {total} daily NPZ files, {total-history_days*len(folders)} training windows, {len(folders)} fires.", flush=True)
