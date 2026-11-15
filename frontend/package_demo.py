"""Package the first five available raw days; never include future observations."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
SOURCE = ROOT / "data/raw/ts-satfire/thomas_fire"
DESTINATION = APP / "demo/thomas_fire_first5"


def main():
    days = sorted((SOURCE / "VIIRS_Day").glob("*.tif"))[:5]
    if len(days) != 5:
        raise ValueError("Thomas Fire needs five daily source files.")
    dates = [p.name[:10] for p in days]
    files = []
    for kind in ("VIIRS_Day", "VIIRS_Night", "FirePred"):
        (DESTINATION / kind).mkdir(parents=True, exist_ok=True)
        for day in dates:
            source = SOURCE / kind / f"{day}_{kind}.tif"
            target = DESTINATION / kind / source.name
            shutil.copy2(source, target)
            files.append({"path": str(target.relative_to(DESTINATION)),
                          "bytes": target.stat().st_size,
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    points = [
        ("Carpinteria community", 34.3989, -119.5185, 9, 2),
        ("Montecito community", 34.4367, -119.6321, 10, 2),
        ("Ojai community", 34.4480, -119.2429, 8, 2),
        ("Lake Casitas eastern shore", 34.4160, -119.3370, 6, 2),
    ]
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
         "properties": {"name": name, "value": value, "radius_km": radius,
                        "note": "Approximate demonstration location; illustrative priority and buffer."}}
        for name, lat, lon, value, radius in points
    ]}
    (DESTINATION / "points.geojson").write_text(json.dumps(geojson, indent=2))
    (DESTINATION / "manifest.json").write_text(json.dumps({
        "fire": "Thomas Fire", "dataset": "TS-SatFire", "dates": dates,
        "native_shape": [596, 594], "files": files,
        "split": "validation (demonstration, not an unseen test)",
        "source": "https://github.com/zhaoyutim/TS-SatFire",
        "selection_reason": "Five initial days precede further westward spread toward Carpinteria and Montecito. Includes communities and reservoir recreation land for varied custom priorities.",
        "historical_context": "https://carpinteriaca.gov/wp-content/uploads/2026/03/6_Safety_CLUP-GP_Revised_Feb2026.pdf",
        "poi_note": "Coordinates are approximate community/landmark demonstration points, not surveyed asset boundaries. Values are editable demo assumptions, not monetary values or official priorities.",
        "forecast_note": "No guarantee the learned model forecasts actual historical impacts. Data after December 8 is deliberately excluded.",
    }, indent=2))
    (DESTINATION / "README.txt").write_text(
        "Thomas Fire: first five available TS-SatFire days, 2017-12-04 through 2017-12-08.\n"
        "15 original, unmodified GeoTIFFs at native resolution. No future days or prepared NPZs.\n"
        "Upload this folder or its ZIP in the app. points.geojson supplies four editable demo locations.\n"
        "All four center pixels are unburned in the day-five source map; buffers may contain burned pixels.\n"
        "The area includes communities and reservoir recreation land, allowing different protection priorities.\n"
        "Approximate coordinates and value scores are demo assumptions, not surveyed facilities or official priorities.\n"
        "This is a validation fire seen during development, not an unseen performance evaluation.\n"
        "See manifest.json for source checksums and historical context.\n"
    )
    archive = DESTINATION.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(DESTINATION.parent))
    print(f"Packaged {len(files)} TIFFs: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
