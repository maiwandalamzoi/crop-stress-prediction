"""
Pull Sentinel-2 vegetation-index time series for every location in
locations.py, from Google Earth Engine.

For each 16-day period between START_DATE and END_DATE, builds a
cloud-masked median composite of Sentinel-2 Surface Reflectance
(COPERNICUS/S2_SR_HARMONIZED), computes NDVI/EVI/SAVI/NDMI, and samples the
composite at every field point (mean over a small buffer + valid-pixel
count). Writes one row per (location, period).

Usage:
    python src/extract_satellite.py
"""
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import ee

from gee_auth import init_ee
from locations import LOCATIONS, FIELD_BUFFER_M

START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)
PERIOD_DAYS = 16
SCALE_M = 20

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "sentinel2_timeseries.csv"

BANDS = ["NDVI", "EVI", "SAVI", "NDMI"]
FIELDS = ["name", "country", "region", "lat", "lon", "climate_zone",
          "period_start", "period_end"] + BANDS + ["pixel_count"]


def mask_and_index(img):
    scl = img.select("SCL")
    # Keep vegetation(4), not-vegetated(5), water(6), unclassified(7).
    # Drop cloud shadow(3), cloud medium(8)/high(9) prob, cirrus(10), snow(11).
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    img = img.updateMask(good)

    b2 = img.select("B2").divide(10000)
    b4 = img.select("B4").divide(10000)
    b8 = img.select("B8").divide(10000)
    b11 = img.select("B11").divide(10000)

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    evi = b8.subtract(b4).multiply(2.5).divide(
        b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
    ).rename("EVI")
    savi = b8.subtract(b4).divide(b8.add(b4).add(0.5)).multiply(1.5).rename("SAVI")
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename("NDMI")

    return img.addBands([ndvi, evi, savi, ndmi])


def build_points_fc():
    feats = []
    for name, country, region, lat, lon, zone in LOCATIONS:
        geom = ee.Geometry.Point([lon, lat]).buffer(FIELD_BUFFER_M)
        feats.append(ee.Feature(geom, {
            "name": name, "country": country, "region": region,
            "lat": lat, "lon": lon, "climate_zone": zone,
        }))
    return ee.FeatureCollection(feats)


def period_ranges():
    d = START_DATE
    while d < END_DATE:
        end = min(d + timedelta(days=PERIOD_DAYS), END_DATE)
        yield d, end
        d = end


def reduce_period(collection, points_fc, start, end, retries=3):
    filtered = collection.filterDate(str(start), str(end))
    for attempt in range(retries):
        try:
            if filtered.size().getInfo() == 0:
                return []
            composite = filtered.map(mask_and_index).select(BANDS).median()
            reducer = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)
            reduced = composite.reduceRegions(collection=points_fc, reducer=reducer, scale=SCALE_M)
            return reduced.getInfo()["features"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! failed {start}..{end} after {retries} attempts: {e}", file=sys.stderr)
                return []
            time.sleep(3 * (attempt + 1))


def main():
    init_ee()
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    points_fc = build_points_fc()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    periods = list(period_ranges())
    print(f"Extracting {len(periods)} periods x {len(LOCATIONS)} locations...")

    rows_written = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for i, (start, end) in enumerate(periods):
            feats = reduce_period(collection, points_fc, start, end)
            for feat in feats:
                p = feat["properties"]
                ndvi_count = p.get("NDVI_count")
                if not ndvi_count:  # no cloud-free pixels this period at this point
                    continue
                row = {
                    "name": p["name"], "country": p["country"], "region": p["region"],
                    "lat": p["lat"], "lon": p["lon"], "climate_zone": p["climate_zone"],
                    "period_start": str(start), "period_end": str(end),
                    "pixel_count": ndvi_count,
                }
                for b in BANDS:
                    row[b] = p.get(f"{b}_mean")
                writer.writerow(row)
                rows_written += 1

            if (i + 1) % 10 == 0 or i == len(periods) - 1:
                f.flush()
                print(f"  [{i+1}/{len(periods)}] periods done, {rows_written} rows so far")

    print(f"Done. Wrote {rows_written} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
