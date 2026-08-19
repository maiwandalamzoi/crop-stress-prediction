"""
Pull daily historical weather for every location in locations.py from the
Open-Meteo Historical Weather (ERA5-based reanalysis) API — free, no API key.
https://open-meteo.com/en/docs/historical-weather-api

Writes one row per (location, date) with daily max/min temperature,
precipitation, mean relative humidity, and reference evapotranspiration.

Usage:
    python src/extract_weather.py
"""
import csv
import time
from pathlib import Path

import requests

from locations import LOCATIONS

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "et0_fao_evapotranspiration",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "weather_timeseries.csv"
FIELDS = ["name", "country", "date"] + DAILY_VARS


def fetch_location(lat, lon, retries=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()["daily"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry ({e})")
            time.sleep(20 * (attempt + 1))


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for name, country, region, lat, lon, zone in LOCATIONS:
            print(f"Fetching weather for {name}, {country} ({lat}, {lon})...")
            daily = fetch_location(lat, lon)
            dates = daily["time"]
            for i, d in enumerate(dates):
                row = {"name": name, "country": country, "date": d}
                for var in DAILY_VARS:
                    row[var] = daily[var][i]
                writer.writerow(row)
                rows_written += 1
            f.flush()
            time.sleep(2)  # be polite to the free public API

    print(f"Done. Wrote {rows_written} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
