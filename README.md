# Crop Stress Prediction from Satellite Data

Predicting vegetation-stress risk 2–4 weeks ahead, from Sentinel-2 satellite imagery and
public weather data, across three countries with very different agriculture: **Afghanistan**,
**the Netherlands**, and **New Zealand**.

Author: **Maiwand Jan Alamzoi** — [m.alamzoi123@gmail.com](mailto:m.alamzoi123@gmail.com) · [github.com/maiwandalamzoi](https://github.com/maiwandalamzoi)

---

## Problem statement

Early warning of crop stress — drought, heat, waterlogging, the early stages of many
diseases — lets farmers and field officers intervene before yield loss is locked in. Free
satellite imagery and public weather data make this feasible almost anywhere in the world,
but there is no public, ground-truthed "this field had crop disease on this date" dataset at
scale. This project builds a defensible **proxy-label** pipeline instead of pretending
otherwise: it detects statistically anomalous vegetation-index dips relative to a location's
own multi-year seasonal baseline — the same anomaly-detection principle operational systems
like USDA VegDRI and FEWS NET use for vegetation condition monitoring — and trains models to
*predict* that anomaly ahead of time from the preceding weeks of satellite + weather signal.

**What this is:** a lead-time predictor for vegetation-stress anomalies (drought/heat/water
stress signatures visible in the vegetation indices), evaluated with a real, honest
train/test split.
**What this is not:** a lab-confirmed plant disease classifier. No leaf-level disease
imagery or pathologist labels are used anywhere in this pipeline — "disease risk" here means
elevated risk of the kind of vegetation stress that often precedes or accompanies disease
pressure, not a diagnosis.

## Data sources

| Source | What | How |
|---|---|---|
| [Sentinel-2 Surface Reflectance (Harmonized)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) | 10 m optical imagery, 2019–2024 | Google Earth Engine, cloud-masked via the Scene Classification (SCL) band, composited into 16-day medians |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Daily max/min temperature, precipitation, humidity, reference evapotranspiration (ERA5 reanalysis) | Free, no API key, 2019–2024 daily |

**36 field locations**, 12 per country, chosen for genuinely different agro-climatic regimes:

- **Afghanistan** — arid irrigated lowland (Helmand, Kandahar, Nimroz), river-valley
  irrigated plains (Nangarhar, Balkh, Herat, Kunduz), rainfed highland wheat belts (Ghazni,
  Badakhshan, Ghor).
- **Netherlands** — intensive temperate arable land and dairy pasture across polder and
  river-clay provinces (Flevoland, Groningen, Friesland, Zeeland, and others).
- **New Zealand** — temperate maritime dairy pasture, arable land, and horticulture across
  the North and South Islands. **Southern Hemisphere** — its growing season is offset ~6
  months from the other two countries; see [Cross-country comparison](#cross-country-comparison).

Full coordinate list: [`src/locations.py`](src/locations.py).

## Method

1. **`src/extract_satellite.py`** — pulls a cloud-masked Sentinel-2 median composite for
   every 16-day period from 2019-01-01 to 2024-12-31, at all 36 locations, and computes
   NDVI, EVI, SAVI, and NDMI (moisture index) via `reduceRegions`.
2. **`src/extract_weather.py`** — pulls daily weather for the same period/locations from
   Open-Meteo and aggregates it to the same 16-day windows (mean temperature, total
   precipitation, mean humidity, total evapotranspiration, heat/dry/frost day counts).
3. **`src/build_features.py`** —
   - **Label**: for each (location, 16-day period-of-year), computes a **leave-one-year-out**
     NDVI baseline (mean, std) from the *other* years of data at that same point and time of
     year, then flags `stress = 1` if the current year's NDVI z-score against that baseline
     is ≤ −1.0 (roughly the bottom ~16% of that location's own seasonal distribution).
   - **Features**: the *previous* period's NDVI/EVI/SAVI/NDMI, the NDVI/EVI trend into that
     period, and weather accumulated over that same prior period — i.e. everything a model
     would actually have in hand 2–4 weeks before the target period, avoiding same-period
     leakage. Plus static context: country, climate zone, latitude/longitude, cyclical
     month-of-year encoding.
4. **`src/train.py`** — trains a **Random Forest** and an **XGBoost** classifier.
   - Time-based split: train on 2019–2022, test on 2023–2024 (no future data leaks into
     training).
   - 5-fold stratified cross-validation on the training set as an independent robustness
     check.
   - `class_weight="balanced"` (RF) / `scale_pos_weight` (XGBoost) to handle the class
     imbalance from the ~21% stress base rate.
5. **`src/evaluate.py`** — confusion matrices, feature importance, and cross-country
   comparison figures, written to `reports/`.
6. **`app_streamlit.py`** — interactive map of predicted stress risk per field, per-location
   NDVI history, and the country comparison.

## Results

Full numbers: [`reports/metrics.json`](reports/metrics.json) and [`reports/summary.md`](reports/summary.md).
Dataset: 3,953 labeled (location, period) rows after feature/label engineering, 21.3% stress class.
**Time-based split** — trained on 2019–2022 (2,728 rows, 21.6% stress), tested on 2023–2024
(1,225 rows, 20.7% stress), so the test set is genuinely unseen future data, not a random
shuffle of the same years.

We lead with **F1 on the stress class**, not raw accuracy: a trivial "always predict no
stress" baseline already scores ~79% accuracy on this imbalanced label and would be useless
for an early-warning system (it would never flag a single real stress event). Accuracy is
reported alongside for context.

| Model | Test accuracy | Test F1 (stress) | Test F1 (macro) | 5-fold CV F1 (train, mean ± std) |
|---|---|---|---|---|
| **Random Forest** (best) | 0.736 | **0.253** | 0.546 | 0.320 ± 0.009 |
| XGBoost | 0.713 | 0.229 | 0.526 | 0.327 ± 0.030 |

Random Forest confusion matrix, test set (2023–2024, n=1,225):

| | Predicted no-stress | Predicted stress |
|---|---|---|
| **Actual no-stress** | 846 | 126 |
| **Actual stress** | 198 | 55 |

That's **precision 0.30** and **recall 0.22** on the stress class — it catches roughly 1 in 5
real stress events 2–4 weeks ahead, with about 3 in 10 stress flags being real. That is a
genuinely modest result, and it is reported as such: predicting a statistical vegetation
anomaly weeks in advance from only satellite-index trajectory and weather is a hard problem,
and this is a first honest baseline, not a tuned production model. See
[`reports/figures/confusion_matrices.png`](reports/figures/confusion_matrices.png).

**Top features** (Random Forest importance — full list in
[`reports/feature_importance_random_forest.csv`](reports/feature_importance_random_forest.csv)):
`NDVI_lag1`, `SAVI_lag1`, `EVI_lag1`, `ndvi_trend`, `humidity_mean`, `NDMI_lag1`, `tmin_mean`,
`et0_sum`, `tmax_mean`, `precip_sum`. Importance is fairly evenly split between the prior
satellite trajectory and the accumulated weather — neither dominates.
See [`reports/figures/feature_importance.png`](reports/figures/feature_importance.png).

## Cross-country comparison

New Zealand's growing season is offset ~6 months from Afghanistan and the Netherlands
(Southern vs. Northern Hemisphere). Because the stress label is computed against each
location's *own* multi-year seasonal baseline — not a shared calendar-month baseline — this
offset does not bias the comparison; a New Zealand field in its December stress dip is
compared to *other Decembers at that same field*, not to Afghan December conditions. See
[`reports/figures/ndvi_seasonality_by_country.png`](reports/figures/ndvi_seasonality_by_country.png)
for the visible 6-month offset in each country's NDVI curve.

**One honest consequence of that design**: overall stress *incidence* across the full
2019–2024 dataset comes out at 21.3% for all three countries, essentially by construction —
a per-location relative-anomaly threshold normalizes away most of the raw difference in how
"stressed" each country's vegetation looks. So raw incidence isn't the interesting
cross-country signal here; **predictability is**. Breaking the Random Forest's test-set
performance out by country ([`reports/figures/stress_rate_by_country.png`](reports/figures/stress_rate_by_country.png)):

| Country | Test rows | Stress rate (test) | Accuracy | F1 (stress) |
|---|---|---|---|---|
| Afghanistan | 507 | 25.6% | 0.746 | **0.332** |
| New Zealand | 393 | 15.8% | 0.720 | 0.214 |
| Netherlands | 325 | 18.8% | 0.738 | 0.158 |

Afghan field stress is the most predictable from the prior period's satellite trajectory and
weather — plausible given Afghanistan's arid/rainfed systems, where a dry, hot period tends to
be followed by a continuation of the same drought signature (persistent, autocorrelated
stress). Dutch fields are the least predictable here, plausibly because intensive irrigation
and drainage decouple short-term vegetation condition from the raw weather signal, and because
denser Sentinel-2 cloud cover over the Netherlands leaves noisier composites. This is a
read of a modest signal from 12 locations per country over 6 years, not a definitive claim
about either country's agriculture.

## Reproduce it

```bash
git clone https://github.com/maiwandalamzoi/crop-stress-prediction.git
cd crop-stress-prediction
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Earth Engine auth: copy .env.example to .env and fill in a service account,
# or run `earthengine authenticate` once for cached user credentials.
cp .env.example .env

python src/extract_satellite.py   # ~20-25 min: pulls Sentinel-2 composites via GEE
python src/extract_weather.py     # ~2-3 min: pulls Open-Meteo daily weather
python src/build_features.py      # joins, labels, engineers lag features
python src/train.py               # trains RF + XGBoost, writes reports/metrics.json
python src/evaluate.py            # writes confusion matrices & comparison figures

streamlit run app_streamlit.py    # interactive map + charts
```

Raw and processed data (`data/raw/*.csv`, `data/processed/features.csv`) and the trained
models (`models/*.joblib`) are committed to this repo, so you can skip straight to
`streamlit run app_streamlit.py` to explore the results without re-running extraction.

## Deploy the Streamlit app

`app_streamlit.py` reads only the committed CSVs and `.joblib` models — it never calls Earth
Engine or Open-Meteo live, so it needs no secrets to deploy. To host it for free on
[Streamlit Community Cloud](https://share.streamlit.io):

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with the
   `maiwandalamzoi` GitHub account.
2. Click **"New app"**.
3. Repository: `maiwandalamzoi/crop-stress-prediction`, branch: `main`,
   main file path: `app_streamlit.py`.
4. Under **"Advanced settings"**, set the requirements file to
   [`requirements-app.txt`](requirements-app.txt) instead of `requirements.txt` — it skips
   `earthengine-api`/`matplotlib`/`requests`, which the app itself never imports, for a
   noticeably faster build. (`requirements.txt` also works, just slower to build.)
5. Click **Deploy**. Build takes a couple of minutes.
6. You'll get a public URL (`https://<app-name>.streamlit.app`) to share.

## Limitations

- The stress label is a **statistical proxy** (NDVI anomaly), not a ground-truthed disease or
  agronomic diagnosis — see [Problem statement](#problem-statement).
- 36 point locations with a ~60 m sampling radius are a sparse sample of each country's
  agriculture, not a wall-to-wall map — this is a methodology demonstration, not a
  production monitoring system.
- Sentinel-2 cloud cover (especially over the Netherlands and New Zealand) creates gaps in
  the 16-day composites; periods with too few valid pixels are dropped rather than
  interpolated.
- 6 years of history (2019–2024) gives a workable but not large per-location baseline for
  the leave-one-year-out anomaly calculation.

## License

MIT — see [LICENSE](LICENSE).
