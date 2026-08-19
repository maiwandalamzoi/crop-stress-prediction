"""
Field locations used for satellite/weather extraction.

36 points across three countries chosen for genuinely different agro-climatic
regimes, to support both a single stress-prediction model and a cross-country
comparison of vegetation stress patterns:

  - Afghanistan (12 pts): arid irrigated lowland, semi-arid plains, and
    rainfed highland wheat belts. Northern Hemisphere growing season.
  - Netherlands (12 pts): intensive temperate arable land and dairy pasture,
    reclaimed polder and river-clay soils. Northern Hemisphere.
  - New Zealand (12 pts): temperate maritime dairy pasture, arable, and
    horticulture. Southern Hemisphere -> growing season is offset ~6 months
    from the other two. This is fine for the anomaly method used downstream
    (each point's stress baseline is computed against its own history), but
    matters when comparing calendar months directly across countries.

Coordinates are representative farmland near each named locality (not exact
field boundaries), picked from public geography.
"""

LOCATIONS = [
    # name, country, region, lat, lon, climate_zone
    # --- Afghanistan (12) ---
    ("Shakardara",   "Afghanistan", "Kabul",      34.65,  69.03, "highland_irrigated"),
    ("Kama",         "Afghanistan", "Nangarhar",  34.42,  70.55, "lowland_irrigated"),
    ("Dawlatabad",   "Afghanistan", "Balkh",      36.85,  66.85, "plains_irrigated"),
    ("Injil",        "Afghanistan", "Herat",      34.28,  62.15, "plains_irrigated"),
    ("Arghandab",    "Afghanistan", "Kandahar",   31.72,  65.65, "arid_irrigated"),
    ("Khanabad",     "Afghanistan", "Kunduz",     36.65,  68.95, "plains_irrigated"),
    ("Faizabad",     "Afghanistan", "Badakhshan", 37.10,  70.55, "highland_rainfed"),
    ("NadAli",       "Afghanistan", "Helmand",    31.55,  64.25, "arid_irrigated"),
    ("GhazniCity",   "Afghanistan", "Ghazni",     33.55,  68.42, "highland_rainfed"),
    ("Maimana",      "Afghanistan", "Faryab",     35.92,  64.78, "plains_rainfed"),
    ("Chaghcharan",  "Afghanistan", "Ghor",       34.52,  65.25, "highland_rainfed"),
    ("Zaranj",       "Afghanistan", "Nimroz",     31.00,  61.87, "arid_irrigated"),

    # --- Netherlands (12) ---
    ("Dronten",      "Netherlands", "Flevoland",       52.53,   5.72, "temperate_arable"),
    ("Veendam",      "Netherlands", "Groningen",       53.10,   6.87, "temperate_arable"),
    ("Sneek",        "Netherlands", "Friesland",       53.03,   5.65, "temperate_dairy_pasture"),
    ("Goes",         "Netherlands", "Zeeland",         51.50,   3.90, "temperate_arable"),
    ("Uden",         "Netherlands", "Noord-Brabant",   51.66,   5.62, "temperate_dairy_pasture"),
    ("Tiel",         "Netherlands", "Gelderland",      51.90,   5.60, "temperate_horticulture"),
    ("Almelo",       "Netherlands", "Overijssel",      52.35,   6.60, "temperate_dairy_pasture"),
    ("Venlo",        "Netherlands", "Limburg",         51.37,   6.17, "temperate_horticulture"),
    ("Wieringermeer","Netherlands", "Noord-Holland",   52.85,   5.00, "temperate_arable"),
    ("Emmen",        "Netherlands", "Drenthe",         52.75,   6.85, "temperate_arable"),
    ("Bunnik",       "Netherlands", "Utrecht",         52.05,   5.20, "temperate_dairy_pasture"),
    ("Alblasserdam", "Netherlands", "Zuid-Holland",    51.87,   4.67, "temperate_dairy_pasture"),

    # --- New Zealand (12) ---
    ("Ashburton",    "New Zealand", "Canterbury",   -43.90, 171.75, "maritime_arable"),
    ("Hamilton",     "New Zealand", "Waikato",      -37.78, 175.28, "maritime_dairy_pasture"),
    ("Gore",         "New Zealand", "Southland",    -46.10, 168.95, "maritime_dairy_pasture"),
    ("PalmerstonN",  "New Zealand", "Manawatu",     -40.35, 175.61, "maritime_dairy_pasture"),
    ("Hastings",     "New Zealand", "HawkesBay",    -39.64, 176.84, "maritime_horticulture"),
    ("Balclutha",    "New Zealand", "Otago",        -46.23, 169.75, "maritime_arable"),
    ("TePuke",       "New Zealand", "BayOfPlenty",  -37.78, 176.32, "maritime_horticulture"),
    ("Stratford",    "New Zealand", "Taranaki",     -39.34, 174.28, "maritime_dairy_pasture"),
    ("Blenheim",     "New Zealand", "Marlborough",  -41.51, 173.96, "maritime_horticulture"),
    ("Masterton",    "New Zealand", "Wairarapa",    -40.96, 175.66, "maritime_arable"),
    ("Kerikeri",     "New Zealand", "Northland",    -35.22, 173.95, "maritime_horticulture"),
    ("Motueka",      "New Zealand", "Tasman",       -41.11, 173.00, "maritime_horticulture"),
]

FIELD_BUFFER_M = 60  # radius of the sampled area around each point, meters
