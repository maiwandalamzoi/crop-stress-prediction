"""
Earth Engine authentication helper.

Tries, in order:
1. GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY environment variables (service
   account JSON key, private key with literal \\n newlines allowed).
2. A local .env file in the project root with the same two keys.
3. Previously-cached user credentials (`earthengine authenticate`).

See .env.example for the expected format.
"""
import os
from pathlib import Path

import ee

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def init_ee(verbose: bool = True) -> None:
    env = {**_load_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    sa = env.get("GEE_SERVICE_ACCOUNT")
    key = env.get("GEE_PRIVATE_KEY")

    if sa and key:
        key = key.replace("\\n", "\n")
        creds = ee.ServiceAccountCredentials(sa, key_data=key)
        ee.Initialize(creds)
        if verbose:
            print(f"Earth Engine initialized with service account: {sa}")
        return

    # Fall back to whatever credentials `earthengine authenticate` cached.
    ee.Initialize()
    if verbose:
        print("Earth Engine initialized with cached user credentials")
