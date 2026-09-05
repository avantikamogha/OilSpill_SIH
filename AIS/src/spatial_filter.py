from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_FILE = PROCESSED_DIR / "ais_clean.csv"

print("Project folder:", PROJECT_DIR)
print("Input file:", INPUT_FILE)
print("Output folder:", OUTPUT_DIR)
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nClean AIS file not found:\n{INPUT_FILE}\n\n"
        "Run cleaning.py first."
    )
print("\nLoading cleaned AIS data...")

ais = pd.read_csv(INPUT_FILE)

print("Loaded successfully.")
print("Rows:", len(ais))
print("Columns:", len(ais.columns))
print("\nLoading cleaned AIS data...")

ais = pd.read_csv(INPUT_FILE)

print("Loaded successfully.")
print("Rows:", len(ais))
print("Columns:", len(ais.columns))
ais_test = ais.head(50000).copy()
geometry = gpd.points_from_xy(
    ais_test["longitude"],
    ais_test["latitude"]
)