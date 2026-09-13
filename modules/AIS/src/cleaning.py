from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Geod

BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "raw" / "synthetic_ais_india_oilspill.csv"
OUTPUT = BASE / "data" / "processed" / "ais_clean.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

before = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before - len(df)

df["base_date_time"] = pd.to_datetime(
    df["base_date_time"],
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["base_date_time"])

df = df[
    df["latitude"].between(-90, 90) &
    df["longitude"].between(-180, 180)
].copy()

df["same_mmsi_timestamp"] = df.duplicated(
    subset=["mmsi", "base_date_time"],
    keep=False
)

df = df.sort_values(
    ["mmsi", "base_date_time"]
).reset_index(drop=True)

df["prev_lat"] = df.groupby("mmsi")["latitude"].shift()
df["prev_lon"] = df.groupby("mmsi")["longitude"].shift()
df["prev_time"] = df.groupby("mmsi")["base_date_time"].shift()

df["time_diff_seconds"] = (
    df["base_date_time"] - df["prev_time"]
).dt.total_seconds()

valid = (
    df["prev_lat"].notna() &
    df["prev_lon"].notna() &
    df["time_diff_seconds"].notna() &
    (df["time_diff_seconds"] > 0)
)

geod = Geod(ellps="WGS84")

df["distance_m"] = np.nan

_, _, distance = geod.inv(
    df.loc[valid, "prev_lon"].to_numpy(),
    df.loc[valid, "prev_lat"].to_numpy(),
    df.loc[valid, "longitude"].to_numpy(),
    df.loc[valid, "latitude"].to_numpy()
)

df.loc[valid, "distance_m"] = distance

df["implied_speed_knots"] = np.nan

df.loc[valid, "implied_speed_knots"] = (
    df.loc[valid, "distance_m"] /
    df.loc[valid, "time_diff_seconds"] *
    1.94384
)

df["movement_anomaly"] = (
    df["implied_speed_knots"] > 50
)

df["zero_time_gap"] = (
    df["time_diff_seconds"] == 0
)

df["ais_gap"] = (
    df["time_diff_seconds"] > 600
)

df["valid_movement"] = (
    (df["time_diff_seconds"] > 0) &
    (~df["movement_anomaly"])
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("AIS cleaning complete.")
print(f"Input rows: {before:,}")
print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Clean rows: {len(df):,}")
print(f"Vessels: {df['mmsi'].nunique():,}")
print(f"AIS gaps >10 min: {df['ais_gap'].sum():,}")
print(f"Movement anomalies: {df['movement_anomaly'].sum():,}")
print(f"Output: {OUTPUT}")