from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Geod

BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "processed" / "candidate_tracks.csv"
OUTPUT = BASE / "data" / "processed" / "candidate_tracks_sorted.csv"

if not INPUT.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT}"
    )

df = pd.read_csv(INPUT)

df["base_date_time"] = pd.to_datetime(
    df["base_date_time"],
    utc=True,
    errors="coerce"
)

df = df.sort_values(
    ["slick_id", "mmsi", "base_date_time"]
).reset_index(drop=True)

group = [
    "slick_id",
    "mmsi"
]

df["prev_lat"] = (
    df.groupby(group)["latitude"].shift()
)

df["prev_lon"] = (
    df.groupby(group)["longitude"].shift()
)

df["prev_time"] = (
    df.groupby(group)["base_date_time"].shift()
)

df["time_diff_seconds"] = (
    df["base_date_time"] -
    df["prev_time"]
).dt.total_seconds()

geod = Geod(ellps="WGS84")

valid = (
    df["prev_lat"].notna() &
    df["prev_lon"].notna() &
    df["time_diff_seconds"].notna() &
    (df["time_diff_seconds"] > 0)
)

df["distance_m"] = np.nan
df["movement_bearing"] = np.nan

az1, _, dist = geod.inv(
    df.loc[valid, "prev_lon"].to_numpy(),
    df.loc[valid, "prev_lat"].to_numpy(),
    df.loc[valid, "longitude"].to_numpy(),
    df.loc[valid, "latitude"].to_numpy()
)

df.loc[valid, "distance_m"] = dist
df.loc[valid, "movement_bearing"] = (
    az1 + 360
) % 360

df["trajectory_start"] = (
    df["time_diff_seconds"].isna()
)

df["segment_speed_knots"] = np.nan

df.loc[valid, "segment_speed_knots"] = (
    df.loc[valid, "distance_m"] /
    df.loc[valid, "time_diff_seconds"] *
    1.94384
)

df.to_csv(
    OUTPUT,
    index=False
)

print("Trajectory reconstruction complete.")
print(f"Observations: {len(df):,}")
print(f"Vessels: {df['mmsi'].nunique():,}")
print(
    f"Candidate trajectories: "
    f"{df[group].drop_duplicates().shape[0]:,}"
)
print(f"Output: {OUTPUT}")