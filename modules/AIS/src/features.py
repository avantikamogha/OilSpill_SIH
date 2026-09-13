from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "processed" / "candidate_tracks_sorted.csv"
OUTPUT = BASE / "data" / "processed" / "vessel_features.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"File not found: {INPUT}")

df = pd.read_csv(INPUT)

df["base_date_time"] = pd.to_datetime(
    df["base_date_time"],
    utc=True,
    errors="coerce"
)

df = df.sort_values(
    ["slick_id", "mmsi", "base_date_time"]
)

group = ["slick_id", "mmsi"]

df["time_diff_seconds"] = (
    df.groupby(group)["base_date_time"]
    .diff()
    .dt.total_seconds()
)

df["ais_gap"] = df["time_diff_seconds"] > 600

df["stopped"] = df["sog"].fillna(0) <= 0.5

features = (
    df.groupby(group)
    .agg(
        observations=("mmsi", "size"),
        first_seen=("base_date_time", "min"),
        last_seen=("base_date_time", "max"),
        min_sog=("sog", "min"),
        max_sog=("sog", "max"),
        mean_sog=("sog", "mean"),
        mean_cog=("cog", "mean"),
        mean_lat=("latitude", "mean"),
        mean_lon=("longitude", "mean"),
        ais_gap_count=("ais_gap", "sum"),
        max_gap_minutes=(
            "time_diff_seconds",
            lambda x: x.max() / 60 if x.notna().any() else 0
        ),
        stopped_observations=("stopped", "sum"),
        vessel_name=("vessel_name", "first"),
        vessel_type=("vessel_type", "first"),
        imo=("imo", "first"),
        length=("length", "first"),
        width=("width", "first"),
        draft=("draft", "first")
    )
    .reset_index()
)

features["stopped_ratio"] = (
    features["stopped_observations"] /
    features["observations"]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT,
    index=False
)

print("Vessel feature extraction complete.")
print(f"Candidate vessel-slick records: {len(features):,}")
print(f"Unique candidate vessels: {features['mmsi'].nunique():,}")
print(f"Output: {OUTPUT}")