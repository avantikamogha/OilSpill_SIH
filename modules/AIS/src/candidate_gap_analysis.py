from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "processed" / "candidate_tracks_sorted.csv"
OUTPUT = BASE / "data" / "processed" / "candidate_gaps.csv"

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
).reset_index(drop=True)

df["time_diff_seconds"] = (
    df.groupby(
        ["slick_id", "mmsi"]
    )["base_date_time"]
    .diff()
    .dt.total_seconds()
)

df["ais_gap"] = df["time_diff_seconds"] > 600
df["gap_minutes"] = df["time_diff_seconds"] / 60

gaps = df[df["ais_gap"]].copy()

columns = [
    "slick_id",
    "mmsi",
    "base_date_time",
    "time_diff_seconds",
    "gap_minutes",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "heading",
    "vessel_name",
    "vessel_type"
]

gaps[columns].to_csv(
    OUTPUT,
    index=False
)

print("Candidate AIS gap analysis complete.")
print(f"Candidate gaps >10 minutes: {len(gaps):,}")
print(f"Candidate vessels with gaps: {gaps['mmsi'].nunique():,}")
print(f"Output: {OUTPUT}")