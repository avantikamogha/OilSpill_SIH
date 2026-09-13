from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "processed" / "candidate_tracks_sorted.csv"
OUTPUT = BASE / "data" / "processed" / "vessel_gaps.csv"
# LOAD DATA
if not INPUT.exists():
    raise FileNotFoundError(f"File not found: {INPUT}")
df = pd.read_csv(INPUT)

# CONVERT TIMESTAMP
df["base_date_time"] = pd.to_datetime(
    df["base_date_time"],
    utc=True,
    errors="coerce"
)

# SORT
df = df.sort_values(
    ["mmsi", "base_date_time"]
).reset_index(drop=True)

# CALCULATE TIME DIFFERENCE
df["time_diff_seconds"] = (
    df.groupby("mmsi")["base_date_time"]
    .diff()
    .dt.total_seconds()
)

# IDENTIFY AIS GAPS
df["ais_gap"] = (
    df["time_diff_seconds"] > 600
)

# GAP DURATION IN MINUTES
df["gap_minutes"] = (
    df["time_diff_seconds"] / 60
)
# CREATE GAP TABLE
gaps = df[
    df["ais_gap"]
].copy()

gaps["gap_end_time"] = gaps["base_date_time"]

gaps["gap_start_time"] = (
    gaps["gap_end_time"]
    - pd.to_timedelta(
        gaps["time_diff_seconds"],
        unit="s"
    )
)

gaps = gaps[
    [
        "mmsi",
        "gap_start_time",
        "gap_end_time",
        "time_diff_seconds",
        "gap_minutes",
        "latitude",
        "longitude",
        "sog",
        "cog",
        "heading"
    ]
]
# SAVE
gaps.to_csv(
    OUTPUT,
    index=False
)
# SUMMARY
print("AIS gap analysis complete.")
print(f"Total gaps > 10 minutes: {len(gaps):,}")

if len(gaps) > 0:
    print(
        f"Longest gap: "
        f"{gaps['gap_minutes'].max():.2f} minutes"
    )

print(f"Output: {OUTPUT}")