# ============================================================
# AIS DATA CLEANING & MOVEMENT VALIDATION
# ============================================================

from pathlib import Path
import os

import pandas as pd
import numpy as np
from pyproj import Geod


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Location of this file:
# OilSpill_SIH/
# └── AIS/
#     └── src/
#         └── cleaning.py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("AIS DATA CLEANING")
print("=" * 60)

print("Current folder:", os.getcwd())
print("Project folder:", PROJECT_DIR)
print("Raw data folder:", RAW_DIR)
print("Processed data folder:", PROCESSED_DIR)


# ============================================================
# 2. CHECK RAW DATA FOLDER
# ============================================================

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"\nRaw data folder not found:\n{RAW_DIR}\n\n"
        "Expected structure:\n"
        "AIS/\n"
        "├── data/\n"
        "│   └── raw/\n"
        "└── src/\n"
        "    └── cleaning.py"
    )


raw_files = list(RAW_DIR.iterdir())

print("\nFiles in raw data:")
for file in raw_files:
    print(" -", file.name)


# ============================================================
# 3. LOCATE AIS DATA FILE
# ============================================================

# Your actual file is:
# AIS/data/raw/ais-2025-01-01
#
# It has no .csv extension, so we use the exact filename.

INPUT_FILE = RAW_DIR / "ais-2025-01-01"


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nAIS input file not found:\n{INPUT_FILE}\n\n"
        "Check the filename inside:\n"
        f"{RAW_DIR}"
    )


print("\nInput file:", INPUT_FILE)


# ============================================================
# 4. LOAD AIS DATA
# ============================================================

print("\nLoading AIS data...")

df = pd.read_csv(INPUT_FILE)

print("Data loaded successfully.")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# ============================================================
# 5. INITIAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("INITIAL DATA QUALITY")
print("=" * 60)


print("\nMissing values:")
print(df.isna().sum())


print(
    "\nExact duplicate rows:",
    df.duplicated().sum()
)


print(
    "Duplicate MMSI + timestamp:",
    df.duplicated(
        subset=["mmsi", "base_date_time"]
    ).sum()
)


# ============================================================
# 6. COORDINATE VALIDATION
# ============================================================

invalid_coords = (
    ~df["latitude"].between(-90, 90) |
    ~df["longitude"].between(-180, 180)
)

print(
    "\nInvalid coordinates:",
    invalid_coords.sum()
)


print("\nLatitude:")
print(df["latitude"].describe())


print("\nLongitude:")
print(df["longitude"].describe())


# ============================================================
# 7. REMOVE EXACT DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("CLEANING")
print("=" * 60)


before_duplicates = len(df)

df = df.drop_duplicates().copy()

after_duplicates = len(df)

print(
    "\nExact duplicate rows removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 8. CONVERT TIMESTAMP TO UTC
# ============================================================

df["base_date_time"] = pd.to_datetime(
    df["base_date_time"],
    utc=True,
    errors="coerce"
)


invalid_timestamps = df["base_date_time"].isna().sum()

print(
    "Invalid timestamps:",
    invalid_timestamps
)


# Remove records where timestamp conversion failed
df = df.dropna(
    subset=["base_date_time"]
).copy()


# ============================================================
# 9. VALIDATE GEOGRAPHIC COORDINATES
# ============================================================

valid_coordinates = (
    df["latitude"].between(-90, 90) &
    df["longitude"].between(-180, 180)
)

invalid_coordinate_rows = (~valid_coordinates).sum()

print(
    "Invalid coordinate rows removed:",
    invalid_coordinate_rows
)


df = df[
    valid_coordinates
].copy()


# ============================================================
# 10. VALIDATE MMSI
# ============================================================

invalid_mmsi = ~df["mmsi"].astype(str).str.fullmatch(
    r"\d{9}"
)

print(
    "Invalid MMSI:",
    invalid_mmsi.sum()
)


# ============================================================
# 11. FLAG SAME MMSI + TIMESTAMP RECORDS
# ============================================================

df["same_mmsi_timestamp"] = df.duplicated(
    subset=["mmsi", "base_date_time"],
    keep=False
)


print(
    "\nSame MMSI + timestamp records:",
    df["same_mmsi_timestamp"].sum()
)


# We DO NOT automatically delete these records.
# They may contain different coordinates or other information.


same_time = df[
    df["same_mmsi_timestamp"]
].copy()


print(
    "MMSI + timestamp groups:",
    same_time.groupby(
        ["mmsi", "base_date_time"]
    ).ngroups
)


if len(same_time) > 0:

    print("\nExamples of repeated MMSI + timestamp records:")

    print(
        same_time[
            [
                "mmsi",
                "base_date_time",
                "longitude",
                "latitude",
                "sog",
                "cog",
                "heading"
            ]
        ]
        .sort_values(
            ["mmsi", "base_date_time"]
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 12. SORT AIS DATA
# ============================================================

df = df.sort_values(
    ["mmsi", "base_date_time"]
).reset_index(drop=True)


# ============================================================
# 13. BASIC AIS FIELD VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("AIS FIELD VALIDATION")
print("=" * 60)


print("\nSOG:")
print(df["sog"].describe())


print("\nCOG:")
print(df["cog"].describe())


print(
    "\nNegative SOG:",
    (df["sog"] < 0).sum()
)


print(
    "SOG > 100:",
    (df["sog"] > 100).sum()
)


print(
    "\nNegative COG:",
    (df["cog"] < 0).sum()
)


print(
    "COG > 360:",
    (df["cog"] > 360).sum()
)


print(
    "\nEarliest timestamp:",
    df["base_date_time"].min()
)


print(
    "Latest timestamp:",
    df["base_date_time"].max()
)


print(
    "Timezone:",
    df["base_date_time"].dt.tz
)


print(
    "\nMMSI dtype:",
    df["mmsi"].dtype
)


print(
    "MMSI min:",
    df["mmsi"].min()
)


print(
    "MMSI max:",
    df["mmsi"].max()
)


# ============================================================
# 14. TIME DIFFERENCE BETWEEN AIS OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("TIME GAP ANALYSIS")
print("=" * 60)


time_diff = (
    df.groupby("mmsi")["base_date_time"]
    .diff()
)


print(
    "\nNegative time differences:",
    (time_diff < pd.Timedelta(0)).sum()
)


print(
    "Zero time differences:",
    (time_diff == pd.Timedelta(0)).sum()
)


print(
    "Positive time differences:",
    (time_diff > pd.Timedelta(0)).sum()
)


print("\nTime difference statistics:")
print(time_diff.describe())


print(
    "\nGaps > 10 minutes:",
    (time_diff > pd.Timedelta(minutes=10)).sum()
)


print(
    "Gaps > 30 minutes:",
    (time_diff > pd.Timedelta(minutes=30)).sum()
)


print(
    "Gaps > 1 hour:",
    (time_diff > pd.Timedelta(hours=1)).sum()
)


# ============================================================
# 15. PREVIOUS POSITION / TIME
# ============================================================

df["prev_lat"] = (
    df.groupby("mmsi")["latitude"].shift(1)
)


df["prev_lon"] = (
    df.groupby("mmsi")["longitude"].shift(1)
)


df["prev_time"] = (
    df.groupby("mmsi")["base_date_time"].shift(1)
)


# ============================================================
# 16. CALCULATE TIME DIFFERENCE IN SECONDS
# ============================================================

df["time_diff_seconds"] = (
    df["base_date_time"] -
    df["prev_time"]
).dt.total_seconds()


print(
    "\nTime difference in seconds:"
)

print(
    df["time_diff_seconds"].describe()
)


# ============================================================
# 17. IDENTIFY AIS GAPS
# ============================================================

df["ais_gap"] = (
    df["time_diff_seconds"] > 600
)


print(
    "\nAIS gaps > 10 minutes:",
    df["ais_gap"].sum()
)


# ============================================================
# 18. VALID MOVEMENT TRANSITIONS
# ============================================================

valid_move = (
    df["prev_lat"].notna() &
    df["prev_lon"].notna() &
    df["prev_time"].notna() &
    df["time_diff_seconds"].notna() &
    (df["time_diff_seconds"] > 0)
)


print(
    "\nValid movement transitions:",
    valid_move.sum()
)


print(
    "Invalid movement transitions:",
    (~valid_move).sum()
)


# ============================================================
# 19. GEODESIC DISTANCE
# ============================================================

print("\n" + "=" * 60)
print("MOVEMENT ANALYSIS")
print("=" * 60)


geod = Geod(
    ellps="WGS84"
)


df["distance_m"] = np.nan


azimuth1, azimuth2, distance = geod.inv(

    df.loc[
        valid_move,
        "prev_lon"
    ].to_numpy(),

    df.loc[
        valid_move,
        "prev_lat"
    ].to_numpy(),

    df.loc[
        valid_move,
        "longitude"
    ].to_numpy(),

    df.loc[
        valid_move,
        "latitude"
    ].to_numpy()
)


df.loc[
    valid_move,
    "distance_m"
] = distance


# ============================================================
# 20. IMPLIED SPEED
# ============================================================

# distance_m / seconds = metres per second
# 1 m/s = 1.94384 knots

df["implied_speed_knots"] = np.nan


df.loc[
    valid_move,
    "implied_speed_knots"
] = (

    df.loc[
        valid_move,
        "distance_m"
    ]

    /

    df.loc[
        valid_move,
        "time_diff_seconds"
    ]

    * 1.94384
)


print(
    "\nImplied speed statistics:"
)

print(
    df["implied_speed_knots"].describe()
)


print(
    "\nImplied speed > 50 knots:",
    (
        df["implied_speed_knots"] > 50
    ).sum()
)


print(
    "Implied speed > 100 knots:",
    (
        df["implied_speed_knots"] > 100
    ).sum()
)


print(
    "Implied speed > 200 knots:",
    (
        df["implied_speed_knots"] > 200
    ).sum()
)


# ============================================================
# 21. SHOW EXTREME MOVEMENT CASES
# ============================================================

print("\nTop movement anomalies:")


print(
    df.nlargest(
        20,
        "implied_speed_knots"
    )[
        [
            "mmsi",
            "base_date_time",
            "prev_time",
            "prev_lat",
            "prev_lon",
            "latitude",
            "longitude",
            "time_diff_seconds",
            "distance_m",
            "implied_speed_knots",
            "sog"
        ]
    ].to_string(index=False)
)


# ============================================================
# 22. MOVEMENT ANOMALY FLAGS
# ============================================================

df["movement_anomaly"] = (
    df["implied_speed_knots"] > 50
)


df["zero_time_gap"] = (
    df["time_diff_seconds"] == 0
)


df["ais_gap"] = (
    df["time_diff_seconds"] > 600
)


print(
    "\nMovement anomalies:",
    df["movement_anomaly"].sum()
)


print(
    "Zero-time records:",
    df["zero_time_gap"].sum()
)


print(
    "AIS gaps >10 min:",
    df["ais_gap"].sum()
)


# ============================================================
# 23. SHOW MOVEMENT ANOMALY EXAMPLES
# ============================================================

print("\nMovement anomaly examples:")


print(
    df.loc[
        df["movement_anomaly"],
        [
            "mmsi",
            "base_date_time",
            "prev_time",
            "time_diff_seconds",
            "distance_m",
            "implied_speed_knots",
            "sog"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 24. COMPARE REPORTED SOG WITH IMPLIED SPEED
# ============================================================

speed_comparison = df[
    df["implied_speed_knots"].notna() &
    df["sog"].notna()
].copy()


speed_comparison["speed_difference"] = (
    speed_comparison["implied_speed_knots"]
    - speed_comparison["sog"]
)


print(
    "\nSpeed comparison:"
)


print(
    speed_comparison[
        [
            "sog",
            "implied_speed_knots",
            "speed_difference"
        ]
    ].describe()
)


# ============================================================
# 25. VALID MOVEMENT FLAG
# ============================================================

df["valid_movement"] = (

    (df["time_diff_seconds"] > 0)

    &

    (~df["movement_anomaly"])

)


print(
    "\nValid movement counts:"
)


print(
    df["valid_movement"].value_counts()
)


# ============================================================
# 26. FINAL DATASET FOR AIS PROCESSING
# ============================================================

m4_columns = [

    "mmsi",

    "base_date_time",

    "longitude",
    "latitude",

    "sog",
    "cog",
    "heading",

    "vessel_name",
    "imo",
    "call_sign",
    "vessel_type",

    "prev_lat",
    "prev_lon",
    "prev_time",

    "time_diff_seconds",

    "distance_m",

    "implied_speed_knots",

    "movement_anomaly",
    "zero_time_gap",
    "ais_gap",

    "valid_movement"
]


# Keep only columns that actually exist.
# This prevents the script from breaking if the
# source dataset changes slightly.

available_columns = [
    column
    for column in m4_columns
    if column in df.columns
]


ais_clean = df[
    available_columns
].copy()


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLEANING SUMMARY")
print("=" * 60)


print(
    "\nFinal AIS rows:",
    len(ais_clean)
)


print(
    "Unique vessels:",
    ais_clean["mmsi"].nunique()
)


print(
    "Final columns:",
    len(ais_clean.columns)
)


print(
    "\nFinal missing-value report:"
)


missing = ais_clean.isna().sum()


missing_percentage = (
    ais_clean.isna().mean() * 100
).round(2)


quality_report = pd.DataFrame({

    "missing_count": missing,

    "missing_percentage":
        missing_percentage

})


print(
    quality_report
)


# ============================================================
# 28. SAVE CLEAN DATA
# ============================================================

OUTPUT_FILE = (
    PROCESSED_DIR /
    "ais_clean.csv"
)


print(
    "\nSaving cleaned AIS data..."
)


ais_clean.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved successfully:"
)


print(
    OUTPUT_FILE
)


print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)