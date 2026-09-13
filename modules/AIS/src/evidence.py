from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent

FEATURES = BASE / "data" / "processed" / "vessel_features.csv"
M3_FILE = BASE.parent / "outputs" / "drift" / "drift_simulation_result.json"
OUTPUT = BASE / "data" / "processed" / "m4_evidence.json"

if not FEATURES.exists():
    raise FileNotFoundError(
        f"Features not found: {FEATURES}"
)

if not M3_FILE.exists():
    raise FileNotFoundError(
        f"M3 output not found: {M3_FILE}"
)

features = pd.read_csv(FEATURES)

with open(M3_FILE, "r") as f:
    m3_results = json.load(f)

m3_lookup = {}

for item in m3_results:

    query = item.get("m4_ais_query")

    if query:
        m3_lookup[
            query["slick_id"]
        ] = query

evidence = []

for _, row in features.iterrows():

    slick_id = row["slick_id"]

    query = m3_lookup.get(
        slick_id
    )

    if not query:
        continue

    start = pd.to_datetime(
        query["start_time"],
        utc=True
    )

    end = pd.to_datetime(
        query["end_time"],
        utc=True
    )

    first_seen = pd.to_datetime(
        row["first_seen"],
        utc=True
    )

    last_seen = pd.to_datetime(
        row["last_seen"],
        utc=True
    )

    overlap_start = max(
        start,
        first_seen
    )

    overlap_end = min(
        end,
        last_seen
    )

    temporal_overlap = (
        overlap_end >= overlap_start
    )

    evidence.append({

        "slick_id": slick_id,

        "mmsi": int(row["mmsi"]),

        "vessel_name": row["vessel_name"],

        "vessel_type": row["vessel_type"],

        "imo": (
            None
            if pd.isna(row["imo"])
            else row["imo"]
        ),

        "first_seen": row["first_seen"],

        "last_seen": row["last_seen"],

        "observations": int(
            row["observations"]
        ),

        "mean_sog": round(
            row["mean_sog"],
            2
        ),

        "max_sog": round(
            row["max_sog"],
            2
        ),

        "ais_gap_count": int(
            row["ais_gap_count"]
        ),

        "max_gap_minutes": round(
            row["max_gap_minutes"],
            2
        ),

        "stopped_ratio": round(
            row["stopped_ratio"],
            3
        ),

        "temporal_overlap": bool(
            temporal_overlap
        ),

        "evidence_note": (
            "Synthetic AIS candidate for "
            "demonstration only."
        )
    })


with open(
    OUTPUT,
    "w"
) as f:

    json.dump(
        evidence,
        f,
        indent=2,
        default=str
    )

print("M4 evidence generation complete.")
print(
    f"Evidence records: "
    f"{len(evidence):,}"
)
print(f"Output: {OUTPUT}")