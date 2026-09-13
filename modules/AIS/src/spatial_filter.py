from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

AIS_FILE = BASE / "data" / "raw" / "synthetic_ais_india_oilspill.csv"
M3_FILE = BASE.parent.parent / "outputs" / "drift" / "drift_simulation_result.json"
OUTPUT_DIR = BASE / "data" / "processed"

if not AIS_FILE.exists():
    raise FileNotFoundError(f"AIS file not found: {AIS_FILE}")

if not M3_FILE.exists():
    raise FileNotFoundError(f"M3 output not found: {M3_FILE}")


def load_m3_queries():
    with open(M3_FILE, "r") as f:
        results = json.load(f)

    queries = [
        item["m4_ais_query"]
        for item in results
        if "m4_ais_query" in item
    ]

    if not queries:
        raise ValueError("No M4 AIS queries found in M3 output.")

    return queries


def get_bbox(box):
    if isinstance(box, dict):

        if {
            "min_lon",
            "max_lon",
            "min_lat",
            "max_lat"
        }.issubset(box):

            return (
                box["min_lon"],
                box["max_lon"],
                box["min_lat"],
                box["max_lat"]
            )

        if {
            "west",
            "east",
            "south",
            "north"
        }.issubset(box):

            return (
                box["west"],
                box["east"],
                box["south"],
                box["north"]
            )

    if isinstance(box, (list, tuple)) and len(box) == 4:

        min_lon, min_lat, max_lon, max_lat = box

        return (
            min_lon,
            max_lon,
            min_lat,
            max_lat
        )

    raise ValueError(
        f"Unsupported bounding box: {box}"
    )


def filter_ais(query):

    min_lon, max_lon, min_lat, max_lat = get_bbox(
        query["bounding_box"]
    )

    start = pd.to_datetime(
        query["start_time"],
        utc=True
    )

    end = pd.to_datetime(
        query["end_time"],
        utc=True
    )

    columns = [
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
        "status",
        "length",
        "width",
        "draft",
        "cargo",
        "transceiver"
    ]

    matches = []

    for chunk in pd.read_csv(
        AIS_FILE,
        usecols=columns,
        chunksize=250000
    ):

        chunk["base_date_time"] = pd.to_datetime(
            chunk["base_date_time"],
            utc=True,
            errors="coerce"
        )

        mask = (
            chunk["base_date_time"].between(
                start,
                end
            )
            &
            chunk["longitude"].between(
                min_lon,
                max_lon
            )
            &
            chunk["latitude"].between(
                min_lat,
                max_lat
            )
        )

        result = chunk.loc[mask].copy()

        if not result.empty:

            result["slick_id"] = query["slick_id"]

            matches.append(result)

    if not matches:

        return pd.DataFrame(
            columns=columns + ["slick_id"]
        )

    return pd.concat(
        matches,
        ignore_index=True
    )


queries = load_m3_queries()

results = []

for query in queries:

    result = filter_ais(query)

    if not result.empty:
        results.append(result)


if not results:

    raise ValueError(
        "No AIS observations matched the M3 spill queries."
    )


tracks = pd.concat(
    results,
    ignore_index=True
)

tracks = tracks.drop_duplicates(
    subset=[
        "slick_id",
        "mmsi",
        "base_date_time"
    ]
)

candidate_vessels = (
    tracks
    .groupby(
        ["slick_id", "mmsi"],
        as_index=False
    )
    .agg(
        first_seen=("base_date_time", "min"),
        last_seen=("base_date_time", "max"),
        observations=("mmsi", "size"),
        vessel_name=("vessel_name", "first"),
        imo=("imo", "first"),
        vessel_type=("vessel_type", "first")
    )
)

candidate_vessels.to_csv(
    OUTPUT_DIR / "candidate_vessels.csv",
    index=False
)

tracks.to_csv(
    OUTPUT_DIR / "candidate_tracks.csv",
    index=False
)

print("M3 → M4 AIS filtering complete.")
print(f"Queries processed: {len(queries):,}")
print(f"Matched observations: {len(tracks):,}")
print(
    f"Candidate vessels: "
    f"{tracks['mmsi'].nunique():,}"
)
print(
    f"Spill scenarios: "
    f"{tracks['slick_id'].nunique():,}"
)
print("Saved candidate_vessels.csv")
print("Saved candidate_tracks.csv")