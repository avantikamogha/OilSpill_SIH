from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Paths
BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "processed" / "ais_clean.csv"
OUTPUT_DIR = BASE / "data" / "processed"

if not INPUT.exists():
    raise FileNotFoundError(f"File not found: {INPUT}")

# Load data
ais = pd.read_csv(INPUT)
ais["base_date_time"] = pd.to_datetime(
    ais["base_date_time"], utc=True, errors="coerce"
)

# Use real AIS observations for testing
ais_test = ais.head(50000).copy()

print("Rows:", len(ais_test))
print("Unique vessels:", ais_test["mmsi"].nunique())

# Convert AIS coordinates to Points
ais_geo = gpd.GeoDataFrame(
    ais_test,
    geometry=gpd.points_from_xy(
        ais_test["longitude"], ais_test["latitude"]),
    crs="EPSG:4326"
)

print("CRS:", ais_geo.crs)

# Plot AIS positions
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 7))

ais_geo.plot(
    ax=ax, markersize=3, alpha=0.6
)
ax.set_title("AIS Vessel Positions")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()

# Real AIS observation as test reference
ref = ais_test.iloc[0]
ref_lat = ref["latitude"]
ref_lon = ref["longitude"]
print("\nTest reference:")
print("MMSI:", ref["mmsi"])
print("Time:", ref["base_date_time"])
print("Latitude:", ref_lat)
print("Longitude:", ref_lon)

# Create reference point
reference_gdf = gpd.GeoDataFrame(
    {
        "event_id": ["TEST_REFERENCE"], "mmsi": [ref["mmsi"]], "timestamp": [ref["base_date_time"]]
    },
    geometry=[Point(ref_lon, ref_lat)], crs="EPSG:4326"
)
# Plot AIS + reference
fig, ax = plt.subplots(figsize=(10, 7))
ais_geo.plot(
    ax=ax, markersize=3, alpha=0.6
)
reference_gdf.plot(
    ax=ax, marker="*", markersize=180
)
ax.set_title("AIS Positions and Test Reference")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()
# Automatically select UTM zone
zone = int((ref_lon + 180) // 6) + 1
epsg = 32600 + zone if ref_lat >= 0 else 32700 + zone
utm = f"EPSG:{epsg}"
print("\nProjected CRS:", utm)
# Project to metres
ais_projected = ais_geo.to_crs(utm)
reference_projected = reference_gdf.to_crs(utm)
# Calculate distance
ais_projected["distance_m"] = (
    ais_projected.geometry.distance(
        reference_projected.geometry.iloc[0]
    )
)
ais_projected["distance_km"] = (
    ais_projected["distance_m"] / 1000
)
# Distance results
print("\nDistance statistics (km):")
print(ais_projected["distance_km"].describe())
print("\nClosest AIS observations:")
print(
    ais_projected[
        [
            "mmsi", "base_date_time", "latitude", "longitude", "distance_km"
        ]
    ]
    .sort_values("distance_km")
    .head(20)
    .to_string(index=False)
)
# Use the complete cleaned AIS dataset
ais_test = ais.copy()

print("Total observations:", len(ais_test))
print("Unique vessels:", ais_test["mmsi"].nunique())
ref = ais_test.iloc[0]
ref_lat = ref["latitude"]
ref_lon = ref["longitude"]
ref_time = ref["base_date_time"]
radius_km = 10
time_before = 30
time_after = 30
nearby = ais_projected[
    ais_projected["distance_km"] <= radius_km
].copy()
print("\nSpatial filter:")
print("Observations within 10 km:", len(nearby))
print("Vessels within 10 km:", nearby["mmsi"].nunique())
start_time = ref_time - pd.Timedelta(minutes=time_before)
end_time = ref_time + pd.Timedelta(minutes=time_after)
candidates = nearby[
    nearby["base_date_time"].between(
        start_time,
        end_time
    )
].copy()
candidate_ids = candidates["mmsi"].unique()
print("\nCandidate vessels:")
print(len(candidate_ids))
print(
    candidates[
        [
            "mmsi", "base_date_time","latitude","longitude","sog","cog","heading","distance_km"
        ]
    ]
    .sort_values("distance_km")
    .head(20)
    .to_string(index=False)
)
candidate_vessels = (
    candidates
    .groupby("mmsi")
    .agg(
        first_seen=("base_date_time", "min"),
        last_seen=("base_date_time", "max"),
        min_distance_km=("distance_km", "min"),
        observations=("mmsi", "size")
    )
    .reset_index()
    .sort_values("min_distance_km")
)
print("\nCandidate vessel summary:")
print(
    candidate_vessels.head(20)
    .to_string(index=False)
)
candidate_vessels.to_csv(
    OUTPUT_DIR / "candidate_vessels.csv",
    index=False
)

print(
    "\nSaved:",
    OUTPUT_DIR / "candidate_vessels.csv"
)
print("\nCandidate MMSIs:", candidate_ids)
candidate_tracks = ais[
    ais["mmsi"].isin(candidate_ids)
].copy()
candidate_tracks = candidate_tracks.sort_values(
    ["mmsi", "base_date_time"]
).reset_index(drop=True)
print("\nCandidate track:")
print(
    candidate_tracks[
        [
            "mmsi","base_date_time","latitude","longitude","sog","cog","heading"
        ]
    ]
    .head(20)
    .to_string(index=False)
)
track_summary = (
    candidate_tracks
    .groupby("mmsi")
    .agg(
        first_seen=("base_date_time", "min"),
        last_seen=("base_date_time", "max"),
        observations=("mmsi", "size")
    )
    .reset_index()
)
print("\nTrack summary:")
print(track_summary.to_string(index=False))
candidate_geo = gpd.GeoDataFrame(
    candidate_tracks,
    geometry=gpd.points_from_xy(
        candidate_tracks["longitude"],
        candidate_tracks["latitude"]
    ),
    crs="EPSG:4326"
)
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 7))

candidate_geo.plot(
    ax=ax,
    markersize=4,
    alpha=0.6
)

reference_gdf.plot(
    ax=ax,
    marker="*",
    markersize=180
)

ax.set_title("Candidate Vessel Track")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()
candidate_tracks.to_csv(
    OUTPUT_DIR / "candidate_tracks.csv",
    index=False
)

print("Saved:", OUTPUT_DIR / "candidate_tracks.csv")