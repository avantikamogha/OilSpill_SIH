"""
M6 backend — serves the M1-M5 pipeline outputs as one combined
'investigation case' per slick_id, for the dashboard frontend.

Run from anywhere with:
    OILSPILL_ROOT=/path/to/OilSpill_SIH uvicorn main:app --reload

If OILSPILL_ROOT isn't set, it assumes this file lives at
<repo_root>/dashboard/backend/main.py and walks up two levels.
"""

import json
import os
from pathlib import Path
from typing import Any
import math

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("OILSPILL_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = REPO_ROOT / "outputs"

SPILL_FILE = OUTPUTS / "spill" / "all_detections.json"
GEOSPATIAL_FILE = OUTPUTS / "geospatial" / "spill_metadata.json"
DRIFT_FILE = OUTPUTS / "drift" / "drift_simulation_result.json"
AIS_FILE = OUTPUTS / "AIS" / "m4_evidence.json"
SCORING_FILE = OUTPUTS / "scoring" / "ranked_vessels.json"


def _load(path: Path) -> Any:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(data: Any) -> list:
    return data if isinstance(data, list) else [data]


def _clean_nan(value: Any) -> Any:
    """Recursively replace NaN/Infinity floats with None (raw AIS track data
    has literal NaN for the first point's prev_lat/prev_lon, which strict
    JSON — and FastAPI's encoder — rejects)."""
    if isinstance(value, float):
        return None if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_nan(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Maritime Pollution Forensics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
FRONTEND_FILE = REPO_ROOT / "dashboard" / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_FILE)

@app.get("/api/health")
def health():
    return {
        "repo_root": str(REPO_ROOT),
        "found": {
            "spill": SPILL_FILE.exists(),
            "geospatial": GEOSPATIAL_FILE.exists(),
            "drift": DRIFT_FILE.exists(),
            "ais": AIS_FILE.exists(),
            "scoring": SCORING_FILE.exists(),
        },
    }


@app.get("/api/cases")
def list_cases():
    """One entry per slick_id that has ranked vessels (the fully-connected chain)."""
    scoring = _as_list(_load(SCORING_FILE))
    drift = {d["simulation"]["slick_id"]: d["simulation"] for d in _as_list(_load(DRIFT_FILE))}

    seen: dict[str, dict] = {}
    for row in scoring:
        sid = row.get("slick_id")
        if sid is None:
            continue
        entry = seen.setdefault(sid, {"slick_id": sid, "vessel_count": 0, "top_score": 0})
        entry["vessel_count"] += 1
        entry["top_score"] = max(entry["top_score"], row.get("overall_score", 0))
        sim = drift.get(sid)
        if sim:
            entry["observation_time"] = sim.get("observation_time")
            entry["origin"] = sim.get("origin_estimation", {}).get("centroid")

    return sorted(seen.values(), key=lambda e: -e["top_score"])


@app.get("/api/case/{slick_id}")
def get_case(slick_id: str):
    scoring = [r for r in _as_list(_load(SCORING_FILE)) if r.get("slick_id") == slick_id]
    if not scoring:
        raise HTTPException(status_code=404, detail=f"No ranked vessels for slick_id '{slick_id}'")

    scoring.sort(key=lambda r: r.get("rank", 999))

    drift_all = _as_list(_load(DRIFT_FILE))
    drift_entry = next(
        (d for d in drift_all if d.get("simulation", {}).get("slick_id") == slick_id), None
    )

    ais_all = _as_list(_load(AIS_FILE))
    ais_by_mmsi = {
        str(a.get("mmsi")): a for a in ais_all if a.get("slick_id") == slick_id
    }

    # geospatial polygon may or may not exist for this id (known pipeline gap
    # between M2's current run and M3-M5's run) — degrade gracefully.
    geospatial_all = _as_list(_load(GEOSPATIAL_FILE))
    spill_entry = next(
        (g for g in geospatial_all if g.get("image_id") == slick_id), None
    )

    vessels = []
    for row in scoring:
        mmsi = str(row.get("vessel_id"))
        ais = ais_by_mmsi.get(mmsi)
        vessels.append(
            {
                "rank": row.get("rank"),
                "vessel_id": row.get("vessel_id"),
                "vessel_name": row.get("vessel_name"),
                "overall_score": row.get("overall_score"),
                "confidence": row.get("confidence"),
                "evidence": row.get("evidence"),
                "explanation": row.get("explanation"),
                "track": (ais or {}).get("trajectory", {}).get("track", []),
                "gaps": (ais or {}).get("gaps", []),
                "vessel_type": (ais or {}).get("vessel_type"),
            }
        )

    origin = None
    hindcast_path = None
    forecast_path = None
    observation_time = None
    if drift_entry:
        sim = drift_entry["simulation"]
        observation_time = sim.get("observation_time")
        origin = sim.get("origin_estimation")
        hindcast_path = sim.get("hindcast_trajectories", {}).get("centroid_path")
        forecast_path = sim.get("forecast_trajectories", {}).get("centroid_path")

    return _clean_nan(
        {
            "slick_id": slick_id,
            "observation_time": observation_time,
            "spill": {
                "geometry": (spill_entry or {}).get("geometry"),
                "area_km2": (spill_entry or {}).get("area_km2"),
                "confidence": (spill_entry or {}).get("confidence"),
            },
            "origin": origin,
            "hindcast_path": hindcast_path,
            "forecast_path": forecast_path,
            "vessels": vessels,
        }
    )

# ---------------------------------------------------------------------------
# Forecast spread / marine sensitivity
# ---------------------------------------------------------------------------

# Prototype sensitive marine areas.
#
# IMPORTANT:
# These are placeholder/demo zones for the dashboard prototype.
# Replace them later with actual protected areas / mangrove zones /
# coral reef zones / fisheries / coastal ecological datasets.
#
# Format:
# {
#     "id": "...",
#     "name": "...",
#     "type": "...",
#     "geometry": GeoJSON Polygon
# }
#
SENSITIVE_MARINE_AREAS = [
    {
        "id": "demo_sensitive_01",
        "name": "Sensitive Marine Habitat — Demo Zone",
        "type": "Marine ecological zone",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [72.72, 19.02],
                [72.78, 19.02],
                [72.78, 19.08],
                [72.72, 19.08],
                [72.72, 19.02],
            ]]
        },
    },
]


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float):
    """
    Approximate destination point on Earth.

    Used only to construct a visual uncertainty/spread area around
    forecast centroid points.
    """
    earth_radius_km = 6371.0

    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    angular_distance = distance_km / earth_radius_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def _spread_circle(lat: float, lon: float, radius_km: float):
    """
    Create a simple GeoJSON polygon approximating a circular spill
    uncertainty/spread area.
    """
    coordinates = []

    # 36-sided polygon is visually smooth enough for the dashboard.
    for bearing in range(0, 360, 10):
        p_lat, p_lon = _destination_point(
            lat,
            lon,
            bearing,
            radius_km,
        )
        coordinates.append([p_lon, p_lat])

    coordinates.append(coordinates[0])

    return {
        "type": "Polygon",
        "coordinates": [coordinates],
    }


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def _extract_forecast_points(simulation):
    """
    Handles the existing drift output structure:

        forecast_trajectories:
            centroid_path:
                lats: [...]
                lons: [...]
                times: [...]   # if available

    Returns a clean list of forecast points.
    """

    path = (
        simulation
        .get("forecast_trajectories", {})
        .get("centroid_path", {})
    )

    lats = path.get("lats", [])
    lons = path.get("lons", [])
    times = path.get("times", [])

    points = []

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if lat is None or lon is None:
            continue

        point = {
            "lat": lat,
            "lon": lon,
        }

        if i < len(times):
            point["time"] = times[i]

        points.append(point)

    return points


@app.get("/api/case/{slick_id}/forecast-risk")
def get_forecast_risk(slick_id: str):
    """
    Forecast spill spread and marine sensitivity view.

    This endpoint is completely separate from the existing case endpoint.
    """

    # ---------------------------------------------------------
    # 1. Find drift simulation for this spill
    # ---------------------------------------------------------

    drift_all = _as_list(_load(DRIFT_FILE))

    drift_entry = next(
        (
            d
            for d in drift_all
            if d.get("simulation", {}).get("slick_id") == slick_id
        ),
        None,
    )

    if not drift_entry:
        raise HTTPException(
            status_code=404,
            detail=f"No drift simulation found for slick_id '{slick_id}'",
        )

    simulation = drift_entry.get("simulation", {})

    # ---------------------------------------------------------
    # 2. Current spill geometry
    # ---------------------------------------------------------

    geospatial_all = _as_list(_load(GEOSPATIAL_FILE))

    spill_entry = next(
        (
            g
            for g in geospatial_all
            if g.get("image_id") == slick_id
        ),
        None,
    )

    # ---------------------------------------------------------
    # 3. Forecast trajectory
    # ---------------------------------------------------------

    forecast_points = _extract_forecast_points(simulation)

    if not forecast_points:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast trajectory available for '{slick_id}'",
        )

    # ---------------------------------------------------------
    # 4. Generate expanding spread around forecast trajectory
    # ---------------------------------------------------------

    spread_zones = []

    for i, point in enumerate(forecast_points):

        # Increasing uncertainty/spread with forecast time.
        #
        # This is a PROTOTYPE visualization model, not a physical
        # oil-spill dispersion model.
        radius_km = min(
            8.0,
            1.0 + (i * 0.35)
        )

        spread_zones.append(
            {
                "index": i,
                "time": point.get("time"),
                "center": point,
                "radius_km": round(radius_km, 2),
                "geometry": _spread_circle(
                    point["lat"],
                    point["lon"],
                    radius_km,
                ),
            }
        )

    # ---------------------------------------------------------
    # 5. Marine sensitivity check
    # ---------------------------------------------------------

    risk_hits = []

    for area in SENSITIVE_MARINE_AREAS:

        # Prototype proximity calculation.
        #
        # For now we calculate distance from forecast centroid
        # points to the first coordinate of the sensitive polygon.
        #
        # Replace this later with actual polygon intersection/
        # nearest-boundary geometry using Shapely/geopandas.

        polygon = area["geometry"]["coordinates"][0]

        min_distance = float("inf")
        closest_forecast_point = None

        for forecast_point in forecast_points:

            for lon, lat in polygon:

                distance = _haversine_km(
                    forecast_point["lat"],
                    forecast_point["lon"],
                    lat,
                    lon,
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_forecast_point = forecast_point

        # Risk thresholds for the prototype.
        if min_distance <= 5:
            risk_level = "HIGH"
        elif min_distance <= 15:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        risk_hits.append(
            {
                "area_id": area["id"],
                "name": area["name"],
                "type": area["type"],
                "risk_level": risk_level,
                "minimum_distance_km": round(min_distance, 2),
                "closest_forecast_point": closest_forecast_point,
                "geometry": area["geometry"],
            }
        )

    overall_risk = "LOW"

    if any(r["risk_level"] == "HIGH" for r in risk_hits):
        overall_risk = "HIGH"
    elif any(r["risk_level"] == "MODERATE" for r in risk_hits):
        overall_risk = "MODERATE"

    return _clean_nan(
        {
            "slick_id": slick_id,

            "observation_time": simulation.get(
                "observation_time"
            ),

            "estimated_spill_time": simulation.get(
                "origin_estimation", {}
            ).get("estimated_spill_time_utc"),

            "current_spill": {
                "geometry": (
                    spill_entry or {}
                ).get("geometry"),

                "area_km2": (
                    spill_entry or {}
                ).get("area_km2"),

                "confidence": (
                    spill_entry or {}
                ).get("confidence"),
            },

            "forecast": {
                "points": forecast_points,
                "spread_zones": spread_zones,
            },

            "marine_risk": {
                "overall": overall_risk,
                "sensitive_areas": risk_hits,
            },

            "prototype_note": (
                "Forecast spread is visualized using expanding "
                "uncertainty buffers around the drift centroid path. "
                "It is not a physical oil-dispersion model."
            ),
        }
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)