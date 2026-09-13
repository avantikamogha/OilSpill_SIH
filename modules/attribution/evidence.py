from pathlib import Path
import json
import math
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

TRACKS = (
    BASE
    / "AIS"
    / "data"
    / "processed"
    / "candidate_tracks_sorted.csv"
)

FEATURES = (
    BASE
    / "AIS"
    / "data"
    / "processed"
    / "vessel_features.csv"
)

M3_FILE = (
    BASE.parent
    / "outputs"
    / "drift"
    / "drift_simulation_result.json"
)

OUTPUT = (
    BASE.parent
    / "outputs"
    / "AIS"
    / "m4_evidence.json"
)


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

if not TRACKS.exists():
    raise FileNotFoundError(
        f"Candidate tracks not found: {TRACKS}"
    )

if not FEATURES.exists():
    raise FileNotFoundError(
        f"Vessel features not found: {FEATURES}"
    )

if not M3_FILE.exists():
    raise FileNotFoundError(
        f"M3 output not found: {M3_FILE}"
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between two
    latitude/longitude coordinates.

    Returns distance in kilometres.
    """

    R = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_float(value, decimals=None):
    """
    Safely convert a value to float.
    Returns None for missing/invalid values.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if decimals is not None:
        return round(value, decimals)

    return value


def safe_int(value):
    """
    Safely convert a value to int.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# LOAD DATA
# ============================================================

tracks = pd.read_csv(TRACKS)

features = pd.read_csv(FEATURES)

with open(
    M3_FILE,
    "r",
    encoding="utf-8"
) as f:

    m3_results = json.load(f)


# ============================================================
# BUILD M3 LOOKUP
# ============================================================
#
# For every slick_id we keep:
#
# - estimated spill time
# - estimated spill origin
# - complete hindcast
# - AIS query window
#
# M3 remains the source of truth.
# ============================================================

m3_lookup = {}


for item in m3_results:

    simulation = item.get(
        "simulation",
        {}
    )

    slick_id = simulation.get(
        "slick_id"
    )

    if not slick_id:
        continue

    origin = simulation.get(
        "origin_estimation",
        {}
    )

    centroid = origin.get(
        "centroid"
    )

    hindcast = simulation.get(
        "hindcast_trajectories",
        {}
    )

    ais_query = (
        simulation.get(
            "b4_sis_query"
        )
        or
        simulation.get(
            "m4_ais_query"
        )
        or
        item.get(
            "m4_ais_query"
        )
    )

    if not centroid:
        continue

    m3_lookup[slick_id] = {

        "estimated_spill_time": (
            origin.get(
                "estimated_spill_time_utc"
            )
        ),

        "origin_lat": centroid["lat"],

        "origin_lon": centroid["lon"],

        "hindcast": hindcast,

        "ais_query": ais_query
    }


if not m3_lookup:

    raise ValueError(
        "No valid M3 spill simulations found."
    )


# ============================================================
# PREPARE AIS TRACK DATA
# ============================================================

tracks["base_date_time"] = pd.to_datetime(
    tracks["base_date_time"],
    utc=True,
    errors="coerce"
)

tracks["prev_time"] = pd.to_datetime(
    tracks["prev_time"],
    utc=True,
    errors="coerce"
)

tracks["latitude"] = pd.to_numeric(
    tracks["latitude"],
    errors="coerce"
)

tracks["longitude"] = pd.to_numeric(
    tracks["longitude"],
    errors="coerce"
)

tracks["mmsi"] = pd.to_numeric(
    tracks["mmsi"],
    errors="coerce"
)


# Remove unusable observations

tracks = tracks.dropna(
    subset=[
        "mmsi",
        "base_date_time",
        "latitude",
        "longitude"
    ]
)


# ============================================================
# PREPARE FEATURES
# ============================================================

features["mmsi"] = pd.to_numeric(
    features["mmsi"],
    errors="coerce"
)

features["first_seen"] = pd.to_datetime(
    features["first_seen"],
    utc=True,
    errors="coerce"
)

features["last_seen"] = pd.to_datetime(
    features["last_seen"],
    utc=True,
    errors="coerce"
)


# ============================================================
# BUILD EVIDENCE
# ============================================================

evidence = []


for slick_id, m3 in m3_lookup.items():

    origin_lat = float(
        m3["origin_lat"]
    )

    origin_lon = float(
        m3["origin_lon"]
    )

    spill_time = pd.to_datetime(
        m3["estimated_spill_time"],
        utc=True,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Get AIS observations belonging to this spill scenario
    # --------------------------------------------------------

    slick_tracks = tracks[
        tracks["slick_id"].astype(str)
        == str(slick_id)
    ].copy()


    if slick_tracks.empty:
        continue


    # --------------------------------------------------------
    # Process each candidate vessel
    # --------------------------------------------------------

    for mmsi, vessel_track in slick_tracks.groupby(
        "mmsi"
    ):

        vessel_track = vessel_track.sort_values(
            "base_date_time"
        ).reset_index(drop=True)


        # ====================================================
        # SPATIAL EVIDENCE
        # ====================================================
        #
        # Calculate distance from EVERY AIS observation
        # to estimated spill origin.
        # ====================================================

        distances = []

        for _, point in vessel_track.iterrows():

            distance = haversine_km(
                point["latitude"],
                point["longitude"],
                origin_lat,
                origin_lon
            )

            distances.append(
                distance
            )


        vessel_track[
            "origin_distance_km"
        ] = distances


        min_distance_km = (
            vessel_track[
                "origin_distance_km"
            ].min()
        )


        closest_idx = (
            vessel_track[
                "origin_distance_km"
            ].idxmin()
        )

        closest_point = vessel_track.loc[
            closest_idx
        ]


        # ====================================================
        # TEMPORAL EVIDENCE
        # ====================================================

        first_seen = vessel_track[
            "base_date_time"
        ].min()

        last_seen = vessel_track[
            "base_date_time"
        ].max()


        closest_time_observation = None
        closest_time_difference_hours = None


        if pd.notna(spill_time):

            # Difference between EVERY AIS observation
            # and estimated spill time.

            time_differences = (
                vessel_track[
                    "base_date_time"
                ]
                - spill_time
            ).abs()


            closest_time_idx = (
                time_differences.idxmin()
            )

            closest_time_point = vessel_track.loc[
                closest_time_idx
            ]

            closest_time_difference_hours = (
                time_differences.loc[
                    closest_time_idx
                ].total_seconds()
                / 3600.0
            )


            closest_time_observation = {

                "time": str(
                    closest_time_point[
                        "base_date_time"
                    ]
                ),

                "latitude": float(
                    closest_time_point[
                        "latitude"
                    ]
                ),

                "longitude": float(
                    closest_time_point[
                        "longitude"
                    ]
                )
            }


            # --------------------------------------------
            # Determine temporal relation
            # --------------------------------------------

            if (
                first_seen
                <= spill_time
                <= last_seen
            ):

                temporal_relation = (
                    "present_at_estimated_spill_time"
                )

                temporal_difference_hours = 0.0

            elif first_seen > spill_time:

                temporal_relation = (
                    "arrived_after_estimated_spill_time"
                )

                temporal_difference_hours = (
                    first_seen
                    - spill_time
                ).total_seconds() / 3600.0

            else:

                temporal_relation = (
                    "left_before_estimated_spill_time"
                )

                temporal_difference_hours = (
                    spill_time
                    - last_seen
                ).total_seconds() / 3600.0

        else:

            temporal_relation = (
                "estimated_spill_time_unavailable"
            )

            temporal_difference_hours = None


        # ====================================================
        # TEMPORAL OVERLAP WITH AIS QUERY
        # ====================================================

        ais_query = m3.get(
            "ais_query"
        )

        temporal_overlap = False

        query_start = None
        query_end = None


        if ais_query:

            query_start = pd.to_datetime(
                ais_query.get("start_time"),
                utc=True,
                errors="coerce"
            )

            query_end = pd.to_datetime(
                ais_query.get("end_time"),
                utc=True,
                errors="coerce"
            )


            if (
                pd.notna(query_start)
                and pd.notna(query_end)
            ):

                overlap_start = max(
                    query_start,
                    first_seen
                )

                overlap_end = min(
                    query_end,
                    last_seen
                )

                temporal_overlap = (
                    overlap_end >= overlap_start
                )


        # ====================================================
        # TRAJECTORY DATA
        # ====================================================
        #
        # Adds:
        #
        # time_from_spill_hours
        #
        # This allows scoring.py to restrict trajectory
        # analysis to the source window instead of using
        # a vessel's entire AIS history.
        # ====================================================

        trajectory_observations = []


        for _, point in vessel_track.iterrows():

            # --------------------------------------------
            # Time relative to estimated spill
            # --------------------------------------------

            if pd.notna(spill_time):

                time_from_spill_hours = (
                    (
                        point["base_date_time"]
                        - spill_time
                    ).total_seconds()
                    / 3600.0
                )

            else:

                time_from_spill_hours = None


            trajectory_observations.append({

                "time": point[
                    "base_date_time"
                ],

                "time_from_spill_hours": (
                    None
                    if pd.isna(spill_time)
                    else round(
                        (
                            point["base_date_time"]
                            - spill_time
                        ).total_seconds() / 3600,
                        3
                    )
                ),

                "latitude": point[
                    "latitude"
                ],

                "longitude": point[
                    "longitude"
                ],

                "prev_time": point.get(
                    "prev_time"
                ),

                "prev_lat": point.get(
                    "prev_lat"
                ),

                "prev_lon": point.get(
                    "prev_lon"
                ),

                "movement_bearing": (
                    None
                    if pd.isna(
                        point.get(
                            "movement_bearing"
                        )
                    )
                    else float(
                        point[
                            "movement_bearing"
                        ]
                    )
                ),

                "segment_speed_knots": (
                    None
                    if pd.isna(
                        point.get(
                            "segment_speed_knots"
                        )
                    )
                    else float(
                        point[
                            "segment_speed_knots"
                        ]
                    )
                ),

                "distance_m": (
                    None
                    if pd.isna(
                        point.get(
                            "distance_m"
                        )
                    )
                    else float(
                        point[
                            "distance_m"
                        ]
                    )
                ),

                "origin_distance_km": round(
                    float(
                        point[
                            "origin_distance_km"
                        ]
                    ),
                    3
                )
            })


        # ====================================================
        # VESSEL FEATURES
        # ====================================================

        feature_rows = features[
            features["mmsi"] == mmsi
        ]


        if not feature_rows.empty:

            feature = feature_rows.iloc[0]


            vessel_behavior = {

                "observations": (
                    safe_int(
                        feature.get(
                            "observations"
                        )
                    )
                    or 0
                ),

                "mean_sog": safe_float(
                    feature.get(
                        "mean_sog"
                    )
                ),

                "min_sog": safe_float(
                    feature.get(
                        "min_sog"
                    )
                ),

                "max_sog": safe_float(
                    feature.get(
                        "max_sog"
                    )
                ),

                "ais_gap_count": (
                    safe_int(
                        feature.get(
                            "ais_gap_count"
                        )
                    )
                    or 0
                ),

                "max_gap_minutes": safe_float(
                    feature.get(
                        "max_gap_minutes"
                    )
                )
            }


            if "stopped_ratio" in feature:

                vessel_behavior[
                    "stopped_ratio"
                ] = safe_float(
                    feature.get(
                        "stopped_ratio"
                    ),
                    3
                )

        else:

            vessel_behavior = {}


        # ====================================================
        # GAP INFORMATION
        # ====================================================

        gaps = []


        if "time_diff_seconds" in vessel_track:

            gap_rows = vessel_track[
                vessel_track[
                    "time_diff_seconds"
                ] > 600
            ]


            for _, gap in gap_rows.iterrows():

                gap_end = gap[
                    "base_date_time"
                ]

                gap_seconds = gap[
                    "time_diff_seconds"
                ]


                if pd.notna(gap_seconds):

                    gap_start = (
                        gap_end
                        -
                        pd.to_timedelta(
                            gap_seconds,
                            unit="s"
                        )
                    )

                    gaps.append({

                        "gap_start_time": str(
                            gap_start
                        ),

                        "gap_end_time": str(
                            gap_end
                        ),

                        "gap_minutes": round(
                            float(
                                gap_seconds
                            ) / 60,
                            2
                        )
                    })


        # ====================================================
        # VESSEL METADATA
        # ====================================================

        first_point = vessel_track.iloc[0]


        vessel_name = first_point.get(
            "vessel_name"
        )

        vessel_type = first_point.get(
            "vessel_type"
        )

        imo = first_point.get(
            "imo"
        )


        if pd.isna(imo):
            imo = None


        # ====================================================
        # DRIFT / HINDCAST EVIDENCE
        # ====================================================
        #
        # IMPORTANT:
        #
        # We store ONLY the relevant M3 hindcast data.
        #
        # scoring.py reads M4 only.
        #
        # M3 remains untouched.
        # ====================================================

        hindcast = m3.get(
            "hindcast",
            {}
        )


        drift_evidence = {

            "hindcast_times": [
                str(t)
                for t in hindcast.get(
                    "times",
                    []
                )
            ],

            "centroid_path": {

                "lats": [
                    float(x)
                    for x in hindcast.get(
                        "centroid_path",
                        {}
                    ).get(
                        "lats",
                        []
                    )
                ],

                "lons": [
                    float(x)
                    for x in hindcast.get(
                        "centroid_path",
                        {}
                    ).get(
                        "lons",
                        []
                    )
                ]
            },

            "bounding_envelope": {

                "min_lats": [
                    float(x)
                    for x in hindcast.get(
                        "bounding_envelope",
                        {}
                    ).get(
                        "min_lats",
                        []
                    )
                ],

                "max_lats": [
                    float(x)
                    for x in hindcast.get(
                        "bounding_envelope",
                        {}
                    ).get(
                        "max_lats",
                        []
                    )
                ],

                "min_lons": [
                    float(x)
                    for x in hindcast.get(
                        "bounding_envelope",
                        {}
                    ).get(
                        "min_lons",
                        []
                    )
                ],

                "max_lons": [
                    float(x)
                    for x in hindcast.get(
                        "bounding_envelope",
                        {}
                    ).get(
                        "max_lons",
                        []
                    )
                ]
            }
        }


        # ====================================================
        # FINAL RAW EVIDENCE RECORD
        # ====================================================

        record = {

            "slick_id": slick_id,

            "mmsi": int(mmsi),

            "vessel_name": (
                None
                if pd.isna(vessel_name)
                else vessel_name
            ),

            "vessel_type": (
                None
                if pd.isna(vessel_type)
                else vessel_type
            ),

            "imo": imo,


            # ----------------------------------------------
            # M3 reference
            # ----------------------------------------------

            "estimated_spill_time": (
                None
                if pd.isna(spill_time)
                else str(
                    spill_time
                )
            ),


            "spill_origin": {

                "latitude": origin_lat,

                "longitude": origin_lon
            },


            # ----------------------------------------------
            # Drift / M3 hindcast
            # ----------------------------------------------

            "drift": drift_evidence,


            # ----------------------------------------------
            # Spatial evidence
            # ----------------------------------------------

            "spatial": {

                "min_distance_km": round(
                    float(
                        min_distance_km
                    ),
                    3
                ),

                "closest_observation": {

                    "time": str(
                        closest_point[
                            "base_date_time"
                        ]
                    ),

                    "latitude": float(
                        closest_point[
                            "latitude"
                        ]
                    ),

                    "longitude": float(
                        closest_point[
                            "longitude"
                        ]
                    )
                }
            },


            # ----------------------------------------------
            # Temporal evidence
            # ----------------------------------------------

            "temporal": {

                "first_seen": str(
                    first_seen
                ),

                "last_seen": str(
                    last_seen
                ),

                "temporal_overlap": bool(
                    temporal_overlap
                ),

                "relation": temporal_relation,

                "difference_from_spill_hours": (
                    None
                    if temporal_difference_hours
                    is None
                    else round(
                        temporal_difference_hours,
                        3
                    )
                ),

                # NEW
                "closest_observation_time": (
                    None
                    if closest_time_observation
                    is None
                    else closest_time_observation[
                        "time"
                    ]
                ),

                # NEW
                "closest_observation_difference_hours": (
                    None
                    if closest_time_difference_hours
                    is None
                    else round(
                        closest_time_difference_hours,
                        4
                    )
                ),

                # NEW
                "closest_observation": (
                    closest_time_observation
                )
            },


            # ----------------------------------------------
            # Trajectory evidence
            # ----------------------------------------------

            "trajectory": {

                "observations": len(
                    trajectory_observations
                ),

                "track": trajectory_observations
            },

            # ----------------------------------------------
            # Drift evidence
            # ----------------------------------------------
            #
            # Normalize the M3 hindcast structure so that
            # scoring.py does not need to know the exact
            # M3 JSON structure.
            #
            # ----------------------------------------------

            "drift": {

                "hindcast_times": m3["hindcast"].get(
                    "times",
                    []
                ),

                "centroid_path": {

                    "lats": m3["hindcast"]
                    .get(
                        "centroid_path",
                        {}
                    )
                    .get(
                        "lats",
                        []
                    ),

                    "lons": m3["hindcast"]
                    .get(
                        "centroid_path",
                        {}
                    )
                    .get(
                        "lons",
                        []
                    )
                },

                "bounding_envelope": {

                    "min_lats": m3["hindcast"]
                    .get(
                        "bounding_envelope",
                        {}
                    )
                    .get(
                        "min_lats",
                        []
                    ),

                    "max_lats": m3["hindcast"]
                    .get(
                        "bounding_envelope",
                        {}
                    )
                    .get(
                        "max_lats",
                        []
                    ),

                    "min_lons": m3["hindcast"]
                    .get(
                        "bounding_envelope",
                        {}
                    )
                    .get(
                        "min_lons",
                        []        
                    ),

                    "max_lons": m3["hindcast"]
                    .get(
                        "bounding_envelope",
                        {}
                    )
                    .get(
                        "max_lons",
                        []
                    )
                },

                "origin": {

                    "latitude": float(
                        origin_lat
                    ),

                    "longitude": float(
                        origin_lon
                    )
                }
            },

            # ----------------------------------------------
            # Vessel behavior
            # ----------------------------------------------

            "behavior": vessel_behavior,


            # ----------------------------------------------
            # AIS gaps
            # ----------------------------------------------

            "gaps": gaps,


            # ----------------------------------------------
            # Explicit note
            # ----------------------------------------------

            "evidence_note": (
                "Raw attribution evidence for "
                "candidate ranking. Spatial distance "
                "is calculated using Haversine distance. "
                "Temporal evidence uses the closest AIS "
                "observation to the estimated spill time. "
                "Drift evidence contains the M3 hindcast "
                "trajectory. This evidence supports "
                "candidate ranking only and does not "
                "establish causation or confirm discharge."
            )
        }


        evidence.append(
            record
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evidence,
        f,
        indent=2,
        default=str
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "M4 evidence generation complete."
)

print(
    f"Evidence records: "
    f"{len(evidence):,}"
)

print(
    f"Spill scenarios: "
    f"{len(set(
        item["slick_id"]
        for item in evidence
    )):,}"
)

print(
    f"Candidate vessels: "
    f"{len(set(
        item["mmsi"]
        for item in evidence
    )):,}"
)

print(
    f"Output: {OUTPUT}"
)