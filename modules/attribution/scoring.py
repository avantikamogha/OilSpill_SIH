import math
import pandas as pd


# def haversine_distance_km(lat1, lon1, lat2, lon2):
#     """
#     Calculate great-circle distance between two
#     latitude/longitude coordinates.

#     Returns distance in kilometers.
#     """

#     R = 6371.0  # Earth radius in km

#     lat1 = math.radians(lat1)
#     lat2 = math.radians(lat2)

#     dlat = lat2 - lat1
#     dlon = math.radians(lon2 - lon1)

#     a = (
#         math.sin(dlat / 2) ** 2
#         + math.cos(lat1)
#         * math.cos(lat2)
#         * math.sin(dlon / 2) ** 2
#     )

#     c = 2 * math.atan2(
#         math.sqrt(a),
#         math.sqrt(1 - a)
#     )

#     return R * c


# def spatial_score(distance_km, scale_km=10):
#     """
#     Convert distance from spill origin into a 0-100 score.
#     """

#     if distance_km is None:
#         return 0.0

#     score = 100 * math.exp(
#         -distance_km / scale_km
#     )

#     return max(0.0, min(100.0, score))


# def temporal_score(
#     first_seen,
#     last_seen,
#     spill_time,
#     tolerance_hours=2
# ):
#     """
#     Score whether the vessel was observed around
#     the estimated spill time.
#     """

#     first_seen = pd.to_datetime(
#         first_seen,
#         utc=True
#     )

#     last_seen = pd.to_datetime(
#         last_seen,
#         utc=True
#     )

#     spill_time = pd.to_datetime(
#         spill_time,
#         utc=True
#     )

#     if first_seen <= spill_time <= last_seen:
#         return 100.0

#     if spill_time < first_seen:
#         delta_hours = (
#             first_seen - spill_time
#         ).total_seconds() / 3600

#     else:
#         delta_hours = (
#             spill_time - last_seen
#         ).total_seconds() / 3600

#     if delta_hours >= tolerance_hours:
#         return 0.0

#     return 100 * (
#         1 - delta_hours / tolerance_hours
#     )

from pathlib import Path
import json
import math
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

INPUT = (
    BASE.parent
    / "outputs"
    / "AIS"
    / "m4_evidence.json"
)

OUTPUT = (
    BASE.parent
    / "outputs"
    / "scoring"
    / "ranked_vessels.json"
)


# ============================================================
# SCORE WEIGHTS
# ============================================================

WEIGHTS = {
    "spatial": 0.30,
    "temporal": 0.25,
    "trajectory": 0.20,
    "drift": 0.15,
    "vessel": 0.10
}


# ============================================================
# CONFIGURATION
# ============================================================

# AIS observations within this window are considered
# relevant for source trajectory analysis.

SOURCE_WINDOW_HOURS = 2.0


# M3 drift comparison is limited to this period
# after estimated spill time.

DRIFT_WINDOW_HOURS = 6.0


# Maximum time difference allowed when matching
# AIS point with an M3 hindcast point.

DRIFT_TIME_TOLERANCE_MINUTES = 30.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(
    value,
    minimum=0.0,
    maximum=100.0
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def linear_score(
    value,
    best_value,
    worst_value
):

    if value <= best_value:
        return 100.0

    if value >= worst_value:
        return 0.0

    score = (
        (worst_value - value)
        /
        (worst_value - best_value)
    ) * 100.0

    return clamp(score)


def parse_time(value):

    if value is None:
        return None

    if isinstance(
        value,
        datetime
    ):

        dt = value

    else:

        text = str(
            value
        ).strip()

        if text.endswith("Z"):

            text = (
                text[:-1]
                + "+00:00"
            )

        try:

            dt = datetime.fromisoformat(
                text
            )

        except ValueError:

            return None

    if dt.tzinfo is None:

        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(
        timezone.utc
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = math.radians(
        lat1
    )

    lon1 = math.radians(
        lon1
    )

    lat2 = math.radians(
        lat2
    )

    lon2 = math.radians(
        lon2
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(
            dlat / 2
        ) ** 2

        +

        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )

    return R * c


# ============================================================
# 1. SPATIAL SCORE
# ============================================================

def score_spatial(evidence):

    spatial = evidence.get(
        "spatial",
        {}
    )

    distance = spatial.get(
        "min_distance_km"
    )

    if distance is None:
        return 0.0

    try:

        distance = float(
            distance
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


    # -----------------------------------------------
    # Distance thresholds
    # -----------------------------------------------

    if distance <= 2:

        return 100.0

    elif distance <= 5:

        return linear_score(
            distance,
            2,
            5
        )

    elif distance <= 10:

        return linear_score(
            distance,
            5,
            10
        )

    elif distance <= 20:

        return linear_score(
            distance,
            10,
            20
        )

    elif distance <= 40:

        return linear_score(
            distance,
            20,
            40
        )

    else:

        return 0.0


# ============================================================
# 2. TEMPORAL SCORE
# ============================================================

def score_temporal(evidence):

    temporal = evidence.get(
        "temporal",
        {}
    )

    difference = temporal.get(
        "closest_observation_difference_hours"
    )

    # Fallback for older M4 files

    if difference is None:

        difference = temporal.get(
            "difference_from_spill_hours"
        )

    if difference is None:
        return 0.0

    try:

        difference = abs(
            float(difference)
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


    # -----------------------------------------------
    # Temporal thresholds
    # -----------------------------------------------

    if difference <= 0.25:

        return 100.0

    elif difference <= 0.5:

        return linear_score(
            difference,
            0.25,
            0.5
        )

    elif difference <= 1:

        return linear_score(
            difference,
            0.5,
            1
        )

    elif difference <= 2:

        return linear_score(
            difference,
            1,
            2
        )

    elif difference <= 4:

        return linear_score(
            difference,
            2,
            4
        )

    elif difference <= 8:

        return linear_score(
            difference,
            4,
            8
        )

    else:

        return 0.0


# ============================================================
# 3. TRAJECTORY SCORE
# ============================================================

def score_trajectory(evidence):

    trajectory = evidence.get(
        "trajectory",
        {}
    )

    track = trajectory.get(
        "track",
        []
    )

    if not track:
        return 0.0


    relevant_points = []


    # --------------------------------------------------------
    # Keep only AIS observations near estimated source time
    # --------------------------------------------------------

    for point in track:

        time_from_spill = point.get(
            "time_from_spill_hours"
        )

        if time_from_spill is not None:

            try:

                time_from_spill = float(
                    time_from_spill
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            if abs(
                time_from_spill
            ) > SOURCE_WINDOW_HOURS:

                continue


        distance = point.get(
            "origin_distance_km"
        )

        if distance is None:
            continue

        try:

            distance = float(
                distance
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        relevant_points.append(
            point
        )


    if not relevant_points:

        return 0.0


    # --------------------------------------------------------
    # Source-region proximity
    # --------------------------------------------------------

    distances = [

        float(
            point[
                "origin_distance_km"
            ]
        )

        for point in relevant_points

        if point.get(
            "origin_distance_km"
        ) is not None
    ]


    if not distances:

        return 0.0


    min_distance = min(
        distances
    )


    if min_distance <= 2:

        distance_score = 100.0

    elif min_distance <= 5:

        distance_score = linear_score(
            min_distance,
            2,
            5
        )

    elif min_distance <= 10:

        distance_score = linear_score(
            min_distance,
            5,
            10
        )

    elif min_distance <= 20:

        distance_score = linear_score(
            min_distance,
            10,
            20
        )

    elif min_distance <= 40:

        distance_score = linear_score(
            min_distance,
            20,
            40
        )

    else:

        distance_score = 0.0


    # --------------------------------------------------------
    # Movement consistency
    # --------------------------------------------------------

    moving = 0
    total = 0


    for point in relevant_points:

        speed = point.get(
            "segment_speed_knots"
        )

        if speed is None:
            continue

        try:

            speed = float(
                speed
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        total += 1

        if speed > 1.0:

            moving += 1


    if total > 0:

        movement_score = (
            moving / total
        ) * 100.0

    else:

        movement_score = 50.0


    # --------------------------------------------------------
    # Final trajectory score
    # --------------------------------------------------------

    final_score = (
        0.75 * distance_score
        +
        0.25 * movement_score
    )


    return round(
        clamp(final_score),
        2
    )


# ============================================================
# DRIFT HELPERS
# ============================================================

def point_inside_envelope(
    lat,
    lon,
    min_lat,
    max_lat,
    min_lon,
    max_lon
):

    return (
        min_lat <= lat <= max_lat
        and
        min_lon <= lon <= max_lon
    )


def envelope_distance_km(
    lat,
    lon,
    min_lat,
    max_lat,
    min_lon,
    max_lon
):

    if point_inside_envelope(
        lat,
        lon,
        min_lat,
        max_lat,
        min_lon,
        max_lon
    ):

        return 0.0


    closest_lat = min(
        max(
            lat,
            min_lat
        ),
        max_lat
    )

    closest_lon = min(
        max(
            lon,
            min_lon
        ),
        max_lon
    )


    return haversine_km(
        lat,
        lon,
        closest_lat,
        closest_lon
    )


def match_hindcast_time(
    ais_time,
    hindcast_times
):

    best_index = None

    best_difference = None


    for index, value in enumerate(
        hindcast_times
    ):

        hindcast_time = parse_time(
            value
        )

        if hindcast_time is None:
            continue


        difference = abs(
            (
                ais_time
                - hindcast_time
            ).total_seconds()
        )


        if (
            best_difference is None
            or
            difference < best_difference
        ):

            best_difference = difference

            best_index = index


    if best_index is None:

        return None, None


    if (
        best_difference
        >
        DRIFT_TIME_TOLERANCE_MINUTES
        * 60
    ):

        return None, None


    return (
        best_index,
        best_difference
    )


# ============================================================
# 4. DRIFT SCORE
# ============================================================

def score_drift(evidence):

    drift = evidence.get(
        "drift",
        {}
    )

    trajectory = evidence.get(
        "trajectory",
        {}
    )

    track = trajectory.get(
        "track",
        []
    )

    if not track:
        return 0.0


    hindcast_times = drift.get(
        "hindcast_times",
        []
    )


    centroid = drift.get(
        "centroid_path",
        {}
    )


    envelope = drift.get(
        "bounding_envelope",
        {}
    )


    lats = centroid.get(
        "lats",
        []
    )

    lons = centroid.get(
        "lons",
        []
    )


    min_lats = envelope.get(
        "min_lats",
        []
    )

    max_lats = envelope.get(
        "max_lats",
        []
    )

    min_lons = envelope.get(
        "min_lons",
        []
    )

    max_lons = envelope.get(
        "max_lons",
        []
    )


    if not hindcast_times:
        return 0.0

    if not lats or not lons:
        return 0.0


    matched_scores = []


    for point in track:

        # ----------------------------------------------------
        # AIS time
        # ----------------------------------------------------

        ais_time = parse_time(
            point.get(
                "time"
            )
        )

        if ais_time is None:
            continue


        # ----------------------------------------------------
        # Limit drift scoring to first 6 hours
        # after estimated spill
        # ----------------------------------------------------

        time_from_spill = point.get(
            "time_from_spill_hours"
        )

        if time_from_spill is not None:

            try:

                time_from_spill = float(
                    time_from_spill
                )

            except (
                TypeError,
                ValueError
            ):

                continue


            if (
                time_from_spill < 0
                or
                time_from_spill
                > DRIFT_WINDOW_HOURS
            ):

                continue


        lat = point.get(
            "latitude"
        )

        lon = point.get(
            "longitude"
        )


        if lat is None or lon is None:
            continue


        try:

            lat = float(lat)
            lon = float(lon)

        except (
            TypeError,
            ValueError
        ):

            continue


        # ----------------------------------------------------
        # Match AIS point to M3 hindcast time
        # ----------------------------------------------------

        index, _ = match_hindcast_time(
            ais_time,
            hindcast_times
        )


        if index is None:
            continue


        if index >= len(lats):
            continue

        if index >= len(lons):
            continue


        # ----------------------------------------------------
        # Distance to modeled oil centroid
        # ----------------------------------------------------

        centroid_distance = haversine_km(

            lat,
            lon,

            float(
                lats[index]
            ),

            float(
                lons[index]
            )
        )


        # ----------------------------------------------------
        # Centroid score
        # ----------------------------------------------------

        if centroid_distance <= 5:

            centroid_score = 100.0

        elif centroid_distance <= 10:

            centroid_score = linear_score(
                centroid_distance,
                5,
                10
            )

        elif centroid_distance <= 20:

            centroid_score = linear_score(
                centroid_distance,
                10,
                20
            )

        elif centroid_distance <= 40:

            centroid_score = linear_score(
                centroid_distance,
                20,
                40
            )

        else:

            centroid_score = 0.0


        # ----------------------------------------------------
        # Bounding envelope
        # ----------------------------------------------------

        inside_envelope = False

        envelope_distance = None


        if (
            index < len(min_lats)
            and
            index < len(max_lats)
            and
            index < len(min_lons)
            and
            index < len(max_lons)
        ):

            min_lat = float(
                min_lats[index]
            )

            max_lat = float(
                max_lats[index]
            )

            min_lon = float(
                min_lons[index]
            )

            max_lon = float(
                max_lons[index]
            )


            inside_envelope = point_inside_envelope(

                lat,
                lon,

                min_lat,
                max_lat,

                min_lon,
                max_lon
            )


            envelope_distance = (
                envelope_distance_km(

                    lat,
                    lon,

                    min_lat,
                    max_lat,

                    min_lon,
                    max_lon
                )
            )


        if inside_envelope:

            envelope_score = 100.0

        elif (
            envelope_distance is not None
            and
            envelope_distance <= 10
        ):

            envelope_score = 50.0

        else:

            envelope_score = 0.0


        # ----------------------------------------------------
        # Combined point score
        # ----------------------------------------------------

        point_score = (
            0.70 * centroid_score
            +
            0.30 * envelope_score
        )


        matched_scores.append(
            point_score
        )


    if not matched_scores:

        return 0.0


    # Strongest spatiotemporal match

    return round(
        clamp(
            max(
                matched_scores
            )
        ),
        2
    )


# ============================================================
# 5. VESSEL / AIS QUALITY SCORE
# ============================================================

def score_vessel(evidence):

    behavior = evidence.get(
        "behavior",
        {}
    )


    observations = behavior.get(
        "observations",
        0
    )

    gap_count = behavior.get(
        "ais_gap_count",
        0
    )

    max_gap = behavior.get(
        "max_gap_minutes",
        0
    )


    try:

        observations = int(
            observations
        )

    except (
        TypeError,
        ValueError
    ):

        observations = 0


    try:

        gap_count = int(
            gap_count
        )

    except (
        TypeError,
        ValueError
    ):

        gap_count = 0


    try:

        max_gap = float(
            max_gap
        )

    except (
        TypeError,
        ValueError
    ):

        max_gap = 0.0


    # --------------------------------------------------------
    # Observation coverage
    # --------------------------------------------------------

    if observations >= 20:

        observation_score = 100.0

    elif observations >= 10:

        observation_score = 85.0

    elif observations >= 5:

        observation_score = 65.0

    elif observations >= 2:

        observation_score = 40.0

    elif observations == 1:

        observation_score = 20.0

    else:

        observation_score = 0.0


    # --------------------------------------------------------
    # AIS continuity
    # --------------------------------------------------------

    if gap_count == 0:

        continuity_score = 100.0

    elif max_gap <= 10:

        continuity_score = 85.0

    elif max_gap <= 30:

        continuity_score = 65.0

    elif max_gap <= 60:

        continuity_score = 40.0

    else:

        continuity_score = 20.0


    final_score = (
        0.60 * continuity_score
        +
        0.40 * observation_score
    )


    return round(
        clamp(final_score),
        2
    )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    scores
):

    overall = (

        WEIGHTS["spatial"]
        * scores["spatial"]

        +

        WEIGHTS["temporal"]
        * scores["temporal"]

        +

        WEIGHTS["trajectory"]
        * scores["trajectory"]

        +

        WEIGHTS["drift"]
        * scores["drift"]

        +

        WEIGHTS["vessel"]
        * scores["vessel"]
    )


    return round(
        clamp(overall),
        2
    )


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    scores
):

    strong = sum(
        score >= 70
        for score in scores.values()
    )

    moderate = sum(
        score >= 60
        for score in scores.values()
    )


    spatial = scores[
        "spatial"
    ]

    temporal = scores[
        "temporal"
    ]

    trajectory = scores[
        "trajectory"
    ]

    drift = scores[
        "drift"
    ]


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if (
        spatial >= 60
        and
        temporal >= 60
        and
        strong >= 3
        and
        (
            trajectory >= 50
            or
            drift >= 50
        )
    ):

        return "HIGH"


    # --------------------------------------------------------
    # MID
    # --------------------------------------------------------

    if (
        spatial >= 40
        and
        temporal >= 40
        and
        moderate >= 2
    ):

        return "MID"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# EXPLANATIONS
# ============================================================

def generate_explanations(
    evidence,
    scores
):

    explanations = []


    # ========================================================
    # SPATIAL
    # ========================================================

    spatial = evidence.get(
        "spatial",
        {}
    )


    distance = spatial.get(
        "min_distance_km"
    )


    if distance is not None:

        distance = float(
            distance
        )


        if distance <= 5:

            explanations.append(
                f"Vessel was within approximately "
                f"{distance:.1f} km of the estimated "
                f"spill origin."
            )

        elif distance <= 20:

            explanations.append(
                f"Vessel approached within approximately "
                f"{distance:.1f} km of the estimated "
                f"spill origin."
            )

        else:

            explanations.append(
                f"Closest recorded vessel position was "
                f"approximately {distance:.1f} km from "
                f"the estimated spill origin."
            )


    # ========================================================
    # TEMPORAL
    # ========================================================

    temporal = evidence.get(
        "temporal",
        {}
    )


    time_difference = temporal.get(
        "closest_observation_difference_hours"
    )


    if time_difference is not None:

        time_difference = float(
            time_difference
        )


        if time_difference <= 0.5:

            explanations.append(
                "Vessel was observed very close to "
                "the estimated spill time."
            )

        elif time_difference <= 2:

            explanations.append(
                "Vessel was observed within approximately "
                "two hours of the estimated spill time."
            )

        else:

            explanations.append(
                "AIS observations were temporally separated "
                "from the estimated spill time."
            )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    if scores[
        "trajectory"
    ] >= 70:

        explanations.append(
            "Vessel trajectory showed strong spatial "
            "consistency with the estimated source region."
        )

    elif scores[
        "trajectory"
    ] >= 50:

        explanations.append(
            "Vessel trajectory showed moderate spatial "
            "consistency with the estimated source region."
        )

    else:

        explanations.append(
            "Vessel trajectory provided limited evidence "
            "of approach to the estimated source region."
        )


    # ========================================================
    # DRIFT
    # ========================================================

    if scores[
        "drift"
    ] >= 70:

        explanations.append(
            "Vessel position showed strong "
            "spatiotemporal consistency with the "
            "modeled drift corridor."
        )

    elif scores[
        "drift"
    ] >= 50:

        explanations.append(
            "Vessel position showed moderate "
            "consistency with the modeled drift corridor."
        )

    else:

        explanations.append(
            "Limited consistency was found between "
            "vessel positions and the modeled drift corridor."
        )


    # ========================================================
    # AIS QUALITY
    # ========================================================

    behavior = evidence.get(
        "behavior",
        {}
    )


    observations = behavior.get(
        "observations",
        0
    )

    gaps = behavior.get(
        "ais_gap_count",
        0
    )


    if scores[
        "vessel"
    ] >= 70:

        explanations.append(
            f"AIS evidence was relatively continuous "
            f"({observations} observations, "
            f"{gaps} recorded gaps)."
        )

    else:

        explanations.append(
            f"AIS coverage was limited or contained "
            f"gaps ({observations} observations, "
            f"{gaps} recorded gaps)."
        )


    return explanations


# ============================================================
# SCORE ONE CANDIDATE
# ============================================================

def score_candidate(
    evidence
):

    scores = {

        "spatial": score_spatial(
            evidence
        ),

        "temporal": score_temporal(
            evidence
        ),

        "trajectory": score_trajectory(
            evidence
        ),

        "drift": score_drift(
            evidence
        ),

        "vessel": score_vessel(
            evidence
        )
    }


    overall_score = calculate_overall_score(
        scores
    )


    confidence = calculate_confidence(
        scores
    )


    explanations = generate_explanations(
        evidence,
        scores
    )


    return {

        "slick_id": evidence.get(
            "slick_id"
        ),

        "vessel_id": str(
            evidence.get(
                "mmsi"
            )
        ),

        "vessel_name": evidence.get(
            "vessel_name"
        ),

        "overall_score": overall_score,

        "evidence": scores,

        "confidence": confidence,

        "explanation": explanations
    }


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_candidates(
    results
):
    """
    Rank vessels independently for EACH slick.

    Example:

    sentinel_10
        rank 1
        rank 2
        rank 3

    sentinel_11
        rank 1
        rank 2
        rank 3
    """

    # --------------------------------------------------------
    # Group by slick
    # --------------------------------------------------------

    grouped = {}


    for result in results:

        slick_id = result.get(
            "slick_id"
        )

        grouped.setdefault(
            slick_id,
            []
        ).append(
            result
        )


    ranked_results = []


    # --------------------------------------------------------
    # Rank each slick independently
    # --------------------------------------------------------

    for slick_id, candidates in grouped.items():

        candidates.sort(
            key=lambda item: (
                item["overall_score"],
                item["evidence"]["spatial"],
                item["evidence"]["temporal"]
            ),
            reverse=True
        )


        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            candidate["rank"] = index

            ranked_results.append(
                candidate
            )


    # --------------------------------------------------------
    # Sort output by slick, then rank
    # --------------------------------------------------------

    ranked_results.sort(
        key=lambda item: (
            str(
                item["slick_id"]
            ),
            item["rank"]
        )
    )


    return ranked_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "M5 — ATTRIBUTION & EVIDENCE SCORING"
    )

    print("=" * 65)


    # ========================================================
    # LOAD M4
    # ========================================================

    if not INPUT.exists():

        raise FileNotFoundError(
            f"M4 evidence file not found:\n{INPUT}"
        )


    with open(
        INPUT,
        "r",
        encoding="utf-8"
    ) as f:

        evidence_records = json.load(
            f
        )


    if not isinstance(
        evidence_records,
        list
    ):

        raise ValueError(
            "m4_evidence.json must contain a list."
        )


    print(
        f"Loaded {len(evidence_records):,} "
        f"candidate records."
    )


    # ========================================================
    # SCORE ALL CANDIDATES
    # ========================================================

    scored_results = []


    for evidence in evidence_records:

        try:

            result = score_candidate(
                evidence
            )

            scored_results.append(
                result
            )

        except Exception as exc:

            print(
                f"WARNING: Failed to score "
                f"{evidence.get('mmsi')}: {exc}"
            )


    # ========================================================
    # RANK
    # ========================================================

    ranked_results = rank_candidates(
        scored_results
    )


    # ========================================================
    # SAVE
    # ========================================================

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
            ranked_results,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    slick_ids = sorted(
        set(
            result["slick_id"]
            for result in ranked_results
        )
    )


    print()

    print(
        f"Scored candidates: "
        f"{len(ranked_results):,}"
    )

    print(
        f"Spill scenarios: "
        f"{len(slick_ids):,}"
    )

    print(
        f"Output: {OUTPUT}"
    )


    # ========================================================
    # PRINT TOP CANDIDATE FOR EACH SLICK
    # ========================================================

    print()

    print(
        "TOP CANDIDATE PER SPILL"
    )

    print(
        "-" * 65
    )


    for slick_id in slick_ids:

        candidates = [
            result
            for result in ranked_results
            if result["slick_id"] == slick_id
        ]


        if not candidates:
            continue


        top = candidates[0]


        print(
            f"{slick_id} | "
            f"Rank #1 | "
            f"MMSI {top['vessel_id']} | "
            f"Score {top['overall_score']:.2f} | "
            f"{top['confidence']}"
        )


    # ========================================================
    # PRINT TOP 10 OVERALL
    # ========================================================

    print()

    print(
        "TOP 10 CANDIDATES"
    )

    print(
        "-" * 65
    )


    top_results = sorted(
        ranked_results,
        key=lambda item: item[
            "overall_score"
        ],
        reverse=True
    )[:10]


    for result in top_results:

        print(
            f"{result['slick_id']} | "
            f"Rank #{result['rank']} | "
            f"{result['vessel_id']} | "
            f"{result['overall_score']:.2f} | "
            f"{result['confidence']}"
        )


    print()

    print(
        "M5 scoring and ranking complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()