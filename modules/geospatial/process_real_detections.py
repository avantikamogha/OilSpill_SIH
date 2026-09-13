"""
Process REAL Member 1 output.

Input : all_detections.json (list of {image_id, detected, confidence, mask_path, bbox})
        + a companion .geojson file per detection (already has real lat/lon geometry)
Output: spill_metadata.json (your M2 handoff, same format as before)

Usage:
    python process_real_detections.py --detections all_detections.json
"""

import argparse
import json
import glob
import os
from shapely.geometry import shape, mapping

from spill_characterisation import geodesic_area_km2, perimeter_km


def find_geojson_for(entry):
    """
    Guess the companion geojson path from mask_path.
    e.g. outputs/spill/palsar_10_mask.png -> outputs/spill/palsar_10.geojson
    Adjust this if M1's actual naming differs.
    """
    base = entry["mask_path"].replace("_mask.png", "")
    candidates = [
        base + ".geojson",
        base.replace("outputs/spill", "outputs/geojson") + ".geojson",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: search anywhere for a file with matching image_id
    matches = glob.glob(f"**/{entry['image_id']}*.geojson", recursive=True)
    return matches[0] if matches else None


def characterise_from_geojson(geojson_path, image_id, confidence):
    with open(geojson_path) as f:
        gj = json.load(f)

    # handle both a raw geometry and a Feature/FeatureCollection
    if gj.get("type") == "FeatureCollection":
        geom = gj["features"][0]["geometry"]
    elif gj.get("type") == "Feature":
        geom = gj["geometry"]
    else:
        geom = gj

    poly = shape(geom)
    centroid = poly.centroid
    minx, miny, maxx, maxy = poly.bounds

    # length/width/orientation via minimum rotated rectangle on the real polygon
    mrr = poly.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = ((mrr_coords[0][0]-mrr_coords[1][0])**2 + (mrr_coords[0][1]-mrr_coords[1][1])**2) ** 0.5
    edge2 = ((mrr_coords[1][0]-mrr_coords[2][0])**2 + (mrr_coords[1][1]-mrr_coords[2][1])**2) ** 0.5
    length_deg, width_deg = max(edge1, edge2), min(edge1, edge2)
    # rough deg->km conversion at this latitude (same approach as before)
    import numpy as np
    m_per_deg_lat = 111.32
    m_per_deg_lon = 111.32 * np.cos(np.radians(centroid.y))
    length_km = length_deg * max(m_per_deg_lat, m_per_deg_lon)
    width_km = width_deg * max(m_per_deg_lat, m_per_deg_lon)

    return {
        "image_id": image_id,
        "detected": True,
        "confidence": confidence,
        "centroid": [round(centroid.x, 5), round(centroid.y, 5)],
        "area_km2": round(geodesic_area_km2(poly), 2),
        "perimeter_km": round(perimeter_km(poly), 2),
        "length_km": round(length_km, 2),
        "width_km": round(width_km, 2),
        "bbox": [round(minx, 5), round(miny, 5), round(maxx, 5), round(maxy, 5)],
        "geometry": mapping(poly),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="Path to all_detections.json")
    parser.add_argument("--top", action="store_true", help="Only process the single highest-confidence detection")
    parser.add_argument("--out", type=str, default="outputs/geospatial/spill_metadata.json")
    args = parser.parse_args()

    with open(args.detections) as f:
        detections = json.load(f)

    detected = [d for d in detections if d.get("detected")]
    if args.top:
        detected = [max(detected, key=lambda d: d["confidence"])]

    results = []
    for entry in detected:
        gj_path = find_geojson_for(entry)
        if not gj_path:
            print(f"WARNING: no geojson found for {entry['image_id']}, skipping")
            continue
        results.append(characterise_from_geojson(gj_path, entry["image_id"], entry["confidence"]))

    out = results[0] if args.top else results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out} ({len(results)} spill(s) processed)")
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "geometry"}, indent=2))


if __name__ == "__main__":
    main()