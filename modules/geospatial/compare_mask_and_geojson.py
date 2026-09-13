"""
Directly compare M1's raw mask.png against its geojson shape, for ONE image.
Helps spot exactly where detail gets lost in the mask -> geojson conversion.

Usage:
    python compare_mask_and_geojson.py palsar_0
"""

import sys
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def main():
    image_id = sys.argv[1] if len(sys.argv) > 1 else "palsar_0"
    mask_path = f"outputs/spill/{image_id}_mask.png"
    geojson_path = f"outputs/spill/{image_id}.geojson"

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    with open(geojson_path) as f:
        gj = json.load(f)

    geom = gj["geometry"] if gj.get("type") == "Feature" else gj
    if gj.get("type") == "FeatureCollection":
        geom = gj["features"][0]["geometry"]
    coords = geom["coordinates"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: raw mask, pixel space
    ax1.imshow(mask, cmap="gray")
    ax1.set_title(f"{image_id} — raw mask (pixels)")
    ax1.axis("off")

    # Right: geojson polygon, lat/lon space
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax2.add_patch(MplPolygon(coords, closed=True, facecolor="firebrick",
                              edgecolor="black", alpha=0.7))
    ax2.set_xlim(min(xs) - 0.005, max(xs) + 0.005)
    ax2.set_ylim(min(ys) - 0.005, max(ys) + 0.005)
    ax2.set_title(f"{image_id} — geojson shape (lat/lon)")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = f"outputs/geospatial/{image_id}_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()