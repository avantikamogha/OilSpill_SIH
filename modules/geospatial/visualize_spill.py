"""
Visualize your M2 output — plots the spill polygon, centroid, and bbox
on a simple lat/lon chart. No internet/basemap needed.

Usage:
    python visualize_spill.py outputs/geospatial/spill_metadata.json
"""

import sys
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def plot_spill(entry, ax):
    geom = entry["geometry"]
    coords = geom["coordinates"][0]  # exterior ring: list of [lon, lat]

    poly_patch = MplPolygon(coords, closed=True, facecolor="firebrick",
                             edgecolor="black", alpha=0.6, label="Detected slick")
    ax.add_patch(poly_patch)

    # centroid
    cx, cy = entry["centroid"]
    ax.plot(cx, cy, "yo", markersize=8, label="Centroid")

    # bbox
    minx, miny, maxx, maxy = entry["bbox"]
    bbox_patch = MplPolygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)],
        closed=True, fill=False, edgecolor="cyan", linestyle="--", label="Bounding box"
    )
    ax.add_patch(bbox_patch)

    ax.set_xlim(minx - 0.05, maxx + 0.05)
    ax.set_ylim(miny - 0.05, maxy + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{entry['image_id']}  |  area={entry['area_km2']} km²  "
                 f"len={entry.get('length_km')} km  wid={entry.get('width_km')} km")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/geospatial/spill_metadata.json"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    with open(path) as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else [data]
    if len(entries) > limit:
        print(f"{len(entries)} entries found — plotting first {limit}. "
              f"Pass a number as 2nd arg to change, e.g. 'visualize_spill.py file.json 20'")
        entries = entries[:limit]

    n = len(entries)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    for i, entry in enumerate(entries):
        ax = axes[i // cols][i % cols]
        plot_spill(entry, ax)

    # hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout()
    out_path = "outputs/geospatial/spill_visualization.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()