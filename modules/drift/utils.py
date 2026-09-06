import numpy as np


def extract_polygon_coordinates(geojson):
    try:
        geometry = geojson.get("geometry", {})
        geom_type = geometry.get("type", "Polygon")
        coordinates = geometry.get("coordinates", [])

        if not coordinates:
            raise ValueError("Coordinates array is empty.")

        lons, lats = [], []

        if geom_type == "Polygon":
            ring = coordinates[0]
            lons = [float(pt[0]) for pt in ring]
            lats = [float(pt[1]) for pt in ring]

        elif geom_type == "MultiPolygon":
            for poly in coordinates:
                ring = poly[0]
                lons.extend([float(pt[0]) for pt in ring])
                lats.extend([float(pt[1]) for pt in ring])

        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

        return lons, lats

    except (IndexError, KeyError, TypeError) as e:
        raise ValueError(f"Invalid M2 GeoJSON structure: {e}")


def compute_bounding_box(lons, lats):
    if len(lons) == 0 or len(lats) == 0:
        raise ValueError("Longitude and latitude arrays cannot be empty.")

    return [
        float(np.min(lons)),
        float(np.min(lats)),
        float(np.max(lons)),
        float(np.max(lats))
    ]