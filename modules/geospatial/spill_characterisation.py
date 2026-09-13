"""
Module 2: Spill Characterisation
Input : binary oil mask (numpy array) + image geo-bounds (corner lat/lon)
Output: centroid, area_km2, perimeter_km, length_km, width_km, orientation_deg,
        bbox, GeoJSON polygon
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from geopy.distance import geodesic


# ---------- 1. Pixel -> Lat/Lon ----------
def pixel_to_latlon(px, py, img_w, img_h, top_left, bottom_right):
    """
    top_left / bottom_right = (lat, lon) of image corners.
    Simple linear interpolation (good enough for prototype scale).
    """
    lat_tl, lon_tl = top_left
    lat_br, lon_br = bottom_right
    lat = lat_tl + (py / img_h) * (lat_br - lat_tl)
    lon = lon_tl + (px / img_w) * (lon_br - lon_tl)
    return lat, lon


# ---------- 2. Mask -> Contour -> Polygon(lat/lon) ----------
def mask_to_geo_polygon(mask, top_left, bottom_right):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)  # ignore small noise blobs

    coords = []
    for pt in largest.squeeze():
        px, py = pt if largest.squeeze().ndim > 1 else (pt[0], pt[1])
        lat, lon = pixel_to_latlon(px, py, w, h, top_left, bottom_right)
        coords.append((lon, lat))  # GeoJSON uses (lon, lat)

    if len(coords) < 3:
        return None, largest
    poly = Polygon(coords)
    return poly, largest


# ---------- 3. Geometry metrics ----------
def geodesic_area_km2(poly):
    # Shoelace-on-lat/lon is inaccurate at scale; approximate via geodesic
    # segment-based planar projection (fine for small regional spills).
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return 0.0
    lat0 = sum(c[1] for c in coords) / len(coords)
    # meters per degree at this latitude
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat0))
    xy = [(lon * m_per_deg_lon, lat * m_per_deg_lat) for lon, lat in coords]
    area_m2 = 0.5 * abs(sum(
        xy[i][0] * xy[i - 1][1] - xy[i - 1][0] * xy[i][1] for i in range(len(xy))
    ))
    return area_m2 / 1e6


def perimeter_km(poly):
    coords = list(poly.exterior.coords)
    total = 0.0
    for i in range(len(coords) - 1):
        p1 = (coords[i][1], coords[i][0])   # (lat, lon)
        p2 = (coords[i + 1][1], coords[i + 1][0])
        total += geodesic(p1, p2).km
    return total


def length_width_orientation(contour, img_w, img_h, top_left, bottom_right):
    rect = cv2.minAreaRect(contour)  # ((cx,cy),(w,h),angle) in pixels
    (cx, cy), (pw, ph), angle = rect
    # convert pixel width/height to km using local scale at centroid
    lat_c, lon_c = pixel_to_latlon(cx, cy, img_w, img_h, top_left, bottom_right)
    m_per_px_x = geodesic(
        pixel_to_latlon(0, cy, img_w, img_h, top_left, bottom_right),
        pixel_to_latlon(img_w, cy, img_w, img_h, top_left, bottom_right)
    ).km * 1000 / img_w
    m_per_px_y = geodesic(
        pixel_to_latlon(cx, 0, img_w, img_h, top_left, bottom_right),
        pixel_to_latlon(cx, img_h, img_w, img_h, top_left, bottom_right)
    ).km * 1000 / img_h

    dim1_km = max(pw, ph) * max(m_per_px_x, m_per_px_y) / 1000
    dim2_km = min(pw, ph) * max(m_per_px_x, m_per_px_y) / 1000
    length_km, width_km = max(dim1_km, dim2_km), min(dim1_km, dim2_km)
    orientation_deg = angle if pw >= ph else angle + 90
    return length_km, width_km, orientation_deg % 180


# ---------- 4. Main entry point (matches team's JSON contract) ----------
def characterise_spill(mask, top_left, bottom_right, image_id="spill_001"):
    h, w = mask.shape
    poly, contour = mask_to_geo_polygon(mask, top_left, bottom_right)
    if poly is None:
        return {"image_id": image_id, "detected": False}

    centroid = poly.centroid
    area_km2 = geodesic_area_km2(poly)
    perim_km = perimeter_km(poly)
    length_km, width_km, orientation = length_width_orientation(contour, w, h, top_left, bottom_right)
    minx, miny, maxx, maxy = poly.bounds

    return {
        "image_id": image_id,
        "centroid": [round(centroid.x, 5), round(centroid.y, 5)],  # [lon, lat]
        "area_km2": round(area_km2, 2),
        "perimeter_km": round(perim_km, 2),
        "length_km": round(length_km, 2),
        "width_km": round(width_km, 2),
        "orientation_deg": round(orientation, 1),
        "bbox": [round(minx, 5), round(miny, 5), round(maxx, 5), round(maxy, 5)],
        "geometry": mapping(poly),  # GeoJSON geometry dict
    }


# ---------- 5. Standalone demo (run this today to prove it works) ----------
if __name__ == "__main__":
    import json

    # Fake mask: an elongated blob simulating an oil slick
    mask = np.zeros((300, 400), dtype=np.uint8)
    cv2.ellipse(mask, (200, 150), (120, 25), 30, 0, 360, 1, -1)

    # Fake georeference: image corners (replace with real Sentinel-1 bounds later)
    top_left = (15.0, 72.0)      # lat, lon
    bottom_right = (14.5, 72.8)

    result = characterise_spill(mask, top_left, bottom_right, image_id="spill_001")
    print(json.dumps(result, indent=2))