# OilTrace (Demo name) — Explainable Maritime Pollution Forensics

An end-to-end forensic pipeline that detects oil spills from satellite
imagery, reconstructs where they came from using ocean drift physics, and
ranks the vessels most likely responsible — with full evidence transparency
at every step.

Built for Smart India Hackathon. This is a working prototype demonstrating
the complete investigative pipeline on a real controlled scenario — not a
production system, and not a legal attribution tool.

---

## The Problem

Illegal oil discharges at sea are hard to catch because:
- Spills drift with currents and wind — where you see it isn't where it started
- AIS data is imperfect — gaps, spoofing, and coverage limits are common
- Existing tools stop at detection. Nobody closes the loop to "who did this?"

OilTrace closes that loop — going from a satellite pixel to a ranked,
evidence-backed list of suspect vessels.

---

## How It Works

```
SATELLITE IMAGE
        ↓
Oil Spill Detection            (M1 — Computer Vision)
        ↓
Spill Characterisation         (M2 — Geospatial Analysis)
        ↓
Drift Simulation                (M3 — Oceanography)
     ↙           ↘
BACKTRACKING     FORECAST
     ↓
Probable Origin Zone
     ↓
AIS Correlation                 (M4 — Vessel Intelligence)
        ↓
Vessel Attribution              (M5 — Evidence Scoring)
        ↓
Ranked Suspects + Evidence
        ↓
Investigation Dashboard         (M6)
```

The output is never a verdict — it's a ranked, explainable investigation
lead, with a full evidence breakdown behind every score.

---

## What's Working So Far

### Detection (M1)
- Model: U-Net with ResNet-18 backbone
- Dataset: Deep-SAR Oil Spill Segmentation (refined)
- Validation results: Dice 0.76, IoU 0.65 (more than 2x the baseline)
- Outputs binary masks, GeoJSON vector shapes, and confidence-scored JSON
  metadata for every scene
- Look-alike filtering via area thresholding and confidence cutoffs to
  reduce false positives from calm seas / biogenic slicks

### Geospatial Characterisation (M2)
- Converts every detected slick into real-world geometry: centroid, area,
  perimeter, length, width, orientation
- Verified against raw detection masks for shape accuracy
- Outputs standardised GeoJSON + JSON, ready for drift modelling and
  attribution scoring

### Drift, AIS, Attribution, Dashboard
In progress — see module folders for current status.

---

## Demo Scenario

Region: Arabian Sea corridor off Mumbai (72.50°E, 18.80°N)
Dataset: 1,615 SAR scenes processed, 1,548 detected slicks, 67 clean water
controls

---

## Tech Stack

| Layer | Tools |
|---|---|
| Detection | PyTorch, U-Net, OpenCV |
| Geospatial | Shapely, GeoPandas-style geometry, rasterio, GeoPy |
| Coordinates | WGS84 / EPSG:4326, [lon, lat] order throughout |
| Time | UTC ISO-8601 |
| Data exchange | GeoJSON + JSON between every module |

---

## Repo Structure

```
OilSpill_SIH/
├── AIS/                    Vessel data & processing
├── docs/
├── modules/
│   ├── detection/          M1 — Oil spill detection model
│   └── geospatial/         M2 — Characterisation & metrics
├── notebooks/
└── outputs/
    ├── spill/              Detection masks, geojsons, metadata
    └── geospatial/         Characterised spill metrics
```

---

## Running the Geospatial Module

```bash
python -m venv venv
venv\Scripts\activate
pip install -r modules/geospatial/requirements.txt

# Characterise the top-confidence detection
python modules/geospatial/process_real_detections.py \
    --detections outputs/spill/all_detections.json --top

# Visualize the result
python modules/geospatial/visualize_spill.py outputs/geospatial/spill_metadata.json
```

---

## Why This Matters

Operational systems like EMSA's CleanSeaNet already prove satellite and AIS
fusion works in the real world, including cases where it directly led to
prosecution. OilTrace's contribution isn't inventing a new sensor or
dataset; it's a transparent, explainable, and modular pipeline that shows
its reasoning at every step, built openly on public data from the ground up.

This prototype proves the architecture works end-to-end. The next phase
focuses on model accuracy, wider data coverage, and robustness, not on
inventing the pipeline from scratch.
