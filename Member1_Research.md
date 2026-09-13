# Member 1 — Satellite Detection Deliverable

## 1. SAR Detection Principles
* **Why SAR?** All-weather, day/night capability capable of penetrating persistent ocean cloud decks.
* **Oil Slick Signature:** Oil dampens capillary waves, causing specular reflection away from radar sensors and resulting in distinct dark backscatter patches.
* **Look-Alikes Handled:** Calm sea areas (< 3 m/s wind) and biogenic slicks were accounted for by setting minimum area thresholding (> 30 px) and confidence filtering (> 0.50).

## 2. Model Baseline & Results
* **Architecture:** U-Net (ResNet-18 Backbone)
* **Dataset:** Deep-SAR Oil Spill Segmentation (Refined)
* **Input Resolution:** 256x256 grayscale SAR patches
* **Evaluation Metrics (Validation Set):**
  * **Val Dice:** 0.7604 (Baseline: 0.3506)
  * **Val IoU:** 0.6509 (Baseline: 0.2416)
  * **Val Loss:** 0.2403
* **Outputs Generated:**
  * Exact 5-Key API Metadata Contracts (`outputs/spill/*.json`)
  * Binary Segmentation Masks (`outputs/spill/*_mask.png`)
  * Geospatial Vector Layers (`outputs/spill/*.geojson`)