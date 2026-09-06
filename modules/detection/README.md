# Detection Pipelines

The notebook workflow is available as reusable Python pipelines. Run commands from the repository root:

```bash
python -m modules.detection.train train --epochs 5
python -m modules.detection.evaluate evaluate --split val --limit 200
python -m modules.detection.infer infer --split val
```

Inference writes one binary mask, GeoJSON file, and five-key metadata JSON per image, plus `all_detections.json`, to `outputs/spill` by default. Use `--image-dir`, `--output`, `--weights`, `--threshold`, `--min-area`, and `--geo-bounds min_lon,min_lat,max_lon,max_lat` to override defaults.