# FeatherDet on VisDrone

This is the clean restart benchmark recipe for the project.

Shortcut script from repo root:

```bash
./scripts/visdrone_benchmark.sh train yolo26n
./scripts/visdrone_benchmark.sh train yolo26p2
./scripts/visdrone_benchmark.sh train featherdet
./scripts/visdrone_benchmark.sh train featherdet-noarea
```

Summaries after each run:

```bash
./scripts/visdrone_benchmark.sh summarize yolo26n
./scripts/visdrone_benchmark.sh summarize yolo26p2
./scripts/visdrone_benchmark.sh summarize featherdet
./scripts/visdrone_benchmark.sh summarize featherdet-noarea
```

Record all numbers in:

```bash
/kaggle/working/YOLAVIS/VISDRONE_SCORECARD.md
```

## 1. Baseline: YOLO26n

```bash
yolo detect train \
  model=/kaggle/working/YOLAVIS/ultralytics/ultralytics/cfg/models/26/yolo26.yaml \
  data=VisDrone.yaml \
  epochs=1000 time=11.5 imgsz=960 batch=16 workers=8 device=0,1 \
  project=/kaggle/working/runs name=visdrone_yolo26n
```

## 2. Small-object baseline: YOLO26-p2

```bash
yolo detect train \
  model=/kaggle/working/YOLAVIS/ultralytics/ultralytics/cfg/models/26/yolo26-p2.yaml \
  data=VisDrone.yaml \
  epochs=1000 time=11.5 imgsz=960 batch=16 workers=8 device=0,1 \
  project=/kaggle/working/runs name=visdrone_yolo26p2
```

## 3. Main model: FeatherDet

```bash
yolo detect train \
  model=/kaggle/working/YOLAVIS/ultralytics/ultralytics/cfg/models/26/featherdet-visdrone.yaml \
  data=VisDrone.yaml \
  epochs=1000 time=11.5 imgsz=960 batch=24 workers=8 device=0,1 \
  project=/kaggle/working/runs name=visdrone_featherdet
```

## 4. Ablation: turn off area-aware loss

Create a copy of the YAML and set `area_focus: False`, then run:

```bash
yolo detect train \
  model=/kaggle/working/YOLAVIS/ultralytics/ultralytics/cfg/models/26/featherdet-visdrone-noarea.yaml \
  data=VisDrone.yaml \
  epochs=1000 time=11.5 imgsz=960 batch=24 workers=8 device=0,1 \
  project=/kaggle/working/runs name=visdrone_featherdet_noarea
```

## 5. Resume

```bash
yolo detect train resume \
  model=/kaggle/working/runs/visdrone_featherdet/weights/last.pt \
  device=0,1 time=11.5
```

## 6. Validation

```bash
yolo detect val \
  model=/kaggle/working/runs/visdrone_featherdet/weights/best.pt \
  data=VisDrone.yaml split=val imgsz=960 device=0,1
```
