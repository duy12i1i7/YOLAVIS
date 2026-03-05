# YOLO26-Nova Ablation Quickstart

This guide compares `YOLO26n` and `YOLO26-Nova` under the same training budget.

## 1) Install

```bash
pip install -e .
```

## 2) Baseline run (YOLO26n)

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26.yaml \
  data=coco.yaml \
  epochs=300 imgsz=640 batch=128 \
  workers=16 device=0 \
  project=runs/nova name=baseline_yolo26n
```

## 3) Nova run

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26-nova.yaml \
  data=coco.yaml \
  epochs=300 imgsz=640 batch=128 \
  workers=16 device=0 \
  project=runs/nova name=nova_yolo26n
```

## 4) Seeded runs (paper-ready mean/std)

```bash
for s in 0 1 2; do
  yolo detect train \
    model=ultralytics/cfg/models/26/yolo26-nova.yaml \
    data=coco.yaml \
    epochs=300 imgsz=640 batch=128 \
    workers=16 device=0 seed=$s \
    project=runs/nova name=nova_seed${s}
done
```

## 5) Validation and speed

```bash
yolo detect val model=runs/nova/nova_yolo26n/weights/best.pt data=coco.yaml
yolo export model=runs/nova/nova_yolo26n/weights/best.pt format=onnx simplify=True
```

Measure ONNXRuntime/TensorRT latency with the same image size and hardware settings used for baseline.

