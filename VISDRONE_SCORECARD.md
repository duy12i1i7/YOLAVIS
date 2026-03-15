# VisDrone Scorecard

Use this file as the single source of truth for benchmark results.

## Protocol

Fixed settings for all main comparisons:

1. Dataset: `VisDrone.yaml`
2. Image size: `960`
3. Time budget per Kaggle session: `11.5h`
4. Device: `0,1`
5. Validation split: `val`

## Main Table

| Model | Params | GFLOPs@640 | Best Epoch | Precision | Recall | mAP50 | mAP50-95 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO26n |  |  |  |  |  |  |  | baseline |
| YOLO26-p2 |  |  |  |  |  |  |  | small-object baseline |
| FeatherDet |  |  |  |  |  |  |  | main model |
| FeatherDet-noarea |  |  |  |  |  |  |  | ablation |

## Project Charter Check

| Check | Pass/Fail | Evidence |
| --- | --- | --- |
| Params < YOLO26n |  |  |
| GFLOPs < YOLO26n |  |  |
| mAP50-95 > YOLO26n |  |  |
| Pedestrian/people behavior improved |  |  |

## Commands

Train:

```bash
./scripts/visdrone_benchmark.sh train yolo26n
./scripts/visdrone_benchmark.sh train yolo26p2
./scripts/visdrone_benchmark.sh train featherdet
./scripts/visdrone_benchmark.sh train featherdet-noarea
```

Summarize:

```bash
./scripts/visdrone_benchmark.sh summarize yolo26n
./scripts/visdrone_benchmark.sh summarize yolo26p2
./scripts/visdrone_benchmark.sh summarize featherdet
./scripts/visdrone_benchmark.sh summarize featherdet-noarea
```
