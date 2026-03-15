# Project Charter

## Mission

Build a new detector that is better than `YOLO26n` on `VisDrone` while staying lighter at inference.

This project is judged by three non-negotiable criteria:

1. Better small-person detection on `VisDrone`.
2. Smaller inference footprint than `YOLO26n`.
3. Sufficient novelty to support a paper-targeted method section.

## Hard Success Criteria

The project is only considered successful when the main model satisfies all of the following:

1. `Params < YOLO26n`.
2. `GFLOPs@640 < YOLO26n`.
3. `VisDrone val mAP50-95 > YOLO26n`.
4. `VisDrone pedestrian + people performance` improves over `YOLO26n`.

`YOLO26-p2` is treated as the small-object baseline, but the main target remains to beat `YOLO26n` on both size and accuracy.

## Method Direction

The rebuild starts from a clean Ultralytics baseline and introduces only three contributions:

1. `DPSStem`:
   detail-preserving pixel-unshuffle stem so tiny persons do not disappear during the first downsample.
2. `MAFBlock`:
   a micro-aware fusion block that mixes local detail, short-range context, and contrast cues with cheap depthwise branches.
3. `AreaAwareDetectionLoss`:
   a train-time-only reweighting rule that gives more weight to assigned small boxes and adds no inference cost.

## Design Rules

1. Keep the architecture simple enough to benchmark cleanly.
2. Prefer `P2/P3/P4` detection over a heavier generic pyramid when targeting VisDrone.
3. Avoid adding complex trainer-wide side effects unless they directly improve the three core objectives.
4. Do not reintroduce the old `Nova` code path.

## Benchmark Protocol

Baselines:

1. `YOLO26n`
2. `YOLO26-p2`

Main candidate:

1. `FeatherDet`

Ablations:

1. `FeatherDet` without `area_focus`
2. `FeatherDet` with `area_focus`
3. Stem ablation: `Conv stem` vs `DPSStem`
4. Block ablation: `C3k2/C2f` vs `MAFBlock`

## Operating Rule

Every future change must answer three questions before it stays:

1. Does it reduce or at least preserve model size?
2. Does it improve VisDrone small-object behavior?
3. Does it strengthen the paper story instead of making it more ad hoc?
