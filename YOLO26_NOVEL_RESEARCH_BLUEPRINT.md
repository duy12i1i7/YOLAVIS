# YOLO26-Nova Blueprint (lighter + higher accuracy + publishable novelty)

Date: 2026-03-04

## 1) Baseline and target

Official YOLO26 release used as baseline: ultralytics 8.4.0 (2026-01-14).

Detection baseline (COCO val, 640):
- YOLO26n: 2.4M params, 5.4B FLOPs, 40.9 AP (40.1 AP e2e), 38.9 ms CPU ONNX
- YOLO26s: 9.5M params, 20.7B FLOPs, 48.6 AP (47.8 AP e2e), 87.2 ms CPU ONNX

Primary target (n-scale):
- <= 2.0M params
- <= 4.5B FLOPs
- >= 41.5 AP (or >= 41.0 AP e2e)
- <= 33 ms CPU ONNX

## 2) Proposed method: YOLO26-Nova

### Contribution C1: SR-Elastic Block (SREB)

Replace C3k2 with a train-time multi-branch block that is fused to single-path at inference.

Train-time branches:
- B1: 3x3 DWConv + 1x1 PWConv
- B2: 5x5 DWConv (or 3x3 dilated) + 1x1 PWConv
- B3: identity/1x1 residual branch

Gate:
- g = softmax(MLP(GAP(x))/tau)
- y = sum_i g_i * B_i(x)

Regularization:
- L_gate = lambda_l1 * ||g||_1 + lambda_ent * H(g)
- branch drop-path to avoid branch collapse

Deploy:
- structural re-parameterization fuses active branches into one conv path.

Why this is strong:
- Accuracy from large receptive field + adaptive branch usage.
- Latency stays low after fusion.

### Contribution C2: Sparse Router Neck (SRN)

Replace dense PAN connectivity with learnable sparse cross-scale routing.

For feature levels {P3, P4, P5}:
- Learn edge logits a_ij for each directed edge i->j
- Use hard-concrete/gumbel gate z_ij during training
- Aggregate only selected edges

Loss terms:
- L_sparse = lambda_s * sum z_ij
- L_lat = lambda_lat * LatencyProxy(z)

Deploy:
- freeze top-k edges and export as static graph (no dynamic overhead).

Why this is strong:
- reduces neck memory traffic and FLOPs.
- keeps important information paths only.

### Contribution C3: U-DAS (Uncertainty-aware Dual Assignment Schedule)

YOLO26 already uses dual branches (o2m/o2o). Extend this with uncertainty-aware scheduling and selective distillation.

Training objective:
- L = L_o2o + alpha(t) * L_o2m + beta(t) * L_distill + L_sparse + L_lat + L_gate

Schedules:
- alpha(t): decay from 0.8 -> 0.1 (curriculum from dense to strict matching)
- beta(t): warm-up then stabilize

Selective distillation:
- teacher: o2m branch
- student: o2o branch
- distill only high-uncertainty positives:
  w = sigmoid(k*(uncertainty - delta))
  L_distill = sum w * KL(p_o2o || p_o2m)

Why this is strong:
- richer supervision early, cleaner o2o behavior late.
- improves AP without inference-time cost.

## 3) Exact replacement map on YOLO26

Current YOLO26 detect backbone/head (yolo26.yaml): Conv, C3k2, SPPF, C2PSA, PAN-like neck, Detect.

Replacement map:
- C3k2 (all stages) -> C3SREB
- C2PSA (P5 stage) -> C2PSA-Lite (grouped attention + low-rank FFN)
- PAN concat paths -> SRN (sparse routed fusion)
- Detect head -> keep Detect for compatibility, add U-DAS in loss only

Inference graph remains single-path and export-friendly.

## 4) Why this is publishable (novelty argument)

Nearest prior art (from literature):
- YOLOv10: consistent dual assignment for NMS-free YOLO
- RT-DETRv3/DEIM: dense positive supervision and improved matching
- RepVGG/RepNeXt: structural re-parameterization
- MobileNetV4/FasterNet/RepViT: mobile-efficient operators and design

Novel combination (inference):
- No known work combines: re-parameterized elastic CSP replacement + latency-aware sparse neck routing + uncertainty-aware dual-assignment distillation in an end-to-end YOLO26-style detector.

Paper claim should focus on:
- Better Pareto frontier (AP vs params/FLOPs/CPU latency)
- Zero additional inference branch cost
- Strong small-object AP gains

## 5) Experiment protocol (must-have for paper)

Datasets:
- COCO 2017 (main)
- Optional: VisDrone or TinyPerson for small-object stress test

Metrics:
- AP, AP50, AP75, APS/APM/APL
- params, FLOPs
- latency: ONNXRuntime CPU + TensorRT FP16 + optional ARM

Ablations (minimum):
- A0: YOLO26n baseline
- A1: +C3SREB
- A2: +SRN
- A3: +U-DAS (schedule only)
- A4: +selective distillation
- A5: full model

Publish quality checks:
- 3 seeds for final model
- report mean +- std
- fair training budget (same epochs, same image size)

## 6) Implementation plan (Ultralytics codebase)

Files to modify:
- ultralytics/nn/modules/block.py
  - add SREB, C3SREB, C2PSALite
- ultralytics/nn/tasks.py
  - register new blocks
- ultralytics/cfg/models/26/yolo26-nova.yaml
  - architecture config
- ultralytics/utils/loss.py
  - add U-DAS loss wrapper
- ultralytics/models/yolo/detect/train.py
  - schedule hooks for alpha(t), beta(t)

## 7) Minimal success criteria

If final model reaches all:
- AP >= baseline +0.5
- params <= baseline -10%
- FLOPs <= baseline -10%
- CPU ONNX latency <= baseline -10%

then project is strong enough for a workshop paper; with stronger gains and broad ablations it is ready for main-track submission.

## 8) Risks and mitigation

Risk:
- Sparse routing can hurt AP if over-pruned.
Mitigation:
- warm-up without sparsity, then gradual lambda_s/lambda_lat ramp.

Risk:
- distillation destabilizes late training.
Mitigation:
- confidence threshold + beta schedule + stop distill in final epochs.

Risk:
- novelty overlap concerns from reviewers.
Mitigation:
- clear contribution decomposition + full ablations + latency-on-device evidence.

