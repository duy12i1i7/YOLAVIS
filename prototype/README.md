# YOLO26-Nova Prototype

This folder contains a research prototype for a new detector that aims to outperform YOLO26 with lower compute.

## Files

- `nova_blocks.py`: prototype modules
- `yolo26-nova.yaml`: draft model config
- `../YOLO26_NOVEL_RESEARCH_BLUEPRINT.md`: full research/paper blueprint

## Integration steps (Ultralytics)

1. Add modules from `nova_blocks.py` into `ultralytics/nn/modules/block.py`.
2. Export symbols in `ultralytics/nn/modules/__init__.py`.
3. Register module names in model parser (`ultralytics/nn/tasks.py`).
4. Copy `yolo26-nova.yaml` into `ultralytics/cfg/models/26/`.
5. Add `UDASLoss` logic into `ultralytics/utils/loss.py` and trainer update hook.

## First ablation order

1. Replace only `C3k2 -> C3SREB`.
2. Add sparse router neck.
3. Enable U-DAS schedule.
4. Enable selective distillation.

Use the same training budget as YOLO26n for fair comparison.

