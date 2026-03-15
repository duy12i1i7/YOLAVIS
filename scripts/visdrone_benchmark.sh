#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  visdrone_benchmark.sh train <variant>
  visdrone_benchmark.sh val <variant>
  visdrone_benchmark.sh resume <variant>
  visdrone_benchmark.sh summarize <variant>

Variants:
  yolo26n
  yolo26p2
  featherdet
  featherdet-noarea

Environment overrides:
  REPO_ROOT   default: /kaggle/working/YOLAVIS
  DATA        default: VisDrone.yaml
  PROJECT     default: /kaggle/working/runs
  DEVICE      default: 0,1
  IMGSZ       default: 960
  EPOCHS      default: 1000
  TIME_HOURS  default: 11.5
  WORKERS     default: 8
  BATCH       override per-variant default
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

mode="$1"
variant="$2"

repo_root="${REPO_ROOT:-/kaggle/working/YOLAVIS}"
cfg_root="${repo_root}/ultralytics/cfg/models/26"
data="${DATA:-VisDrone.yaml}"
project="${PROJECT:-/kaggle/working/runs}"
device="${DEVICE:-0,1}"
imgsz="${IMGSZ:-960}"
epochs="${EPOCHS:-1000}"
time_hours="${TIME_HOURS:-11.5}"
workers="${WORKERS:-8}"

case "$variant" in
  yolo26n)
    model="${cfg_root}/yolo26.yaml"
    run_name="visdrone_yolo26n"
    batch_default=16
    ;;
  yolo26p2)
    model="${cfg_root}/yolo26-p2.yaml"
    run_name="visdrone_yolo26p2"
    batch_default=16
    ;;
  featherdet)
    model="${cfg_root}/featherdet-visdrone.yaml"
    run_name="visdrone_featherdet"
    batch_default=24
    ;;
  featherdet-noarea)
    model="${cfg_root}/featherdet-visdrone-noarea.yaml"
    run_name="visdrone_featherdet_noarea"
    batch_default=24
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    usage
    exit 1
    ;;
esac

batch="${BATCH:-$batch_default}"
weights="${project}/${run_name}/weights"
run_dir="${project}/${run_name}"

case "$mode" in
  train)
    mkdir -p "${run_dir}"
    yolo detect train \
      model="${model}" \
      data="${data}" \
      epochs="${epochs}" time="${time_hours}" imgsz="${imgsz}" \
      batch="${batch}" workers="${workers}" device="${device}" \
      project="${project}" name="${run_name}" 2>&1 | tee "${run_dir}/train.log"
    ;;
  val)
    mkdir -p "${run_dir}"
    yolo detect val \
      model="${weights}/best.pt" \
      data="${data}" split=val imgsz="${imgsz}" device="${device}" 2>&1 | tee "${run_dir}/val.log"
    ;;
  resume)
    mkdir -p "${run_dir}"
    yolo detect train resume \
      model="${weights}/last.pt" \
      device="${device}" time="${time_hours}" 2>&1 | tee -a "${run_dir}/train.log"
    ;;
  summarize)
    python3 "${repo_root}/scripts/summarize_visdrone.py" "${run_dir}"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage
    exit 1
    ;;
esac
