#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def find_metric(row: dict[str, str], *candidates: str) -> float | None:
    for key in candidates:
        if key in row and row[key] not in {"", "nan", "None"}:
            return float(row[key])
    return None


def load_best_metrics(results_csv: Path) -> dict[str, float] | None:
    if not results_csv.exists():
        return None

    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    map_key = None
    for candidate in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95(B)", "mAP50-95"):
        if candidate in rows[0]:
            map_key = candidate
            break
    if map_key is None:
        return None

    best = max(rows, key=lambda r: float(r.get(map_key, 0.0) or 0.0))
    return {
        "epoch": float(best.get("epoch", 0.0) or 0.0),
        "precision": find_metric(best, "metrics/precision(B)", "metrics/precision"),
        "recall": find_metric(best, "metrics/recall(B)", "metrics/recall"),
        "map50": find_metric(best, "metrics/mAP50(B)", "metrics/mAP50"),
        "map5095": find_metric(best, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def load_model_stats(log_path: Path) -> dict[str, str] | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    summary = re.search(
        r"summary:\s+\d+\s+layers,\s+([\d,]+)\s+parameters.*?([\d.]+)\s+GFLOPs",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not summary:
        return None

    return {"params": summary.group(1), "gflops": summary.group(2)}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: summarize_visdrone.py <run_dir>", file=sys.stderr)
        return 1

    run_dir = Path(sys.argv[1]).resolve()
    metrics = load_best_metrics(run_dir / "results.csv")
    stats = load_model_stats(run_dir / "train.log")

    print(f"run_dir: {run_dir}")
    if stats:
        print(f"params: {stats['params']}")
        print(f"gflops@model_summary: {stats['gflops']}")
    else:
        print("params: n/a")
        print("gflops@model_summary: n/a")

    if metrics:
        precision = f"{metrics['precision']:.4f}" if metrics["precision"] is not None else "n/a"
        recall = f"{metrics['recall']:.4f}" if metrics["recall"] is not None else "n/a"
        map50 = f"{metrics['map50']:.4f}" if metrics["map50"] is not None else "n/a"
        map5095 = f"{metrics['map5095']:.4f}" if metrics["map5095"] is not None else "n/a"
        print(f"best_epoch: {metrics['epoch']:.0f}")
        if metrics["precision"] is not None:
            print(f"precision: {metrics['precision']:.5f}")
        if metrics["recall"] is not None:
            print(f"recall: {metrics['recall']:.5f}")
        if metrics["map50"] is not None:
            print(f"mAP50: {metrics['map50']:.5f}")
        if metrics["map5095"] is not None:
            print(f"mAP50-95: {metrics['map5095']:.5f}")
        print(
            "| run | params | GFLOPs | best epoch | P | R | mAP50 | mAP50-95 |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        print(
            f"| {run_dir.name} | "
            f"{stats['params'] if stats else 'n/a'} | "
            f"{stats['gflops'] if stats else 'n/a'} | "
            f"{metrics['epoch']:.0f} | "
            f"{precision} | "
            f"{recall} | "
            f"{map50} | "
            f"{map5095} |"
        )
    else:
        print("metrics: n/a")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
