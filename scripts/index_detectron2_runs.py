#!/usr/bin/env python3
"""
Index Detectron2 training run folders and write a CSV summary.

Scans a root directory containing Detectron2 output folders (as saved during
training on Google Colab), parses folder names for metadata, extracts the
final evaluation metrics from each metrics.json, and records which artefacts
(model weights, inference predictions) exist.

Usage examples
--------------
# Index all runs (typical Colab usage):
  python scripts/index_detectron2_runs.py \
    --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \
    --out_csv /content/drive/MyDrive/TESE/results/runs_index.csv

# Only bright-subset runs:
  python scripts/index_detectron2_runs.py \
    --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \
    --out_csv bright_runs.csv \
    --selection_glob "bright_*"

# Local testing with the example folder:
  python scripts/index_detectron2_runs.py \
    --runs_root . \
    --out_csv runs_index.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path


# ── Folder-name parsing ─────────────────────────────────────────────────────

SUBSET_TAGS = ["bright", "dark", "vague", "lowres", "total"]
MODEL_FAMILIES = ["faster_rcnn", "mask_rcnn", "retinanet", "cascade_rcnn"]
BACKBONE_RE = re.compile(r"(R_50|R_101|X_101)")
SCHEDULE_RE = re.compile(r"(?:^|_)(1x|3x)(?:_|$)")
DATETIME_RE = re.compile(r"(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})$")


def parse_run_name(name: str) -> dict:
    """Extract metadata from a run-folder name.

    Example name: bright_100_faster_rcnn_R_50_FPN_3x_01-06-2023_08-21-12
    """
    info = {
        "subset": "",
        "cap_tag": "",
        "model_family": "",
        "backbone": "",
        "schedule": "",
        "run_date": "",
        "run_time": "",
    }

    name_lower = name.lower()

    # Subset
    for tag in SUBSET_TAGS:
        if tag in name_lower:
            info["subset"] = tag
            break

    # Cap tag (e.g. "_100_" or "_300_")
    cap_match = re.search(r"_(\d{2,4})_", name)
    if cap_match:
        info["cap_tag"] = cap_match.group(1)

    # Model family
    for fam in MODEL_FAMILIES:
        if fam in name_lower:
            info["model_family"] = fam
            break

    # Backbone
    bb = BACKBONE_RE.search(name)
    if bb:
        info["backbone"] = bb.group(1)

    # Schedule
    sch = SCHEDULE_RE.search(name)
    if sch:
        info["schedule"] = sch.group(1)

    # Date and time at the end  (dd-mm-yyyy_hh-mm-ss)
    dt = DATETIME_RE.search(name)
    if dt:
        info["run_date"] = dt.group(1)
        info["run_time"] = dt.group(2)

    return info


# ── metrics.json parsing ────────────────────────────────────────────────────

# Per-class AP keys we look for (normalised to column-friendly names)
PER_CLASS_MAP = {
    "bbox/AP-E.coli":        "AP_E_coli",
    "bbox/AP-P.aeruginosa":  "AP_P_aeruginosa",
    "bbox/AP-S.aureus":      "AP_S_aureus",
}

# Core COCO metrics
CORE_METRICS = {
    "bbox/AP":   "AP",
    "bbox/AP50": "AP50",
    "bbox/AP75": "AP75",
    "bbox/APs":  "APs",
    "bbox/APm":  "APm",
    "bbox/APl":  "APl",
}


def parse_metrics_json(path: Path) -> dict:
    """Read metrics.json and return the final eval record.

    metrics.json is newline-delimited JSON.  Lines with ``bbox/AP`` are
    evaluation records; the rest are training-loss records.  We want the
    *last* evaluation record.

    Returns a dict with normalised keys (AP, AP50, …, AP_E_coli, …,
    eval_iteration) or an empty dict if nothing found.
    """
    last_eval = None
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if "bbox/AP" in record:
                    last_eval = record
    except Exception:
        return {}

    if last_eval is None:
        return {}

    result = {}

    # Iteration
    result["eval_iteration"] = last_eval.get("iteration", "")

    # Core COCO metrics
    for src_key, dst_col in CORE_METRICS.items():
        val = last_eval.get(src_key)
        result[dst_col] = round(val, 4) if val is not None else ""

    # Per-class APs
    for src_key, dst_col in PER_CLASS_MAP.items():
        val = last_eval.get(src_key)
        result[dst_col] = round(val, 4) if val is not None else ""

    # Catch any other per-class AP keys we haven't hard-coded
    for key, val in last_eval.items():
        if key.startswith("bbox/AP-") and key not in PER_CLASS_MAP:
            col = key.replace("bbox/", "").replace(".", "_").replace("-", "_")
            result[col] = round(val, 4) if val is not None else ""

    return result


# ── Artefact detection ───────────────────────────────────────────────────────

def check_artefacts(run_dir: Path) -> dict:
    """Check which artefact files exist in a run directory."""
    model_final = run_dir / "model_final.pth"
    inference_json = run_dir / "inference" / "coco_instances_results.json"
    instances_pth = run_dir / "inference" / "instances_predictions.pth"

    return {
        "has_model_final":   model_final.exists(),
        "has_inference_json": inference_json.exists(),
        "has_instances_pth": instances_pth.exists(),
        "model_final_path":  str(model_final) if model_final.exists() else "",
        "inference_json_path": str(inference_json) if inference_json.exists() else "",
        "instances_pth_path":  str(instances_pth) if instances_pth.exists() else "",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

# Fixed column order — dynamic per-class columns are appended at the end
FIXED_COLUMNS = [
    "run_dir",
    "run_name",
    "subset",
    "cap_tag",
    "model_family",
    "backbone",
    "schedule",
    "run_date",
    "run_time",
    "eval_iteration",
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AP_E_coli",
    "AP_P_aeruginosa",
    "AP_S_aureus",
    "has_model_final",
    "has_inference_json",
    "has_instances_pth",
    "metrics_json_path",
    "model_final_path",
    "inference_json_path",
    "instances_pth_path",
]


def find_run_dirs(runs_root: Path, selection_glob: str = "*"):
    """Yield run directories that contain a metrics.json."""
    for candidate in sorted(runs_root.glob(selection_glob)):
        if candidate.is_dir() and (candidate / "metrics.json").exists():
            yield candidate


def index_runs(runs_root: Path, selection_glob: str = "*"):
    """Index all run directories and return a list of row dicts."""
    rows = []
    extra_columns = set()

    for run_dir in find_run_dirs(runs_root, selection_glob):
        run_name = run_dir.name
        metrics_path = run_dir / "metrics.json"

        # Parse folder name
        name_info = parse_run_name(run_name)

        # Parse metrics
        metrics_info = parse_metrics_json(metrics_path)

        # Check artefacts
        artefacts = check_artefacts(run_dir)

        row = {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "metrics_json_path": str(metrics_path),
            **name_info,
            **metrics_info,
            **artefacts,
        }

        # Track any extra per-class columns
        for key in metrics_info:
            if key not in FIXED_COLUMNS:
                extra_columns.add(key)

        rows.append(row)

    return rows, sorted(extra_columns)


def write_csv(rows: list, extra_columns: list, out_csv: Path):
    """Write rows to CSV with a stable column order."""
    columns = FIXED_COLUMNS + extra_columns

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list):
    """Print a short summary of the indexed runs."""
    total = len(rows)
    has_eval = sum(1 for r in rows if r.get("AP") not in ("", None))
    missing_model = sum(1 for r in rows if not r.get("has_model_final"))
    missing_inference = sum(1 for r in rows if not r.get("has_inference_json"))

    print()
    print("=" * 50)
    print(f"  Total run dirs scanned:      {total}")
    print(f"  Runs with eval metrics:      {has_eval}")
    print(f"  Runs missing model_final:    {missing_model}")
    print(f"  Runs missing inference json: {missing_inference}")
    print("=" * 50)

    # Subset breakdown
    subsets = {}
    for r in rows:
        s = r.get("subset") or "(unknown)"
        subsets[s] = subsets.get(s, 0) + 1
    if subsets:
        print("\n  Subset breakdown:")
        for s in sorted(subsets):
            print(f"    {s:>12}: {subsets[s]}")

    # Model family breakdown
    families = {}
    for r in rows:
        f = r.get("model_family") or "(unknown)"
        families[f] = families.get(f, 0) + 1
    if families:
        print("\n  Model family breakdown:")
        for f in sorted(families):
            print(f"    {f:>15}: {families[f]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Index Detectron2 training run folders and write a CSV summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs_root",
        required=True,
        type=Path,
        help="Root folder containing run directories (e.g. /content/drive/MyDrive/TESE/outputs_detectron2)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        type=Path,
        help="Output CSV path",
    )
    parser.add_argument(
        "--selection_glob",
        default="*",
        help='Only include run dirs matching this glob pattern (default: "*")',
    )
    args = parser.parse_args()

    if not args.runs_root.is_dir():
        print(f"Error: runs_root does not exist or is not a directory: {args.runs_root}")
        raise SystemExit(1)

    print(f"Scanning: {args.runs_root}")
    print(f"Glob:     {args.selection_glob}")
    print(f"Output:   {args.out_csv}")

    rows, extra_columns = index_runs(args.runs_root, args.selection_glob)

    if not rows:
        print("\nNo run directories with metrics.json found.")
        raise SystemExit(0)

    write_csv(rows, extra_columns, args.out_csv)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")

    if extra_columns:
        print(f"Extra per-class columns detected: {extra_columns}")

    print_summary(rows)


if __name__ == "__main__":
    main()
