#!/usr/bin/env python3
"""
Index Detectron2 training run folders and write a CSV summary.

Recursively scans a root directory for Detectron2 output folders (as saved
during training on Google Colab), parses folder names for metadata, extracts
evaluation metrics from each metrics.json, and records which artefacts
(model weights, inference predictions) are present.

Usage examples
--------------
# Index all runs under a Google Drive mount (typical Colab usage):
  python scripts/index_detectron2_runs.py \\
    --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
    --out_csv /content/drive/MyDrive/TESE/results/runs_index.csv

# Use the best-AP evaluation record instead of the last:
  python scripts/index_detectron2_runs.py \\
    --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
    --out_csv best_ap_runs.csv \\
    --eval_policy best_ap

# Only bright-subset runs:
  python scripts/index_detectron2_runs.py \\
    --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
    --out_csv bright_runs.csv \\
    --selection_glob "bright_*"

# Local testing with debug self-check:
  python scripts/index_detectron2_runs.py \\
    --runs_root . --out_csv runs.csv --debug
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

SUBSET_TAGS = ("bright", "dark", "vague", "lowres", "total")
MODEL_FAMILIES = ("faster_rcnn", "mask_rcnn", "retinanet", "cascade_rcnn")

BACKBONE_RE = re.compile(r"(R_50|R_101|X_101)")
SCHEDULE_RE = re.compile(r"(?:^|_)(1x|3x)(?:_|$)")
DATETIME_RE = re.compile(r"(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})$")

EVAL_POLICIES = ("last", "best_ap")

# Core COCO metric keys → CSV column names
CORE_METRICS = {
    "bbox/AP":   "AP",
    "bbox/AP50": "AP50",
    "bbox/AP75": "AP75",
    "bbox/APs":  "APs",
    "bbox/APm":  "APm",
    "bbox/APl":  "APl",
}

# Known per-class AP keys → CSV column names
PER_CLASS_MAP = {
    "bbox/AP-E.coli":       "AP_E_coli",
    "bbox/AP-P.aeruginosa": "AP_P_aeruginosa",
    "bbox/AP-S.aureus":     "AP_S_aureus",
}

# Fixed column order (extra per-class columns are appended alphabetically)
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
    "eval_policy",
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
    "metrics_json_path",
    "model_final_path",
    "inference_json_path",
    "instances_pth_path",
    "has_model_final",
    "has_inference_json",
    "has_instances_pth",
]

_FIXED_SET = set(FIXED_COLUMNS)


# ── Folder-name parsing ─────────────────────────────────────────────────────

def parse_run_name(name: str) -> dict:
    """Extract metadata from a Detectron2 run-folder name.

    Robust: never raises; returns empty-string values for fields it cannot
    detect.

    Parameters
    ----------
    name : str
        Folder name, e.g. ``bright_100_faster_rcnn_R_50_FPN_3x_01-06-2023_08-21-12``
        or ``final_noaugm_retinanet_R_101_FPN_3x_22-06-2023_17-09-30``.

    Returns
    -------
    dict with keys: subset, cap_tag, model_family, backbone, schedule,
    run_date, run_time.
    """
    info: dict[str, str] = {
        "subset":       "",
        "cap_tag":      "",
        "model_family": "",
        "backbone":     "",
        "schedule":     "",
        "run_date":     "",
        "run_time":     "",
    }

    try:
        name_lower = name.lower()

        # Subset — check start first, then anywhere
        for tag in SUBSET_TAGS:
            if name_lower.startswith(tag):
                info["subset"] = tag
                break
        if not info["subset"]:
            for tag in SUBSET_TAGS:
                if tag in name_lower:
                    info["subset"] = tag
                    break

        # Cap tag  (e.g. "_100_" or "_300_")
        # Must not be preceded by R_ or X_ (those are backbone identifiers)
        cap_match = re.search(r"(?<!R)(?<!X)_(\d{2,4})_", name)
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
    except Exception:
        pass  # robustness: return whatever was parsed so far

    return info


# ── metrics.json parsing ────────────────────────────────────────────────────

def _normalise_key(raw: str) -> str:
    """Turn ``bbox/AP-E.coli`` into ``AP_E_coli``."""
    return raw.replace("bbox/", "").replace(".", "_").replace("-", "_")


def _extract_metrics(record: dict) -> dict:
    """Extract normalised metric columns from a single eval record."""
    result: dict = {}

    result["eval_iteration"] = record.get("iteration", "")

    # Core COCO metrics
    for src, dst in CORE_METRICS.items():
        val = record.get(src)
        result[dst] = round(val, 4) if isinstance(val, (int, float)) else ""

    # Known per-class APs
    for src, dst in PER_CLASS_MAP.items():
        val = record.get(src)
        result[dst] = round(val, 4) if isinstance(val, (int, float)) else ""

    # Auto-detect any other per-class keys
    for key in record:
        if key.startswith("bbox/AP-") and key not in PER_CLASS_MAP:
            col = _normalise_key(key)
            val = record[key]
            result[col] = round(val, 4) if isinstance(val, (int, float)) else ""

    return result


def parse_metrics_json(path: Path, eval_policy: str = "last") -> dict:
    """Read metrics.json and return the selected eval record.

    Parameters
    ----------
    path : Path
        Path to a newline-delimited JSON file.
    eval_policy : str
        ``"last"`` — use the last record containing ``bbox/AP``.
        ``"best_ap"`` — use the record with the highest ``bbox/AP``.

    Returns
    -------
    dict  with normalised keys (AP, AP50, …) or empty dict if no eval
    records are found.
    """
    selected = None

    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(record, dict) or "bbox/AP" not in record:
                    continue

                if eval_policy == "best_ap":
                    if selected is None or record["bbox/AP"] > selected["bbox/AP"]:
                        selected = record
                else:  # "last"
                    selected = record
    except Exception:
        return {}

    if selected is None:
        return {}

    return _extract_metrics(selected)


# ── Artefact detection ───────────────────────────────────────────────────────

def check_artefacts(run_dir: Path) -> dict:
    """Check which artefact files exist in a run directory.

    Boolean flags are stored as 0/1 integers for CSV friendliness.
    """
    model_final   = run_dir / "model_final.pth"
    inference_json = run_dir / "inference" / "coco_instances_results.json"
    instances_pth  = run_dir / "inference" / "instances_predictions.pth"

    return {
        "model_final_path":    str(model_final)   if model_final.exists()   else "",
        "inference_json_path": str(inference_json) if inference_json.exists() else "",
        "instances_pth_path":  str(instances_pth)  if instances_pth.exists()  else "",
        "has_model_final":     int(model_final.exists()),
        "has_inference_json":  int(inference_json.exists()),
        "has_instances_pth":   int(instances_pth.exists()),
    }


# ── Discovery ───────────────────────────────────────────────────────────────

def find_run_dirs(runs_root: Path, selection_glob: str = "*"):
    """Recursively yield directories that contain a metrics.json.

    The *selection_glob* is matched against the directory **name** (not the
    full path), so ``"bright_*"`` selects all directories whose name starts
    with ``bright_``.
    """
    seen: set[Path] = set()
    for metrics_file in sorted(runs_root.rglob("metrics.json")):
        run_dir = metrics_file.parent
        if run_dir in seen:
            continue
        if not run_dir.is_dir():
            continue
        # Apply selection glob to directory name
        if not _matches_glob(run_dir.name, selection_glob):
            continue
        seen.add(run_dir)
        yield run_dir


def _matches_glob(name: str, pattern: str) -> bool:
    """Check if *name* matches a simple glob *pattern* using pathlib."""
    from fnmatch import fnmatch
    return fnmatch(name, pattern)


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_runs(
    runs_root: Path,
    selection_glob: str = "*",
    eval_policy: str = "last",
) -> tuple[list[dict], list[str]]:
    """Index all run directories and return (rows, extra_columns)."""
    rows: list[dict] = []
    extra_columns: set[str] = set()

    for run_dir in find_run_dirs(runs_root, selection_glob):
        run_name = run_dir.name
        metrics_path = run_dir / "metrics.json"

        name_info    = parse_run_name(run_name)
        metrics_info = parse_metrics_json(metrics_path, eval_policy)
        artefacts    = check_artefacts(run_dir)

        row = {
            "run_dir":           str(run_dir),
            "run_name":          run_name,
            "eval_policy":       eval_policy,
            "metrics_json_path": str(metrics_path),
            **name_info,
            **metrics_info,
            **artefacts,
        }

        # Track extra per-class columns not in FIXED_COLUMNS
        for key in metrics_info:
            if key not in _FIXED_SET:
                extra_columns.add(key)

        rows.append(row)

    return rows, sorted(extra_columns)


# ── CSV output ───────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], extra_columns: list[str], out_csv: Path) -> None:
    """Write rows to CSV with a deterministic column order."""
    columns = FIXED_COLUMNS + extra_columns

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    """Print a short summary of the indexed runs to stdout."""
    total = len(rows)
    has_eval = sum(1 for r in rows if r.get("AP") not in ("", None))
    missing_model     = sum(1 for r in rows if r.get("has_model_final") == 0)
    missing_inference = sum(1 for r in rows if r.get("has_inference_json") == 0)

    print()
    print("=" * 54)
    print(f"  Total run dirs found:          {total}")
    print(f"  Runs with ≥1 eval record:      {has_eval}")
    print(f"  Runs missing model_final:      {missing_model}")
    print(f"  Runs missing inference json:   {missing_inference}")
    print("=" * 54)

    # Subset breakdown
    subset_counts: Counter[str] = Counter()
    for r in rows:
        subset_counts[r.get("subset") or "unknown"] += 1
    print("\n  Subset breakdown:")
    for s in sorted(subset_counts):
        print(f"    {s:>12}: {subset_counts[s]}")

    # Model family breakdown
    family_counts: Counter[str] = Counter()
    for r in rows:
        family_counts[r.get("model_family") or "unknown"] += 1
    print("\n  Model family breakdown:")
    for f in sorted(family_counts):
        print(f"    {f:>15}: {family_counts[f]}")
    print()


# ── Debug self-check ─────────────────────────────────────────────────────────

def _debug_selfcheck() -> None:
    """Parse two example folder names and print the results."""
    examples = [
        "bright_100_faster_rcnn_R_50_FPN_3x_01-06-2023_08-21-12",
        "final_noaugm_transferlearn_retinanet_R_101_FPN_3x_22-06-2023_17-09-30",
    ]
    print("─── parse_run_name self-check ──────────────────────")
    for name in examples:
        info = parse_run_name(name)
        print(f"\n  Input:  {name}")
        for k, v in info.items():
            print(f"    {k:>14}: {v!r}")
    print("────────────────────────────────────────────────────\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Detectron2 training run folders and write a CSV summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs_root",
        required=True,
        type=Path,
        help="Root folder containing run directories "
             "(e.g. /content/drive/MyDrive/TESE/outputs_detectron2)",
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
        help='Only include run dirs whose name matches this glob (default: "*")',
    )
    parser.add_argument(
        "--eval_policy",
        default="last",
        choices=EVAL_POLICIES,
        help='How to select the eval record: "last" (default) or "best_ap"',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print parse_run_name self-check for example folder names",
    )
    args = parser.parse_args()

    # Debug self-check
    if args.debug:
        _debug_selfcheck()

    if not args.runs_root.is_dir():
        print(f"Error: runs_root is not a directory: {args.runs_root}")
        raise SystemExit(1)

    print(f"Scanning:     {args.runs_root}")
    print(f"Glob:         {args.selection_glob}")
    print(f"Eval policy:  {args.eval_policy}")
    print(f"Output:       {args.out_csv}")

    rows, extra_columns = index_runs(
        args.runs_root, args.selection_glob, args.eval_policy,
    )

    if not rows:
        print("\nNo run directories with metrics.json found.")
        raise SystemExit(0)

    write_csv(rows, extra_columns, args.out_csv)
    print(f"\nWrote {len(rows)} rows → {args.out_csv}")

    if extra_columns:
        print(f"Extra per-class columns: {extra_columns}")

    print_summary(rows)


if __name__ == "__main__":
    main()
