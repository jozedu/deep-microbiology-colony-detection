#!/usr/bin/env python3
"""
Index Detectron2 test-evaluation run folders and write CSV summaries.

Scans a root directory for Detectron2 output folders, parses test.txt for
evaluation results at ALL score thresholds, and writes:

  - **Long CSV** (``runs_index_long.csv``): one row per (run, threshold).
  - **Wide CSV** (``runs_index_wide.csv``, optional): one row per run with
    threshold-suffixed metric columns (e.g. ``AP_t05``).

Usage examples
--------------
# Index all runs and write long CSV only:
  python scripts/test_detectron2_runs.py \\
      --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
      --out_dir /content/drive/MyDrive/TESE/reports

# Also produce the wide CSV:
  python scripts/test_detectron2_runs.py \\
      --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
      --out_dir /content/drive/MyDrive/TESE/reports \\
      --make_wide

# Only bright-subset runs, specific thresholds:
  python scripts/test_detectron2_runs.py \\
      --runs_root /content/drive/MyDrive/TESE/outputs_detectron2 \\
      --out_dir /content/drive/MyDrive/TESE/reports \\
      --selection_glob "bright_*" \\
      --thresh_set "0.0,0.5"

# Local testing (workspace root as runs_root):
  python scripts/test_detectron2_runs.py \\
      --runs_root . \\
      --out_dir ./reports
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

SUBSET_TAGS = ("bright", "dark", "vague", "lowres", "total")
MODEL_FAMILIES = ("faster_rcnn", "mask_rcnn", "retinanet", "cascade_rcnn")

BACKBONE_RE = re.compile(r"(R_50|R_101|X_101)")
SCHEDULE_RE = re.compile(r"(?:^|_)(1x|3x)(?:_|$)")
DATETIME_RE = re.compile(r"(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})$")

# Regex to extract threshold + bbox dict from ``test.txt`` result tuples.
# Matches lines like:
#   (0.5, OrderedDict([('bbox', {'AP': 47.86, ..., 'AP-E.coli': 55.27})]))
RESULT_TUPLE_RE = re.compile(
    r"^\((\d+\.?\d*),\s*OrderedDict\(\[\('bbox',\s*(\{[^}]+\})\)\]\)\)",
    re.MULTILINE,
)

# Regex to extract AR values from raw COCO API output lines.
# Matches lines like:
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538
AR_LINE_RE = re.compile(
    r"Average Recall\s+\(AR\)\s+@\[\s*IoU=([\d.]+:[\d.]+)\s*\|"
    r"\s*area=\s*(\w+)\s*\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*([\d.]+)"
)

# Core bbox metric keys (order matters for CSV columns).
METRIC_KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl")

# Core AR metric keys.
AR_METRIC_KEYS = ("AR1", "AR10", "AR100", "ARs", "ARm", "ARl")

# Per-class AP normalisation: raw key → CSV column name.
PER_CLASS_NORM: dict[str, str] = {
    "AP-S.aureus":     "AP_S_aureus",
    "AP-P.aeruginosa": "AP_P_aeruginosa",
    "AP-E.coli":       "AP_E_coli",
}

# Canonical per-class column order.
PER_CLASS_COLUMNS = ("AP_E_coli", "AP_P_aeruginosa", "AP_S_aureus")

# All metric column names in canonical order.
ALL_METRIC_COLUMNS = list(METRIC_KEYS) + list(AR_METRIC_KEYS) + list(PER_CLASS_COLUMNS)

# Fixed column order for the long CSV.
LONG_COLUMNS = [
    "run_dir",
    "run_name",
    "subset",
    "cap_tag",
    "model_family",
    "backbone",
    "schedule",
    "run_date",
    "run_time",
    "test_score_thresh",
    # -- metrics --
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
    "AP_E_coli",
    "AP_P_aeruginosa",
    "AP_S_aureus",
    # -- paths --
    "test_txt_path",
    "inference_json_path",
    "instances_pth_path",
    "model_final_path",
    "metrics_json_path",
    # -- flags --
    "has_test_txt",
    "has_inference_json",
    "has_instances_pth",
    "has_model_final",
    "ci_ready",
]

# Metadata columns shared between long and wide formats.
META_COLUMNS = [
    "run_dir",
    "run_name",
    "subset",
    "cap_tag",
    "model_family",
    "backbone",
    "schedule",
    "run_date",
    "run_time",
]

ARTEFACT_COLUMNS = [
    "test_txt_path",
    "inference_json_path",
    "instances_pth_path",
    "model_final_path",
    "metrics_json_path",
    "has_test_txt",
    "has_inference_json",
    "has_instances_pth",
    "has_model_final",
    "ci_ready",
]


# ── Threshold suffix ────────────────────────────────────────────────────────

def thresh_suffix(t: float) -> str:
    """Deterministic suffix for a threshold value.

    0.0 → ``t0``,  0.25 → ``t025``,  0.5 → ``t05``,  0.75 → ``t075``.
    """
    s = str(t).replace(".", "")
    s = s.rstrip("0") or "0"
    return f"t{s}"


# ── Folder-name parsing ─────────────────────────────────────────────────────

def parse_run_name(name: str) -> dict[str, str]:
    """Extract metadata from a Detectron2 run-folder name.

    Returns a dict with keys: subset, cap_tag, model_family, backbone,
    schedule, run_date, run_time.  Never raises.
    """
    info: dict[str, str] = {
        "subset":       "unknown",
        "cap_tag":      "",
        "model_family": "unknown",
        "backbone":     "",
        "schedule":     "",
        "run_date":     "",
        "run_time":     "",
    }

    try:
        name_lower = name.lower()

        # Subset
        for tag in SUBSET_TAGS:
            if name_lower.startswith(tag):
                info["subset"] = tag
                break
        if info["subset"] == "unknown":
            for tag in SUBSET_TAGS:
                if tag in name_lower:
                    info["subset"] = tag
                    break

        # Cap tag (e.g. "_100_")
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

        # Date and time (dd-mm-yyyy_hh-mm-ss)
        dt = DATETIME_RE.search(name)
        if dt:
            info["run_date"] = dt.group(1)
            info["run_time"] = dt.group(2)
    except Exception:
        pass

    return info


# ── test.txt parsing ─────────────────────────────────────────────────────────

def _normalise_metric_key(raw: str) -> str:
    """Normalise a bbox metric key to a CSV-safe column name.

    ``'AP-E.coli'`` → ``'AP_E_coli'``.
    """
    if raw in PER_CLASS_NORM:
        return PER_CLASS_NORM[raw]
    return raw.replace(".", "_").replace("-", "_")


def _parse_ar_block(text_block: str) -> dict[str, object]:
    """Extract AR metrics from a block of raw COCO eval output.

    Looks for lines like::

        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538

    Returns dict with keys AR1, AR10, AR100, ARs, ARm, ARl (scaled to %).
    """
    ar: dict[str, object] = {k: "" for k in AR_METRIC_KEYS}

    for m in AR_LINE_RE.finditer(text_block):
        area = m.group(2)       # "all", "small", "medium", "large"
        max_dets = m.group(3)   # "1", "10", "100"
        value = float(m.group(4)) * 100  # COCO prints 0–1, we store %

        if area == "all":
            if max_dets == "1":
                ar["AR1"] = round(value, 4)
            elif max_dets == "10":
                ar["AR10"] = round(value, 4)
            elif max_dets == "100":
                ar["AR100"] = round(value, 4)
        elif area == "small" and max_dets == "100":
            ar["ARs"] = round(value, 4)
        elif area == "medium" and max_dets == "100":
            ar["ARm"] = round(value, 4)
        elif area == "large" and max_dets == "100":
            ar["ARl"] = round(value, 4)

    return ar


def parse_test_txt(path: Path) -> tuple[dict[float, dict[str, object]], str]:
    """Parse all evaluation blocks from a Detectron2 ``test.txt`` log.

    Extracts both AP (from the OrderedDict tuple) and AR (from the raw COCO
    API output lines that precede each tuple).

    Parameters
    ----------
    path : Path
        Full path to test.txt.

    Returns
    -------
    (results, error)
        *results*: mapping ``{threshold: {metric_col: value, ...}}``.
        *error*: empty string on success, or an error message.
    """
    results: dict[float, dict[str, object]] = {}

    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        return results, f"Cannot read file: {exc}"

    # Find all tuple matches with their positions so we can look backward
    # for the AR lines in the preceding COCO output.
    tuple_matches = list(RESULT_TUPLE_RE.finditer(text))
    if not tuple_matches:
        return results, "No evaluation tuples found"

    errors: list[str] = []
    for i, match in enumerate(tuple_matches):
        thresh_str = match.group(1)
        dict_str = match.group(2)

        try:
            threshold = float(thresh_str)
        except ValueError:
            errors.append(f"Bad threshold: {thresh_str!r}")
            continue

        try:
            bbox_dict = ast.literal_eval(dict_str)
        except (ValueError, SyntaxError) as exc:
            errors.append(f"Cannot parse bbox dict at threshold {thresh_str}: {exc}")
            continue

        if not isinstance(bbox_dict, dict):
            errors.append(f"Unexpected type for bbox dict at threshold {thresh_str}")
            continue

        row: dict[str, object] = {}

        # AP metrics from the OrderedDict
        for key in METRIC_KEYS:
            val = bbox_dict.get(key)
            row[key] = round(val, 4) if isinstance(val, (int, float)) else ""

        # AR metrics from the raw COCO output above this tuple
        block_start = tuple_matches[i - 1].end() if i > 0 else 0
        block_end = match.start()
        ar_block = text[block_start:block_end]
        ar_metrics = _parse_ar_block(ar_block)
        row.update(ar_metrics)

        # Per-class AP
        for raw_key, col in PER_CLASS_NORM.items():
            val = bbox_dict.get(raw_key)
            row[col] = round(val, 4) if isinstance(val, (int, float)) else ""

        # Auto-detect extra per-class keys
        for key in bbox_dict:
            if key.startswith("AP-") and key not in PER_CLASS_NORM and key not in METRIC_KEYS:
                col = _normalise_metric_key(key)
                val = bbox_dict[key]
                row[col] = round(val, 4) if isinstance(val, (int, float)) else ""

        results[threshold] = row

    error = "; ".join(errors) if errors else ""
    return results, error


# ── Artefact detection ───────────────────────────────────────────────────────

def check_artefacts(run_dir: Path) -> dict[str, object]:
    """Check which artefact files exist in a run directory."""
    test_txt       = run_dir / "test.txt"
    model_final    = run_dir / "model_final.pth"
    inference_json = run_dir / "inference" / "coco_instances_results.json"
    instances_pth  = run_dir / "inference" / "instances_predictions.pth"
    metrics_json   = run_dir / "metrics.json"

    return {
        "test_txt_path":       str(test_txt)       if test_txt.exists()       else "",
        "model_final_path":    str(model_final)    if model_final.exists()    else "",
        "inference_json_path": str(inference_json) if inference_json.exists() else "",
        "instances_pth_path":  str(instances_pth)  if instances_pth.exists()  else "",
        "metrics_json_path":   str(metrics_json)   if metrics_json.exists()   else "",
        "has_test_txt":        int(test_txt.exists()),
        "has_model_final":     int(model_final.exists()),
        "has_inference_json":  int(inference_json.exists()),
        "has_instances_pth":   int(instances_pth.exists()),
    }


# ── Discovery ───────────────────────────────────────────────────────────────

def find_run_dirs(runs_root: Path, selection_glob: str = "*"):
    """Yield directories that contain test.txt or metrics.json."""
    seen: set[Path] = set()

    # Search for test.txt files
    for f in sorted(runs_root.rglob("test.txt")):
        d = f.parent
        if d not in seen and d.is_dir() and fnmatch(d.name, selection_glob):
            seen.add(d)
            yield d

    # Also include dirs with metrics.json but no test.txt
    for f in sorted(runs_root.rglob("metrics.json")):
        d = f.parent
        if d not in seen and d.is_dir() and fnmatch(d.name, selection_glob):
            seen.add(d)
            yield d


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_runs(
    runs_root: Path,
    selection_glob: str,
    thresh_set: list[float],
) -> tuple[list[dict], list[str]]:
    """Index all run directories.

    Returns
    -------
    (long_rows, failures)
        *long_rows*: one dict per (run, threshold) for thresholds present in
        ``thresh_set``.
        *failures*: list of ``"run_name: error"`` strings.
    """
    long_rows: list[dict] = []
    failures: list[str] = []

    for run_dir in find_run_dirs(runs_root, selection_glob):
        run_name = run_dir.name
        name_info = parse_run_name(run_name)
        artefacts = check_artefacts(run_dir)

        # Parse test.txt
        test_txt_path = run_dir / "test.txt"
        if test_txt_path.exists():
            test_results, error = parse_test_txt(test_txt_path)
            if error:
                failures.append(f"{run_name}: {error}")
        else:
            test_results = {}

        has_inference = artefacts["has_inference_json"] == 1

        # Build one row per requested threshold
        found_any = False
        for t in thresh_set:
            metrics = test_results.get(t, {})
            if not metrics and t not in test_results:
                # Threshold not present in test.txt → emit row with blanks
                metrics = {}

            ap_val = metrics.get("AP", "")
            ci_ready = int(has_inference and ap_val != "")

            row = {
                "run_dir":  str(run_dir),
                "run_name": run_name,
                **name_info,
                "test_score_thresh": t,
                **{col: metrics.get(col, "") for col in ALL_METRIC_COLUMNS},
                **artefacts,
                "ci_ready": ci_ready,
            }
            long_rows.append(row)
            if metrics:
                found_any = True

        if not found_any and test_txt_path.exists():
            # test.txt exists but none of the requested thresholds matched
            found_thresholds = sorted(test_results.keys())
            failures.append(
                f"{run_name}: test.txt has thresholds {found_thresholds} "
                f"but none match --thresh_set"
            )

    return long_rows, failures


# ── CSV output ───────────────────────────────────────────────────────────────

def write_long_csv(rows: list[dict], out_path: Path) -> None:
    """Write the long-format CSV (one row per run-threshold)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LONG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_wide_csv(
    long_rows: list[dict],
    thresh_set: list[float],
    out_path: Path,
) -> None:
    """Write the wide-format CSV (one row per run, threshold-suffixed cols)."""
    # Group rows by run_dir
    runs: dict[str, dict] = {}
    for row in long_rows:
        key = row["run_dir"]
        if key not in runs:
            runs[key] = {col: row.get(col, "") for col in META_COLUMNS + ARTEFACT_COLUMNS}
        t = row["test_score_thresh"]
        sfx = thresh_suffix(t)
        for metric in ALL_METRIC_COLUMNS:
            runs[key][f"{metric}_{sfx}"] = row.get(metric, "")

    # Build column list
    wide_columns = list(META_COLUMNS)
    for t in thresh_set:
        sfx = thresh_suffix(t)
        for metric in ALL_METRIC_COLUMNS:
            wide_columns.append(f"{metric}_{sfx}")
    wide_columns.extend(ARTEFACT_COLUMNS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=wide_columns, extrasaction="ignore")
        writer.writeheader()
        for row in runs.values():
            writer.writerow(row)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(
    long_rows: list[dict],
    failures: list[str],
    total_dirs: int,
    dirs_with_test_txt: int,
) -> None:
    """Print a reviewer-oriented summary to stdout."""
    ci_ready_count = sum(1 for r in long_rows if r.get("ci_ready") == 1)
    rows_with_ap = sum(1 for r in long_rows if r.get("AP") not in ("", None))

    print()
    print("=" * 60)
    print(f"  Total run folders scanned:           {total_dirs}")
    print(f"  Runs with test.txt:                  {dirs_with_test_txt}")
    print(f"  Runs with parsing failures:          {len(failures)}")
    print(f"  Total (run, threshold) rows written: {len(long_rows)}")
    print(f"  Rows with AP (non-blank):            {rows_with_ap}")
    print(f"  CI-ready rows:                       {ci_ready_count}")
    print("=" * 60)

    if failures:
        print("\n  Parsing failures:")
        for msg in failures:
            print(f"    ⚠ {msg}")

    # Subset breakdown
    unique_runs: dict[str, dict] = {}
    for r in long_rows:
        key = r["run_dir"]
        if key not in unique_runs:
            unique_runs[key] = r

    subset_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for r in unique_runs.values():
        subset_counts[r.get("subset", "unknown")] += 1
        family_counts[r.get("model_family", "unknown")] += 1

    print("\n  Subset breakdown:")
    for s in sorted(subset_counts):
        print(f"    {s:>12}: {subset_counts[s]}")

    print("\n  Model family breakdown:")
    for f in sorted(family_counts):
        print(f"    {f:>15}: {family_counts[f]}")

    # Top-3 per subset at threshold 0.5
    t05_rows = [r for r in long_rows if r.get("test_score_thresh") == 0.5 and r.get("AP") not in ("", None)]
    if t05_rows:
        print("\n  Top-3 by AP at threshold 0.5 per subset:")
        by_subset: dict[str, list[dict]] = {}
        for r in t05_rows:
            by_subset.setdefault(r.get("subset", "unknown"), []).append(r)
        for subset in sorted(by_subset):
            ranked = sorted(by_subset[subset], key=lambda r: r["AP"], reverse=True)
            print(f"\n    [{subset}]")
            for i, r in enumerate(ranked[:3], 1):
                print(f"      {i}. {r['run_name']}  AP={r['AP']}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index Detectron2 test-evaluation run folders and write long/wide CSV summaries."
        ),
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
        "--out_dir",
        required=True,
        type=Path,
        help="Directory to write output CSVs",
    )
    parser.add_argument(
        "--selection_glob",
        default="*",
        help='Only include run dirs whose name matches this glob (default: "*")',
    )
    parser.add_argument(
        "--make_wide",
        action="store_true",
        default=False,
        help="Also write a wide-format CSV (one row per run)",
    )
    parser.add_argument(
        "--thresh_set",
        default="0.0,0.25,0.5,0.75",
        help=(
            "Comma-separated score thresholds to include "
            '(default: "0.0,0.25,0.5,0.75")'
        ),
    )
    args = parser.parse_args()

    # Parse threshold set
    try:
        thresh_set = [float(t.strip()) for t in args.thresh_set.split(",")]
    except ValueError:
        print(f"Error: invalid --thresh_set: {args.thresh_set!r}")
        raise SystemExit(1)

    if not args.runs_root.is_dir():
        print(f"Error: runs_root is not a directory: {args.runs_root}")
        raise SystemExit(1)

    long_csv = args.out_dir / "runs_index_long.csv"
    wide_csv = args.out_dir / "runs_index_wide.csv"

    print(f"Scanning:      {args.runs_root}")
    print(f"Glob:          {args.selection_glob}")
    print(f"Thresholds:    {thresh_set}")
    print(f"Long CSV:      {long_csv}")
    if args.make_wide:
        print(f"Wide CSV:      {wide_csv}")

    # Discover run directories
    all_dirs = list(find_run_dirs(args.runs_root, args.selection_glob))
    total_dirs = len(all_dirs)
    dirs_with_test_txt = sum(1 for d in all_dirs if (d / "test.txt").exists())

    if total_dirs == 0:
        print("\nNo run directories found.")
        raise SystemExit(0)

    # Index
    long_rows, failures = index_runs(args.runs_root, args.selection_glob, thresh_set)

    # Write outputs
    write_long_csv(long_rows, long_csv)
    print(f"\nWrote {len(long_rows)} rows → {long_csv}")

    if args.make_wide:
        write_wide_csv(long_rows, thresh_set, wide_csv)
        unique_runs = len({r["run_dir"] for r in long_rows})
        print(f"Wrote {unique_runs} rows → {wide_csv}")

    print_summary(long_rows, failures, total_dirs, dirs_with_test_txt)


if __name__ == "__main__":
    main()
