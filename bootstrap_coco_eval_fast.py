"""
bootstrap_coco_eval_fast.py
----------------------------
Drop-in replacement for bootstrap_coco_eval() and bootstrap_coco_eval_roboflow()
in postprocess_eval_results.ipynb.

Key optimisations vs. the original:
  1. No temp files — COCO objects are built in memory from dicts.
  2. Single COCOeval run per replicate — per-class AP is extracted from the
     internal precision array instead of running a separate COCOeval per category.

Both improvements together give roughly 10–20× speed-up, making 500 replicates
practical where 50 was slow before.

Usage (drop-in):
  from bootstrap_coco_eval_fast import bootstrap_coco_eval_fast as bootstrap_coco_eval
  from bootstrap_coco_eval_fast import bootstrap_coco_eval_roboflow_fast as bootstrap_coco_eval_roboflow
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coco_from_dict(dataset_dict):
    """Build a COCO object directly from a dict — no file I/O."""
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = dataset_dict
    coco.createIndex()
    return coco


def _extract_metrics(coco_eval, cat_ids, cat_names):
    """
    Extract overall and per-class AP from a single evaluated COCOeval object.

    coco_eval.eval['precision'] has shape [T, R, K, A, M]:
      T = IoU thresholds (10: 0.50..0.95)
      R = recall thresholds (101)
      K = categories
      A = area ranges (4: all, small, medium, large)
      M = max detections (3: 1, 10, 100)

    AP @ IoU=0.50:0.95, area=all, maxDets=100 → averaged over T, R for each K.
    """
    precision = coco_eval.eval['precision']  # shape [T, R, K, A, M]

    results = {
        'AP':   float(coco_eval.stats[0]) * 100,
        'AP50': float(coco_eval.stats[1]) * 100,
        'APs':  float(coco_eval.stats[3]) * 100,
    }

    for k_idx, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
        # precision[:, :, k_idx, 0, 2] → all IoU thresholds, all recalls,
        # this category, area=all, maxDets=100
        p = precision[:, :, k_idx, 0, 2]
        valid = p[p > -1]
        ap_cat = float(np.mean(valid)) * 100 if len(valid) > 0 else 0.0
        results[f'AP_{cat_name}'] = ap_cat

    return results


def _run_one_replicate(boot_gt_dict, boot_preds, cat_ids, cat_names):
    """Run COCOeval on one bootstrap sample entirely in memory."""
    from pycocotools.cocoeval import COCOeval

    # Suppress pycocotools stdout
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_gt = _coco_from_dict(boot_gt_dict)
        if not boot_preds:
            return None
        coco_dt = coco_gt.loadRes(boot_preds)

        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    finally:
        sys.stdout = old_stdout

    return _extract_metrics(ev, cat_ids, cat_names)


def _build_boot_sample(orig_img_ids, coco_gt, pred_by_image, rng):
    """
    Resample image IDs with replacement and build bootstrap GT dict + predictions.
    Image IDs are remapped to sequential integers to avoid duplicates confusing
    pycocotools when the same image is drawn more than once.
    """
    sampled_ids = rng.choice(orig_img_ids, size=len(orig_img_ids), replace=True)

    boot_images = []
    boot_anns = []
    boot_preds = []
    new_id = 1

    for orig_id in sampled_ids:
        orig_id = int(orig_id)
        img_info_list = coco_gt.loadImgs(orig_id)
        if not img_info_list:
            continue

        new_img = img_info_list[0].copy()
        new_img['id'] = new_id

        ann_ids = coco_gt.getAnnIds(imgIds=orig_id)
        for ann in coco_gt.loadAnns(ann_ids):
            new_ann = ann.copy()
            new_ann['image_id'] = new_id
            boot_anns.append(new_ann)

        for pred in pred_by_image.get(orig_id, []):
            new_pred = pred.copy()
            new_pred['image_id'] = new_id
            boot_preds.append(new_pred)

        boot_images.append(new_img)
        new_id += 1

    # Re-index annotation IDs (pycocotools requires unique ann IDs)
    for i, ann in enumerate(boot_anns):
        ann['id'] = i + 1

    boot_gt_dict = {
        'images':      boot_images,
        'annotations': boot_anns,
        'categories':  coco_gt.dataset['categories'],
    }

    return boot_gt_dict, boot_preds


# ─────────────────────────────────────────────────────────────────────────────
# Public API — AGAR subsets
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_coco_eval_fast(gt_json_path, pred_json_path, n_boot=500, seed=12345):
    """
    Bootstrap 95% CI for AP, AP50, APs, and per-class AP.

    Drop-in replacement for bootstrap_coco_eval() — same return format:
      {
        'AP':   {'mean': ..., 'ci_low': ..., 'ci_high': ...},
        'AP50': {...},
        'APs':  {...},
        'AP_E_coli': {...},
        ...
      }
    """
    from pycocotools.coco import COCO

    # ── Load GT ──────────────────────────────────────────────────────────────
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_gt = COCO(gt_json_path)
    finally:
        sys.stdout = old_stdout

    # ── Load predictions ─────────────────────────────────────────────────────
    with open(pred_json_path) as f:
        pred_list = json.load(f)

    if not pred_list:
        print(f"    [warn] Empty predictions file: {pred_json_path}")
        return None

    # ── Validate image ID overlap ─────────────────────────────────────────────
    gt_img_ids  = set(coco_gt.getImgIds())
    pred_img_ids = set(p['image_id'] for p in pred_list)
    if not (gt_img_ids & pred_img_ids):
        print(f"    [error] No image ID overlap between GT and predictions.")
        print(f"            GT sample:   {sorted(gt_img_ids)[:5]}")
        print(f"            Pred sample: {sorted(pred_img_ids)[:5]}")
        return None

    valid_ids = list(gt_img_ids & pred_img_ids)
    if len(valid_ids) < len(gt_img_ids):
        print(f"    [warn] {len(valid_ids)}/{len(gt_img_ids)} GT images have predictions.")

    # ── Category info ─────────────────────────────────────────────────────────
    cat_ids   = coco_gt.getCatIds()
    cat_names = [
        c['name'].replace(' ', '_').replace('.', '')
        for c in coco_gt.loadCats(cat_ids)
    ]

    # ── Index predictions by image_id ─────────────────────────────────────────
    pred_by_image = defaultdict(list)
    for p in pred_list:
        pred_by_image[int(p['image_id'])].append(p)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(valid_ids)

    accum = defaultdict(list)
    n_successful = 0

    for i in range(n_boot):
        boot_gt_dict, boot_preds = _build_boot_sample(
            orig_img_ids, coco_gt, pred_by_image, rng
        )
        if not boot_gt_dict['images'] or not boot_preds:
            continue

        metrics = _run_one_replicate(boot_gt_dict, boot_preds, cat_ids, cat_names)
        if metrics is None:
            continue

        for k, v in metrics.items():
            accum[k].append(v)
        n_successful += 1

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_boot} replicates done ({n_successful} successful)", end='\r')

    print(f"    {n_boot}/{n_boot} replicates done ({n_successful} successful)    ")

    if n_successful == 0:
        print("    [warn] All replicates failed.")
        return None

    # ── Summarise ─────────────────────────────────────────────────────────────
    return {
        metric: {
            'mean':     float(np.mean(vals)),
            'ci_low':   float(np.percentile(vals, 2.5)),
            'ci_high':  float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Roboflow / curated dataset
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_coco_eval_roboflow_fast(gt_json_path, pred_json_path, n_boot=500, seed=12345):
    """
    Bootstrap CI for the curated (Roboflow) dataset.

    Applies two Roboflow-specific fixes before bootstrapping:
      1. Removes the parent 'Colonies' class (category id=0) from GT.
      2. Remaps prediction image IDs from filenames to integer GT IDs if needed.

    Drop-in replacement for bootstrap_coco_eval_roboflow() — same return format.
    """
    # ── Load and patch GT ────────────────────────────────────────────────────
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    cat_info = {c['id']: c['name'] for c in gt_data['categories']}
    if 0 in cat_info and cat_info[0].lower() in ('colonies', 'colony'):
        print("    [info] Removing parent 'Colonies' class (id=0) from GT.")
        gt_data['categories']  = [c for c in gt_data['categories']  if c['id'] != 0]
        gt_data['annotations'] = [a for a in gt_data['annotations'] if a['category_id'] != 0]

    # ── Load predictions ─────────────────────────────────────────────────────
    with open(pred_json_path) as f:
        pred_list = json.load(f)

    if not pred_list:
        print(f"    [warn] Empty predictions file: {pred_json_path}")
        return None

    # ── Remap prediction image IDs if needed ──────────────────────────────────
    gt_img_ids   = {img['id'] for img in gt_data['images']}
    pred_img_ids = {p['image_id'] for p in pred_list}
    overlap = gt_img_ids & pred_img_ids

    if len(overlap) < len(pred_img_ids) * 0.5:
        print("    [info] Remapping prediction image IDs (filename → integer).")
        fname_to_id = {}
        for img in gt_data['images']:
            fname = img.get('file_name', '')
            fname_to_id[fname] = img['id']
            fname_to_id[os.path.splitext(fname)[0]] = img['id']

        remapped = 0
        for p in pred_list:
            orig = p['image_id']
            if orig in fname_to_id:
                p['image_id'] = fname_to_id[orig]
                remapped += 1
            elif isinstance(orig, str):
                base = os.path.splitext(orig)[0]
                if base in fname_to_id:
                    p['image_id'] = fname_to_id[base]
                    remapped += 1

        print(f"    [info] Remapped {remapped}/{len(pred_list)} predictions.")
        overlap = {p['image_id'] for p in pred_list} & gt_img_ids
        if not overlap:
            print("    [error] Remapping failed — no image ID overlap after fix.")
            return None

    # ── Build in-memory COCO GT ───────────────────────────────────────────────
    coco_gt = _coco_from_dict(gt_data)

    cat_ids   = coco_gt.getCatIds()
    cat_names = [
        c['name'].replace(' ', '_').replace('.', '')
        for c in coco_gt.loadCats(cat_ids)
    ]

    valid_ids = list({int(p['image_id']) for p in pred_list} & set(coco_gt.getImgIds()))
    if not valid_ids:
        print("    [error] No valid image IDs after remapping.")
        return None

    # ── Index predictions ─────────────────────────────────────────────────────
    pred_by_image = defaultdict(list)
    for p in pred_list:
        pred_by_image[int(p['image_id'])].append(p)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(valid_ids)

    accum = defaultdict(list)
    n_successful = 0

    for i in range(n_boot):
        boot_gt_dict, boot_preds = _build_boot_sample(
            orig_img_ids, coco_gt, pred_by_image, rng
        )
        if not boot_gt_dict['images'] or not boot_preds:
            continue

        metrics = _run_one_replicate(boot_gt_dict, boot_preds, cat_ids, cat_names)
        if metrics is None:
            continue

        for k, v in metrics.items():
            accum[k].append(v)
        n_successful += 1

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_boot} replicates done ({n_successful} successful)", end='\r')

    print(f"    {n_boot}/{n_boot} replicates done ({n_successful} successful)    ")

    if n_successful == 0:
        print("    [warn] All replicates failed.")
        return None

    return {
        metric: {
            'mean':    float(np.mean(vals)),
            'ci_low':  float(np.percentile(vals, 2.5)),
            'ci_high': float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }
