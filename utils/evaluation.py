"""
Evaluation utilities for microbial colony detection.

Contains:
- COCO mAP evaluation helpers.
- Colony counting evaluation with per-class metrics (AE, sAPE, MAE).
- Training metrics plotting (loss curves + AP).
- Bootstrap confidence intervals for COCO metrics (reviewer response).
- Multi-threshold / per-size COCO evaluation (reviewer response).
"""

import os
import json
import copy
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import ops


# =============================================================================
# COCO mAP evaluation
# =============================================================================

def evaluate_model(cfg, predictor, test_name, output_dir, max_dets=100):
    """Run COCO evaluation on a test set and return results.

    Args:
        cfg: Detectron2 config.
        predictor: DefaultPredictor instance.
        test_name: Registered dataset name for the test set.
        output_dir: Directory to save evaluation results.
        max_dets: Maximum detections per image.

    Returns:
        dict: COCO evaluation results.
    """
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    os.makedirs(output_dir, exist_ok=True)

    evaluator = COCOEvaluator(
        test_name, output_dir=output_dir, max_dets_per_image=max_dets
    )
    val_loader = build_detection_test_loader(cfg, test_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    return results


# =============================================================================
# Colony counting evaluation
# =============================================================================

def evaluate_bbox_counting(annotations_file, predictions_file,
                           score_threshold=0.5, iou_threshold=0.5,
                           output_dir=None, num_classes=3):
    """Evaluate bounding box counting accuracy per image and per class.

    Compares the number of predicted boxes against ground truth boxes,
    computing Absolute Error (AE), symmetric APE (sAPE), and per-class
    AE/MAE metrics.

    Args:
        annotations_file: Path to COCO-format ground truth JSON.
        predictions_file: Path to COCO-format predictions JSON.
        score_threshold: Minimum score to keep a prediction.
        iou_threshold: IoU threshold for NMS.
        output_dir: If set, saves results as an Excel file.
        num_classes: Number of object classes (default: 3).

    Returns:
        pd.DataFrame: Per-image counting metrics.
    """
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    # Count GT boxes per image and per class
    image_data = defaultdict(int)
    class_data = defaultdict(lambda: defaultdict(int))

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        image_data[image_id] += 1
        class_data[image_id][category_id] += 1

    all_image_ids = {img["id"] for img in data["images"]}
    image_ids = list(all_image_ids)

    num_boxes = [image_data[img_id] for img_id in image_ids]
    num_boxes_per_class = {
        c: [class_data[img_id][c] for img_id in image_ids]
        for c in range(num_classes)
    }

    # Process predictions
    image_counts = {}
    num_images = 0

    for prediction in predictions:
        image_id = prediction["image_id"]

        pred_boxes = torch.as_tensor(prediction["bbox"], dtype=torch.float32).unsqueeze(0)
        pred_scores = torch.as_tensor(prediction["score"], dtype=torch.float32).unsqueeze(0)
        pred_classes = torch.as_tensor(prediction["category_id"], dtype=torch.float32).unsqueeze(0)

        # Apply NMS
        nms_indices = ops.nms(pred_boxes, pred_scores, iou_threshold)
        pred_boxes = pred_boxes[nms_indices]
        pred_scores = pred_scores[nms_indices]
        pred_classes = pred_classes[nms_indices]

        # Filter by score threshold
        score_mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[score_mask]
        pred_scores = pred_scores[score_mask]
        pred_classes = pred_classes[score_mask]
        pred_bbox_count = pred_boxes.shape[0]

        if image_id not in image_counts:
            num_images += 1
            idx = image_ids.index(image_id)
            entry = {
                "Num Images": num_images,
                "Image ID": image_id,
                "GT Total Boxes": num_boxes[idx],
                "Pred Total Boxes": 0,
            }
            for c in range(num_classes):
                entry[f"GT Class {c}"] = num_boxes_per_class[c][idx]
                entry[f"Pred Class {c} Boxes"] = 0
            image_counts[image_id] = entry

        image_counts[image_id]["Pred Total Boxes"] += pred_bbox_count

        for c in range(num_classes):
            class_count = len([cl for cl in pred_classes if cl.item() == c])
            image_counts[image_id][f"Pred Class {c} Boxes"] += class_count

    # Build DataFrame with metrics
    df = pd.DataFrame(list(image_counts.values()))

    df["AE"] = abs(df["Pred Total Boxes"] - df["GT Total Boxes"])
    gt_plus_pred = abs(df["Pred Total Boxes"] + df["GT Total Boxes"])
    df["sAPE"] = df["AE"] / gt_plus_pred.replace(0, np.nan)

    for c in range(num_classes):
        df[f"c{c}_AE"] = abs(df[f"Pred Class {c} Boxes"] - df[f"GT Class {c}"])
        df[f"c{c}_MAE"] = df[f"c{c}_AE"] / df["Num Images"]

    df["Accumulated GT Total Boxes"] = df["GT Total Boxes"].sum()
    df["Accumulated Pred Total Boxes"] = df["Pred Total Boxes"].sum()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"count_score_{score_threshold}_iou_{iou_threshold}.xlsx"
        output_file = os.path.join(output_dir, filename)
        df.to_excel(output_file, index=False)
        print(f"Counting results saved to '{output_file}'")

    return df


# =============================================================================
# Training metrics plotting
# =============================================================================

def load_metrics(metrics_json_path):
    """Load Detectron2 metrics.json (one JSON object per line)."""
    lines = []
    with open(metrics_json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_training_curves(metrics_path, save_path=None):
    """Plot training loss, validation loss, and AP from metrics.json.

    Args:
        metrics_path: Path to the metrics.json file.
        save_path: If set, saves the plot as a PNG file.
    """
    metrics = load_metrics(metrics_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss curves
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.plot(
        [x["iteration"] for x in metrics if "total_loss" in x],
        [x["total_loss"] for x in metrics if "total_loss" in x],
        color="#3878a2", label="Training Loss",
    )
    ax1.plot(
        [x["iteration"] for x in metrics if "validation_loss" in x],
        [x["validation_loss"] for x in metrics if "validation_loss" in x],
        color="orange", label="Validation Loss",
    )
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    # AP curve (secondary axis)
    ap_entries = [x for x in metrics if "bbox/AP" in x]
    if ap_entries:
        ax2 = ax1.twinx()
        ax2.set_ylabel("AP")
        ax2.plot(
            [x["iteration"] for x in ap_entries],
            [x["bbox/AP"] for x in ap_entries],
            color="tab:green", label="AP",
        )
        ax2.tick_params(axis="y")
        ax2.legend(loc="upper right")

    plt.title("Training Curves")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# =============================================================================
# Bootstrap confidence intervals (reviewer response)
# =============================================================================

def bootstrap_coco_eval(gt_path: str, predictions_path: str,
                        n_bootstrap: int = 1000,
                        confidence_level: float = 0.95,
                        seed: int = 42,
                        max_dets: int = 100,
                        iou_thresholds: Optional[List[float]] = None,
                        ) -> dict:
    """Compute bootstrap confidence intervals for COCO mAP metrics.

    Resamples the test set (with replacement) N times and runs
    COCOeval on each resample to estimate variance of AP, AP50, AP75, AR.

    Args:
        gt_path: Path to COCO-format ground truth JSON.
        predictions_path: Path to COCO-format predictions JSON.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: CI level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        max_dets: Maximum detections per image.
        iou_thresholds: Custom IoU thresholds (None = COCO default 0.50:0.95).

    Returns:
        dict: {metric_name: {"mean": float, "std": float,
               "ci_low": float, "ci_high": float, "all_values": list}}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Load ground truth and predictions
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(predictions_path)

    all_img_ids = sorted(coco_gt.getImgIds())
    n_images = len(all_img_ids)

    rng = np.random.RandomState(seed)
    alpha = 1.0 - confidence_level

    # Metrics to track: indices into COCOeval.stats
    # [0]=AP@.50:.95, [1]=AP@.50, [2]=AP@.75,
    # [3]=AP-small, [4]=AP-medium, [5]=AP-large,
    # [6]=AR@1, [7]=AR@10, [8]=AR@maxDets,
    # [9]=AR-small, [10]=AR-medium, [11]=AR-large
    metric_names = [
        "AP", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", f"AR@{max_dets}",
        "AR_small", "AR_medium", "AR_large",
    ]
    all_stats = {name: [] for name in metric_names}

    print(f"Running {n_bootstrap} bootstrap iterations on {n_images} images...")

    for i in range(n_bootstrap):
        # Resample image IDs with replacement
        sampled_ids = rng.choice(all_img_ids, size=n_images, replace=True).tolist()

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sampled_ids
        coco_eval.params.maxDets = [1, 10, max_dets]

        if iou_thresholds is not None:
            coco_eval.params.iouThrs = np.array(iou_thresholds)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        for j, name in enumerate(metric_names):
            all_stats[name].append(coco_eval.stats[j])

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap}")

    # Compute summary statistics
    results = {}
    for name in metric_names:
        values = np.array(all_stats[name])
        results[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_low": float(np.percentile(values, 100 * alpha / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
            "median": float(np.median(values)),
            "all_values": values.tolist(),
        }

    return results


def format_ci_table(results: dict, confidence_level: float = 0.95) -> pd.DataFrame:
    """Format bootstrap CI results as a readable DataFrame.

    Args:
        results: Output from bootstrap_coco_eval().
        confidence_level: The confidence level used (for column header).

    Returns:
        pd.DataFrame with columns: Metric, Mean, Std, CI Low, CI High.
    """
    ci_pct = int(confidence_level * 100)
    rows = []
    for name, vals in results.items():
        rows.append({
            "Metric": name,
            "Mean": f"{vals['mean']:.4f}",
            "Std": f"{vals['std']:.4f}",
            f"{ci_pct}% CI Low": f"{vals['ci_low']:.4f}",
            f"{ci_pct}% CI High": f"{vals['ci_high']:.4f}",
        })
    return pd.DataFrame(rows)


def plot_bootstrap_distributions(results: dict, metrics: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
    """Plot histograms of bootstrap AP distributions.

    Args:
        results: Output from bootstrap_coco_eval().
        metrics: Which metrics to plot (default: AP, AP50, AP75).
        save_path: If set, save figure to this path.
    """
    if metrics is None:
        metrics = ["AP", "AP50", "AP75"]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, metrics):
        vals = results[name]["all_values"]
        mean = results[name]["mean"]
        ci_low = results[name]["ci_low"]
        ci_high = results[name]["ci_high"]

        ax.hist(vals, bins=40, alpha=0.7, color="#3878a2", edgecolor="white")
        ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean={mean:.4f}")
        ax.axvline(ci_low, color="orange", linestyle="--", linewidth=1.5, label=f"95% CI")
        ax.axvline(ci_high, color="orange", linestyle="--", linewidth=1.5)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.suptitle("Bootstrap Distribution of COCO Metrics", fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# =============================================================================
# Multi-threshold and per-size evaluation (reviewer response)
# =============================================================================

def multi_threshold_evaluate(gt_path: str, predictions_path: str,
                             iou_thresholds: Optional[List[float]] = None,
                             max_dets: int = 100,
                             per_category: bool = True) -> dict:
    """Run COCO evaluation at multiple IoU thresholds individually.

    This addresses reviewer concerns about IoU threshold justification
    for small colonies by reporting AP at each threshold separately.

    Args:
        gt_path: Path to COCO-format ground truth JSON.
        predictions_path: Path to COCO-format predictions JSON.
        iou_thresholds: List of IoU thresholds to evaluate at
            (default: [0.25, 0.5, 0.75, 0.9]).
        max_dets: Maximum detections per image.
        per_category: If True, also report per-category results.

    Returns:
        dict: {iou_threshold: {"AP": float, "AR": float,
               "per_category": {cat_id: AP}, "per_size": {...}}}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5, 0.75, 0.9]

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(predictions_path)
    cat_ids = sorted(coco_gt.getCatIds())

    results = {}

    for iou_thr in iou_thresholds:
        print(f"\n{'='*50}")
        print(f"IoU threshold = {iou_thr}")
        print(f"{'='*50}")

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([iou_thr])
        coco_eval.params.maxDets = [1, 10, max_dets]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        entry = {
            "AP": float(coco_eval.stats[0]),
            "AR": float(coco_eval.stats[8]),
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
        }

        # Per-category breakdown
        if per_category:
            entry["per_category"] = {}
            for cat_id in cat_ids:
                coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval_cat.params.iouThrs = np.array([iou_thr])
                coco_eval_cat.params.catIds = [cat_id]
                coco_eval_cat.params.maxDets = [1, 10, max_dets]
                coco_eval_cat.evaluate()
                coco_eval_cat.accumulate()
                coco_eval_cat.summarize()
                cat_name = coco_gt.cats[cat_id]["name"]
                entry["per_category"][cat_name] = float(coco_eval_cat.stats[0])

        results[iou_thr] = entry

    return results


def format_multi_threshold_table(results: dict) -> pd.DataFrame:
    """Format multi-threshold results as a DataFrame.

    Args:
        results: Output from multi_threshold_evaluate().

    Returns:
        pd.DataFrame with rows per IoU threshold.
    """
    rows = []
    for iou_thr, vals in sorted(results.items()):
        row = {
            "IoU Threshold": iou_thr,
            "AP": f"{vals['AP']*100:.1f}",
            "AR": f"{vals['AR']*100:.1f}",
            "AP_small": f"{vals['AP_small']*100:.1f}",
            "AP_medium": f"{vals['AP_medium']*100:.1f}",
            "AP_large": f"{vals['AP_large']*100:.1f}",
        }
        if "per_category" in vals:
            for cat, ap in vals["per_category"].items():
                row[f"AP_{cat}"] = f"{ap*100:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def size_distribution_analysis(gt_path: str, save_path: Optional[str] = None) -> dict:
    """Analyze the size distribution of ground truth bounding boxes.

    Categorizes annotations into COCO size buckets (small/medium/large)
    and computes area statistics to justify evaluation threshold choices.

    Args:
        gt_path: Path to COCO ground truth JSON.
        save_path: If set, save histogram plot.

    Returns:
        dict: Size distribution statistics.
    """
    from pycocotools.coco import COCO

    coco = COCO(gt_path)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    areas = np.array([ann["area"] for ann in anns])
    widths = np.array([ann["bbox"][2] for ann in anns])
    heights = np.array([ann["bbox"][3] for ann in anns])

    # COCO size definitions: small < 32², medium < 96², large >= 96²
    small_mask = areas < 32**2
    medium_mask = (areas >= 32**2) & (areas < 96**2)
    large_mask = areas >= 96**2

    stats = {
        "total_annotations": len(anns),
        "area": {
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas)),
            "std": float(np.std(areas)),
            "min": float(np.min(areas)),
            "max": float(np.max(areas)),
        },
        "width": {
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
        },
        "height": {
            "mean": float(np.mean(heights)),
            "median": float(np.median(heights)),
        },
        "coco_size_buckets": {
            "small (< 32²)": int(np.sum(small_mask)),
            "medium (32²–96²)": int(np.sum(medium_mask)),
            "large (≥ 96²)": int(np.sum(large_mask)),
        },
        "coco_size_percentages": {
            "small": f"{100 * np.sum(small_mask) / len(anns):.1f}%",
            "medium": f"{100 * np.sum(medium_mask) / len(anns):.1f}%",
            "large": f"{100 * np.sum(large_mask) / len(anns):.1f}%",
        },
    }

    if save_path or True:  # always plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Area distribution
        axes[0].hist(areas, bins=80, alpha=0.7, color="#3878a2", edgecolor="white")
        axes[0].axvline(32**2, color="red", linestyle="--", label="Small/Medium boundary (32²)")
        axes[0].axvline(96**2, color="orange", linestyle="--", label="Medium/Large boundary (96²)")
        axes[0].set_xlabel("Bounding Box Area (pixels²)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Annotation Area Distribution")
        axes[0].legend()

        # Width vs Height scatter
        axes[1].scatter(widths, heights, alpha=0.1, s=5, color="#3878a2")
        axes[1].set_xlabel("Width (pixels)")
        axes[1].set_ylabel("Height (pixels)")
        axes[1].set_title("Bounding Box Dimensions")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()

    return stats


def filter_sensitivity_analysis(
    gt_path: str,
    threshold: int = 100,
    save_path: Optional[str] = None,
) -> dict:
    """Analyze the impact of filtering images with >threshold annotations.

    Compares dataset statistics with and without the filter to quantify
    how much data is excluded and whether it biases the evaluation.

    Args:
        gt_path: Path to the UNFILTERED COCO annotation file
                 (e.g. annotations/new_cats/updated_total_train_annotations.json).
        threshold: Max annotations per image (images with more are filtered out).
        save_path: If set, save comparison plot.

    Returns:
        dict with keys 'original', 'filtered', 'excluded', and 'impact'.
    """
    from pycocotools.coco import COCO

    coco = COCO(gt_path)
    img_ids = coco.getImgIds()

    # Count annotations per image
    anns_per_img = {}
    all_areas_orig = []
    all_areas_filt = []
    all_areas_excl = []

    excluded_imgs = []
    kept_imgs = []

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        n_anns = len(anns)
        areas = [ann["area"] for ann in anns]
        anns_per_img[img_id] = n_anns
        all_areas_orig.extend(areas)

        if n_anns > threshold:
            excluded_imgs.append(img_id)
            all_areas_excl.extend(areas)
        else:
            kept_imgs.append(img_id)
            all_areas_filt.extend(areas)

    counts = list(anns_per_img.values())
    all_areas_orig = np.array(all_areas_orig)
    all_areas_filt = np.array(all_areas_filt) if all_areas_filt else np.array([])
    all_areas_excl = np.array(all_areas_excl) if all_areas_excl else np.array([])

    stats = {
        "threshold": threshold,
        "original": {
            "num_images": len(img_ids),
            "num_annotations": int(len(all_areas_orig)),
            "mean_anns_per_img": float(np.mean(counts)),
            "median_anns_per_img": float(np.median(counts)),
            "max_anns_per_img": int(np.max(counts)),
            "mean_area": float(np.mean(all_areas_orig)) if len(all_areas_orig) > 0 else 0,
            "median_area": float(np.median(all_areas_orig)) if len(all_areas_orig) > 0 else 0,
        },
        "filtered": {
            "num_images": len(kept_imgs),
            "num_annotations": int(len(all_areas_filt)),
            "mean_area": float(np.mean(all_areas_filt)) if len(all_areas_filt) > 0 else 0,
            "median_area": float(np.median(all_areas_filt)) if len(all_areas_filt) > 0 else 0,
        },
        "excluded": {
            "num_images": len(excluded_imgs),
            "num_annotations": int(len(all_areas_excl)),
            "pct_images_excluded": f"{100 * len(excluded_imgs) / len(img_ids):.1f}%",
            "pct_annotations_excluded": (
                f"{100 * len(all_areas_excl) / len(all_areas_orig):.1f}%"
                if len(all_areas_orig) > 0 else "0%"
            ),
            "mean_area": float(np.mean(all_areas_excl)) if len(all_areas_excl) > 0 else 0,
            "median_area": float(np.median(all_areas_excl)) if len(all_areas_excl) > 0 else 0,
        },
        "impact": {
            "area_shift": (
                float(np.mean(all_areas_filt) - np.mean(all_areas_orig))
                if len(all_areas_filt) > 0 else 0
            ),
            "area_shift_pct": (
                f"{100 * (np.mean(all_areas_filt) - np.mean(all_areas_orig)) / np.mean(all_areas_orig):.1f}%"
                if len(all_areas_orig) > 0 and np.mean(all_areas_orig) > 0 else "0%"
            ),
        },
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Annotations-per-image histogram
    axes[0].hist(counts, bins=50, alpha=0.7, color="#3878a2", edgecolor="white")
    axes[0].axvline(threshold, color="red", linewidth=2, linestyle="--",
                    label=f"Threshold = {threshold}")
    axes[0].set_xlabel("Annotations per Image")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_title("Annotations per Image Distribution")
    axes[0].legend()

    # (b) Area distributions: original vs filtered
    if len(all_areas_filt) > 0:
        axes[1].hist(all_areas_orig, bins=80, alpha=0.5, color="gray",
                     label=f"Original (n={len(all_areas_orig)})", density=True)
        axes[1].hist(all_areas_filt, bins=80, alpha=0.5, color="#3878a2",
                     label=f"Filtered ≤{threshold} (n={len(all_areas_filt)})", density=True)
        axes[1].set_xlabel("Area (px²)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Area Distribution Comparison")
        axes[1].legend()

    # (c) Summary text box
    axes[2].axis("off")
    text = (
        f"Filter: remove images with >{threshold} annotations\n\n"
        f"Original:  {stats['original']['num_images']} images, "
        f"{stats['original']['num_annotations']} annotations\n"
        f"Filtered:  {stats['filtered']['num_images']} images, "
        f"{stats['filtered']['num_annotations']} annotations\n"
        f"Excluded:  {stats['excluded']['num_images']} images "
        f"({stats['excluded']['pct_images_excluded']}), "
        f"{stats['excluded']['num_annotations']} annotations "
        f"({stats['excluded']['pct_annotations_excluded']})\n\n"
        f"Mean area shift: {stats['impact']['area_shift']:.1f} px² "
        f"({stats['impact']['area_shift_pct']})"
    )
    axes[2].text(0.05, 0.5, text, fontsize=12, fontfamily="monospace",
                 verticalalignment="center", transform=axes[2].transAxes,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[2].set_title("Filter Impact Summary")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()

    return stats