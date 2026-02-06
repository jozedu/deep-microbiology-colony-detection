"""
Ensemble utilities for combining predictions from multiple models.

Contains:
- Weighted Boxes Fusion (WBF) ensemble.
- NMS-based ensemble.
- Grid search for optimal WBF parameters.
- COCO evaluation of ensemble results.
"""

import os
import json

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import ensemble_boxes


# =============================================================================
# Prediction loading
# =============================================================================

def load_predictions(gt_path, prediction_paths):
    """Load ground truth and multiple prediction files for ensemble.

    Args:
        gt_path: Path to the COCO-format ground truth JSON.
        prediction_paths: List of paths to coco_instances_results.json files.

    Returns:
        coco_gt: COCO ground truth object.
        coco_dts: List of COCO detection result objects.
        img_ids: List of image IDs.
    """
    coco_gt = COCO(gt_path)
    coco_dts = [coco_gt.loadRes(p) for p in prediction_paths]
    img_ids = coco_gt.getImgIds()
    return coco_gt, coco_dts, img_ids


# =============================================================================
# Weighted Boxes Fusion
# =============================================================================

def run_wbf(coco_gt, coco_dts, img_ids, iou_thr=0.75,
            skip_box_thr=0.01, weights=None):
    """Apply Weighted Boxes Fusion to combine predictions from multiple models.

    Args:
        coco_gt: COCO ground truth object.
        coco_dts: List of COCO detection objects (one per model).
        img_ids: List of image IDs to process.
        iou_thr: IoU threshold for WBF.
        skip_box_thr: Minimum score to keep a box.
        weights: Per-model weights (list of floats, same length as coco_dts).

    Returns:
        list[dict]: Ensemble annotations in COCO format.
    """
    ensemble = []
    cnt_id = 0

    for img_id in tqdm(img_ids, desc="WBF Ensemble"):
        img_info = coco_gt.loadImgs(img_id)[0]
        height = float(img_info["height"])
        width = float(img_info["width"])

        boxes_list, scores_list, labels_list = [], [], []

        for coco_dt in coco_dts:
            boxes, scores, labels = [], [], []
            for ann in coco_dt.imgToAnns.get(img_id, []):
                x1, y1 = ann["bbox"][0], ann["bbox"][1]
                x2 = x1 + ann["bbox"][2]
                y2 = y1 + ann["bbox"][3]

                # Normalize to [0, 1] and clamp
                x1 = max(0.0, min(1.0, x1 / width))
                x2 = max(0.0, min(1.0, x2 / width))
                y1 = max(0.0, min(1.0, y1 / height))
                y2 = max(0.0, min(1.0, y2 / height))

                boxes.append([x1, y1, x2, y2])
                scores.append(ann["score"])
                labels.append(ann["category_id"])

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        fused_boxes, fused_scores, fused_labels = ensemble_boxes.weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            ensemble.append({
                "image_id": img_id,
                "category_id": label,
                "bbox": [x1 * width, y1 * height,
                         (x2 - x1) * width, (y2 - y1) * height],
                "score": float(score),
                "id": cnt_id,
            })
            cnt_id += 1

    return ensemble


def run_nms(coco_gt, coco_dts, img_ids, iou_thr=0.5, weights=None):
    """Apply NMS-based ensemble (single-model or multi-model).

    Args:
        coco_gt: COCO ground truth object.
        coco_dts: List of COCO detection objects.
        img_ids: List of image IDs.
        iou_thr: NMS IoU threshold.
        weights: Per-model weights.

    Returns:
        list[dict]: Ensemble annotations in COCO format.
    """
    ensemble = []
    cnt_id = 0

    for img_id in tqdm(img_ids, desc="NMS Ensemble"):
        img_info = coco_gt.loadImgs(img_id)[0]
        height = float(img_info["height"])
        width = float(img_info["width"])

        boxes_list, scores_list, labels_list = [], [], []

        for coco_dt in coco_dts:
            boxes, scores, labels = [], [], []
            for ann in coco_dt.imgToAnns.get(img_id, []):
                x1, y1 = ann["bbox"][0], ann["bbox"][1]
                x2 = x1 + ann["bbox"][2]
                y2 = y1 + ann["bbox"][3]

                x1 = max(0.0, min(1.0, x1 / width))
                x2 = max(0.0, min(1.0, x2 / width))
                y1 = max(0.0, min(1.0, y1 / height))
                y2 = max(0.0, min(1.0, y2 / height))

                boxes.append([x1, y1, x2, y2])
                scores.append(ann["score"])
                labels.append(ann["category_id"])

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        nms_boxes, nms_scores, nms_labels = ensemble_boxes.nms(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr,
        )

        for box, score, label in zip(nms_boxes, nms_scores, nms_labels):
            x1, y1, x2, y2 = box
            ensemble.append({
                "image_id": img_id,
                "category_id": label,
                "bbox": [x1 * width, y1 * height,
                         (x2 - x1) * width, (y2 - y1) * height],
                "score": float(score),
                "id": cnt_id,
            })
            cnt_id += 1

    return ensemble


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ensemble(coco_gt, ensemble_results, category_ids=None):
    """Run COCO evaluation on ensemble results.

    Args:
        coco_gt: COCO ground truth object.
        ensemble_results: List of prediction dicts from run_wbf or run_nms.
        category_ids: If set, evaluate only these category IDs.

    Returns:
        COCOeval object after summarize().
    """
    coco_ensemble = coco_gt.loadRes(ensemble_results)
    coco_eval = COCOeval(coco_gt, coco_ensemble, "bbox")

    if category_ids is not None:
        coco_eval.params.catIds = category_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def save_ensemble(ensemble_results, output_path):
    """Save ensemble annotations to a JSON file.

    Args:
        ensemble_results: List of prediction dicts.
        output_path: Output JSON file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ensemble_results, f)
    print(f"Ensemble annotations saved to {output_path}")


# =============================================================================
# Grid search for WBF parameters
# =============================================================================

def grid_search_wbf(coco_gt, coco_dts, img_ids,
                    iou_thresholds, skip_box_thresholds, weight_options,
                    output_dir=None):
    """Grid search over WBF parameters to find the best combination.

    Args:
        coco_gt: COCO ground truth object.
        coco_dts: List of COCO detection objects.
        img_ids: List of image IDs.
        iou_thresholds: List of IoU thresholds to try.
        skip_box_thresholds: List of skip_box_thr values to try.
        weight_options: List of weight lists to try.
        output_dir: If set, saves each ensemble result.

    Returns:
        list[dict]: Results with parameters and mAP for each combination.
    """
    from itertools import product

    results = []

    for iou_thr, skip_thr, weights in product(
        iou_thresholds, skip_box_thresholds, weight_options
    ):
        print(f"\n{'='*60}")
        print(f"WBF: iou_thr={iou_thr}, skip_box_thr={skip_thr}, weights={weights}")
        print(f"{'='*60}")

        ensemble = run_wbf(
            coco_gt, coco_dts, img_ids,
            iou_thr=iou_thr, skip_box_thr=skip_thr, weights=weights,
        )

        coco_eval = evaluate_ensemble(coco_gt, ensemble)
        mAP = coco_eval.stats[0]

        entry = {
            "iou_thr": iou_thr,
            "skip_box_thr": skip_thr,
            "weights": weights,
            "mAP": mAP,
        }
        results.append(entry)

        if output_dir:
            name = f"wbf_{iou_thr}_{skip_thr}_{weights}.json"
            save_ensemble(ensemble, os.path.join(output_dir, name))

    # Sort by mAP descending
    results.sort(key=lambda x: x["mAP"], reverse=True)
    print(f"\n{'='*60}")
    print(f"Best: mAP={results[0]['mAP']:.4f} with {results[0]}")
    print(f"{'='*60}")

    return results
