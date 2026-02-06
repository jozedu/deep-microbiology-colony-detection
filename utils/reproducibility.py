"""
Reproducibility utilities for the microbial colony detection project.

Addresses reviewer concerns about missing configuration documentation by:
- Reconstructing and exporting the full resolved Detectron2 config for any
  trained model (anchors, augmentation, normalization, input sizes, etc.).
- Generating a human-readable reproducibility report.

Usage:
    from utils.reproducibility import dump_full_config, generate_reproducibility_report

    # Dump resolved config for a single model
    dump_full_config("faster_rcnn_R50", num_classes=3, output_path="config_dump.yaml")

    # Generate full report for a trained model directory
    generate_reproducibility_report(
        model_key="faster_rcnn_R50",
        model_dir="/path/to/output_dir",
        num_classes=3,
    )
"""

import os
import json
import yaml
from typing import Optional

from detectron2 import model_zoo
from detectron2.config import get_cfg


def build_resolved_cfg(model_key: str, num_classes: int = 3,
                       weights_path: Optional[str] = None,
                       train_dataset: str = "", val_dataset: str = "",
                       batch_size: int = 8, base_lr: float = 0.005,
                       max_iter: int = 0, checkpoint_period: int = 0,
                       score_thresh: float = 0.5,
                       detections_per_image: int = 100):
    """Build a fully resolved Detectron2 CfgNode with all defaults visible.

    This reconstructs the exact config used during training by merging the
    model zoo YAML with the user overrides, exposing every default
    (anchors, normalization, augmentation, input sizes, etc.).

    Args:
        model_key: Key into config.MODELS (e.g. 'faster_rcnn_R50').
        num_classes: Number of detection classes.
        weights_path: Path to model weights (optional, for documentation only).
        train_dataset: Registered training dataset name.
        val_dataset: Registered validation dataset name.
        batch_size: Images per batch.
        base_lr: Base learning rate.
        max_iter: Total training iterations (0 = leave as YAML default).
        checkpoint_period: Checkpoint save interval.
        score_thresh: Score threshold for test-time predictions.
        detections_per_image: Max detections per image at test time.

    Returns:
        CfgNode: The fully resolved config.
    """
    import config as project_config

    config_file = project_config.MODELS[model_key]
    is_retinanet = project_config.is_retinanet(model_key)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))

    # --- User overrides (exactly as used during training) ---
    if train_dataset:
        cfg.DATASETS.TRAIN = (train_dataset,)
    if val_dataset:
        cfg.DATASETS.TEST = (val_dataset,)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"

    if max_iter > 0:
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.STEPS = (int(0.7 * max_iter),)

    if checkpoint_period > 0:
        cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
        cfg.TEST.EVAL_PERIOD = checkpoint_period

    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image

    if is_retinanet:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    return cfg


def dump_full_config(model_key: str, num_classes: int = 3,
                     output_path: Optional[str] = None, **kwargs) -> str:
    """Dump the fully resolved Detectron2 config as a YAML string.

    Args:
        model_key: Key into config.MODELS.
        num_classes: Number of object classes.
        output_path: If provided, saves YAML to this path.
        **kwargs: Additional overrides passed to build_resolved_cfg().

    Returns:
        str: The full config as YAML text.
    """
    cfg = build_resolved_cfg(model_key, num_classes, **kwargs)
    yaml_str = cfg.dump()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_str)
        print(f"Full config saved to: {output_path}")

    return yaml_str


def extract_key_config_summary(model_key: str, num_classes: int = 3,
                               **kwargs) -> dict:
    """Extract a human-readable summary of the most important config values.

    Returns a dict covering exactly what reviewers asked for:
    anchors, augmentation, normalization, input sizes, optimizer, etc.
    """
    import config as project_config

    cfg = build_resolved_cfg(model_key, num_classes, **kwargs)
    is_retinanet = project_config.is_retinanet(model_key)

    summary = {
        "model": {
            "architecture": model_key,
            "config_file": project_config.MODELS[model_key],
            "backbone": cfg.MODEL.BACKBONE.NAME,
            "backbone_freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
            "fpn_in_features": list(cfg.MODEL.FPN.IN_FEATURES),
            "fpn_out_channels": cfg.MODEL.FPN.OUT_CHANNELS,
        },
        "anchor_generator": {
            "sizes": list(cfg.MODEL.ANCHOR_GENERATOR.SIZES),
            "aspect_ratios": list(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS),
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        },
        "input_preprocessing": {
            "pixel_mean": list(cfg.MODEL.PIXEL_MEAN),
            "pixel_std": list(cfg.MODEL.PIXEL_STD),
            "format": cfg.INPUT.FORMAT,
            "min_size_train": list(cfg.INPUT.MIN_SIZE_TRAIN),
            "max_size_train": cfg.INPUT.MAX_SIZE_TRAIN,
            "min_size_test": cfg.INPUT.MIN_SIZE_TEST,
            "max_size_test": cfg.INPUT.MAX_SIZE_TEST,
            "min_size_train_sampling": cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        },
        "data_augmentation_train": {
            "note": "Detectron2 DefaultTrainer defaults",
            "augmentations": [
                {
                    "name": "ResizeShortestEdge",
                    "short_edge_length": list(cfg.INPUT.MIN_SIZE_TRAIN),
                    "max_size": cfg.INPUT.MAX_SIZE_TRAIN,
                    "sample_style": cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                },
                {
                    "name": "RandomFlip",
                    "probability": 0.5,
                    "direction": "horizontal",
                },
            ],
            "custom_augmentations": "None — only Detectron2 defaults used",
        },
        "data_augmentation_inference": {
            "augmentations": [
                {
                    "name": "ResizeShortestEdge",
                    "short_edge_length": cfg.INPUT.MIN_SIZE_TEST,
                    "max_size": cfg.INPUT.MAX_SIZE_TEST,
                },
            ],
        },
        "optimizer": {
            "type": "SGD",
            "base_lr": cfg.SOLVER.BASE_LR,
            "momentum": cfg.SOLVER.MOMENTUM,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "lr_scheduler": cfg.SOLVER.LR_SCHEDULER_NAME,
            "lr_decay_steps": list(cfg.SOLVER.STEPS),
            "gamma": cfg.SOLVER.GAMMA,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "warmup_method": cfg.SOLVER.WARMUP_METHOD,
        },
        "training": {
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "max_iterations": cfg.SOLVER.MAX_ITER,
            "checkpoint_period": cfg.SOLVER.CHECKPOINT_PERIOD,
            "eval_period": cfg.TEST.EVAL_PERIOD,
        },
        "detection_head": {},
        "random_seed": {
            "seed": cfg.SEED if hasattr(cfg, "SEED") else "not set",
            "cudnn_benchmark": cfg.CUDNN_BENCHMARK if hasattr(cfg, "CUDNN_BENCHMARK") else "default",
            "note": "No explicit seed was set during training. "
                    "Results may vary across runs due to non-deterministic GPU operations.",
        },
    }

    # Detection head details depend on architecture
    if is_retinanet:
        summary["detection_head"] = {
            "type": "RetinaNet",
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "score_thresh_test": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
            "nms_thresh_test": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            "topk_candidates_test": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "num_convs": cfg.MODEL.RETINANET.NUM_CONVS,
        }
    else:
        summary["detection_head"] = {
            "type": "ROI Heads (Faster/Mask R-CNN)",
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "nms_thresh_test": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }
        summary["rpn"] = {
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "pre_nms_topk_train": cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            "pre_nms_topk_test": cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
            "post_nms_topk_train": cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            "post_nms_topk_test": cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
            "iou_thresholds": list(cfg.MODEL.RPN.IOU_THRESHOLDS),
            "iou_labels": list(cfg.MODEL.RPN.IOU_LABELS),
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
        }

    return summary


def generate_reproducibility_report(model_key: str, model_dir: str,
                                    num_classes: int = 3,
                                    save_yaml: bool = True,
                                    save_json: bool = True,
                                    **kwargs) -> dict:
    """Generate a full reproducibility report for a trained model.

    Creates:
    - full_config.yaml: Complete resolved Detectron2 config.
    - config_summary.json: Human-readable summary of key settings.

    Args:
        model_key: Key into config.MODELS.
        model_dir: Directory containing the trained model.
        num_classes: Number of classes.
        save_yaml: Save full config as YAML.
        save_json: Save summary as JSON.

    Returns:
        dict: The config summary.
    """
    report_dir = os.path.join(model_dir, "reproducibility")
    os.makedirs(report_dir, exist_ok=True)

    # Full resolved config → YAML
    if save_yaml:
        yaml_path = os.path.join(report_dir, "full_config.yaml")
        dump_full_config(model_key, num_classes, output_path=yaml_path, **kwargs)

    # Key summary → JSON
    summary = extract_key_config_summary(model_key, num_classes, **kwargs)
    if save_json:
        json_path = os.path.join(report_dir, "config_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Config summary saved to: {json_path}")

    return summary


def print_config_summary(summary: dict):
    """Pretty-print a config summary dict."""
    for section, values in summary.items():
        print(f"\n{'='*60}")
        print(f"  {section.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
