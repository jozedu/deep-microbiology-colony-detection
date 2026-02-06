"""
Centralized configuration for the microbial colony detection project.

This file contains all dataset paths, model definitions, hyperparameters,
and trained model output paths used throughout the project.

Modify the BASE_DIR and DRIVE_DIR variables to match your environment.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Base directories
# =============================================================================

# Google Colab: mount point for Google Drive
DRIVE_DIR = "/content/drive/MyDrive/TESE"

# Directory containing the AGAR dataset images
AGAR_IMG_DIR = os.path.join(DRIVE_DIR, "images")

# Directory for all detectron2 training outputs
OUTPUTS_DIR = os.path.join(DRIVE_DIR, "outputs_detectron2")

# Directory for ensemble and evaluation results
RESULTS_DIR = os.path.join(DRIVE_DIR, "results")


# =============================================================================
# Model architectures
# =============================================================================

MODELS = {
    "faster_rcnn_R50":  "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_R101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "retinanet_R50":    "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "retinanet_R101":   "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    "mask_rcnn_R50":    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "mask_rcnn_R101":   "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
}


def is_retinanet(model_key: str) -> bool:
    """Check if a model key refers to a RetinaNet architecture."""
    return "retinanet" in model_key.lower()


# =============================================================================
# Visualization colors (per class)
# =============================================================================

CLASS_COLORS = [
    (223, 141, 141),  # S. aureus  — soft red
    (126, 199, 173),  # P. aeruginosa — soft green
    (226, 228, 121),  # E. coli — soft yellow
]


# =============================================================================
# Training hyperparameters
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters for a single experiment."""

    # Dataset
    train_name: str = ""
    val_name: str = ""
    test_name: str = ""

    # Model
    model_key: str = "faster_rcnn_R50"  # key into MODELS dict
    num_classes: int = 3

    # Optimizer
    batch_size: int = 8
    base_lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005

    # Schedule
    num_epochs: int = 10
    num_train_images: int = 0  # set after loading dataset
    warmup_iters: int = 1000
    warmup_factor: float = 1.0 / 1000
    warmup_method: str = "linear"

    # RoI / detection head
    roi_batch_size_per_image: int = 512
    score_thresh_test: float = 0.5
    detections_per_image: int = 100
    filter_empty_annotations: bool = False

    # Checkpointing
    checkpoint_period: int = 0  # auto-computed if 0
    eval_period: int = 0        # auto-computed if 0

    # Pretrained weights (path or model_zoo URL)
    weights: Optional[str] = None  # None = use model_zoo defaults

    # Output
    output_dir: str = ""

    @property
    def config_file(self) -> str:
        return MODELS[self.model_key]

    @property
    def max_iter(self) -> int:
        if self.num_train_images == 0:
            raise ValueError("Set num_train_images before computing max_iter")
        return self.num_epochs * self.num_train_images // self.batch_size

    @property
    def lr_decay_steps(self) -> List[int]:
        """Decay LR at 70% and 90% of training."""
        m = self.max_iter
        return [int(0.7 * m), int(0.9 * m)]

    def auto_checkpoint_period(self) -> int:
        """Save checkpoint once per epoch."""
        return self.num_train_images // self.batch_size


# =============================================================================
# Part 1: AGAR dataset annotation paths
# =============================================================================

_AGAR_ANN_DIR = os.path.join(
    DRIVE_DIR, "annotations/less_100_anns_img/full_datasets/val_test_by_imgs"
)

AGAR_DATASETS = {
    "total": {
        "train": os.path.join(_AGAR_ANN_DIR, "train_total100.json"),
        "val":   os.path.join(_AGAR_ANN_DIR, "val_total100.json"),
        "test":  os.path.join(_AGAR_ANN_DIR, "test_total100.json"),
    },
    "bright": {
        "train": os.path.join(_AGAR_ANN_DIR, "train_bright100.json"),
        "val":   os.path.join(_AGAR_ANN_DIR, "val_bright100.json"),
        "test":  os.path.join(_AGAR_ANN_DIR, "test_bright100.json"),
    },
    "dark": {
        "train": os.path.join(_AGAR_ANN_DIR, "train_dark100.json"),
        "val":   os.path.join(_AGAR_ANN_DIR, "val_dark100.json"),
        "test":  os.path.join(_AGAR_ANN_DIR, "test_dark100.json"),
    },
    "vague": {
        "train": os.path.join(_AGAR_ANN_DIR, "train_vague100.json"),
        "val":   os.path.join(_AGAR_ANN_DIR, "val_vague100.json"),
        "test":  os.path.join(_AGAR_ANN_DIR, "test_vague100.json"),
    },
    "lowres": {
        "train": os.path.join(_AGAR_ANN_DIR, "train_lowres100.json"),
        "val":   os.path.join(_AGAR_ANN_DIR, "val_lowres100.json"),
        "test":  os.path.join(_AGAR_ANN_DIR, "test_lowres100.json"),
    },
}


# Unfiltered annotation paths (before >100 annotation filter was applied)
# Used for filter sensitivity analysis
_AGAR_UNFILTERED_DIR = os.path.join(DRIVE_DIR, "annotations/new_cats")

AGAR_UNFILTERED = {
    "total_train": os.path.join(_AGAR_UNFILTERED_DIR, "updated_total_train_annotations.json"),
    "total_test":  os.path.join(_AGAR_UNFILTERED_DIR, "updated_total_test_annotations.json"),
    "bright_train": os.path.join(_AGAR_UNFILTERED_DIR, "updated_bright_train_annotations.json"),
    "bright_test":  os.path.join(_AGAR_UNFILTERED_DIR, "updated_bright_test_annotations.json"),
    "dark_train":   os.path.join(_AGAR_UNFILTERED_DIR, "updated_dark_train_annotations.json"),
    "dark_test":    os.path.join(_AGAR_UNFILTERED_DIR, "updated_dark_test_annotations.json"),
    "vague_train":  os.path.join(_AGAR_UNFILTERED_DIR, "updated_vague_train_annotations.json"),
    "vague_test":   os.path.join(_AGAR_UNFILTERED_DIR, "updated_vague_test_annotations.json"),
    "lowres_train": os.path.join(_AGAR_UNFILTERED_DIR, "updated_lowres_train_annotations.json"),
    "lowres_test":  os.path.join(_AGAR_UNFILTERED_DIR, "updated_lowres_test_annotations.json"),
}


# =============================================================================
# Part 2: New (Roboflow) dataset paths
# =============================================================================

ROBOFLOW_DATASETS = {
    "curated": {
        "train":     "/content/train/_annotations.coco.json",
        "val":       "/content/valid/_annotations.coco.json",
        "test":      "/content/test/_annotations.coco.json",
        "train_dir": "/content/train",
        "val_dir":   "/content/valid",
        "test_dir":  "/content/test",
    },
}

# Roboflow download URL for the curated dataset (well-balanced, no augmentation)
ROBOFLOW_DOWNLOAD_URL = "https://app.roboflow.com/ds/YSxWunE7dO?key=ggWKkflKfV"


# =============================================================================
# Part 1: Trained model output paths (for evaluation / ensemble / transfer)
# =============================================================================

AGAR_TRAINED_MODELS = {
    # --- Bright subset ---
    "bright_faster_rcnn_R50":       os.path.join(OUTPUTS_DIR, "bright_100_faster_rcnn_R_50_FPN_3x_01-06-2023_08-21-12"),
    "bright_faster_rcnn_R101":      os.path.join(OUTPUTS_DIR, "bright_100_faster_rcnn_R_101_FPN_3x_01-06-2023_16-34-26"),
    "bright_retinanet_R50":         os.path.join(OUTPUTS_DIR, "bright_100_retinanet_R_50_FPN_3x_05-06-2023_19-24-07"),
    "bright_retinanet_R101":        os.path.join(OUTPUTS_DIR, "bright_100_retinanet_R_101_FPN_3x_12-06-2023_16-35-32"),
    "bright_transfer_faster_R101":  os.path.join(OUTPUTS_DIR, "bright_100_transferlearn_faster_rcnn_R_101_FPN_3x_15-06-2023_21-41-45"),
    "bright_mask_rcnn_R50":         os.path.join(OUTPUTS_DIR, "bright_100_mask_rcnn_R_50_FPN_3x_22-06-2023_12-54-36"),
    "bright_mask_rcnn_R101":        os.path.join(OUTPUTS_DIR, "bright_100_mask_rcnn_R_101_FPN_3x_22-06-2023_18-47-20"),
    # --- Dark subset ---
    "dark_faster_rcnn_R50":         os.path.join(OUTPUTS_DIR, "dark_100_faster_rcnn_R_50_FPN_3x_02-06-2023_08-58-47"),
    "dark_faster_rcnn_R101":        os.path.join(OUTPUTS_DIR, "dark_100_faster_rcnn_R_101_FPN_3x_11-06-2023_18-37-41"),
    "dark_retinanet_R50":           os.path.join(OUTPUTS_DIR, "dark_100_retinanet_R_50_FPN_3x_07-06-2023_09-36-08"),
    "dark_retinanet_R101":          os.path.join(OUTPUTS_DIR, "dark_100_retinanet_R_101_FPN_3x_08-06-2023_13-01-51"),
    "dark_mask_rcnn_R50":           os.path.join(OUTPUTS_DIR, "dark_100_mask_rcnn_R_50_FPN_3x_23-06-2023_16-50-56"),
    "dark_mask_rcnn_R101":          os.path.join(OUTPUTS_DIR, "dark_100_mask_rcnn_R_101_FPN_3x_23-06-2023_19-32-10"),
    # --- Vague subset ---
    "vague_faster_rcnn_R50":        os.path.join(OUTPUTS_DIR, "vague_100_faster_rcnn_R_50_FPN_3x_01-06-2023_09-33-20"),
    "vague_faster_rcnn_R101":       os.path.join(OUTPUTS_DIR, "vague_100_faster_rcnn_R_101_FPN_3x_01-06-2023_18-05-59"),
    "vague_retinanet_R50":          os.path.join(OUTPUTS_DIR, "vague_100_retinanet_R_50_FPN_3x_05-06-2023_21-11-22"),
    "vague_retinanet_R101":         os.path.join(OUTPUTS_DIR, "vague_100_retinanet_R_101_FPN_3x_12-06-2023_18-34-26"),
    "vague_transfer_faster_R101":   os.path.join(OUTPUTS_DIR, "vague_100_transferlearn_faster_rcnn_R_101_FPN_3x_16-06-2023_10-52-00"),
    "vague_mask_rcnn_R50":          os.path.join(OUTPUTS_DIR, "vague_100_mask_rcnn_R_50_FPN_3x_22-06-2023_16-04-26"),
    "vague_mask_rcnn_R101":         os.path.join(OUTPUTS_DIR, "vague_100_mask_rcnn_R_101_FPN_3x_23-06-2023_15-55-51"),
    # --- Lowres subset ---
    "lowres_faster_rcnn_R50":       os.path.join(OUTPUTS_DIR, "lowres_100_faster_rcnn_R_50_FPN_3x_01-06-2023_10-46-18"),
    "lowres_faster_rcnn_R101":      os.path.join(OUTPUTS_DIR, "lowres_100_faster_rcnn_R_101_FPN_3x_01-06-2023_19-12-17"),
    "lowres_retinanet_R50":         os.path.join(OUTPUTS_DIR, "lowres_100_retinanet_R_50_FPN_3x_06-06-2023_16-51-07"),
    "lowres_retinanet_R101":        os.path.join(OUTPUTS_DIR, "lowres_100_retinanet_R_101_FPN_3x_12-06-2023_19-35-20"),
    "lowres_transfer_faster_R101":  os.path.join(OUTPUTS_DIR, "lowres_100_transferlearn_faster_rcnn_R_101_FPN_3x_16-06-2023_12-36-13"),
    "lowres_mask_rcnn_R50":         os.path.join(OUTPUTS_DIR, "lowres_100_mask_rcnn_R_50_FPN_3x_22-06-2023_17-00-48"),
    "lowres_mask_rcnn_R101":        os.path.join(OUTPUTS_DIR, "lowres_100_mask_rcnn_R_101_FPN_3x_24-06-2023_13-04-32"),
    # --- Total dataset ---
    "total_faster_rcnn_R50":        os.path.join(OUTPUTS_DIR, "total_100_faster_rcnn_R_50_FPN_3x_06-06-2023_09-17-10"),
    "total_faster_rcnn_R101":       os.path.join(OUTPUTS_DIR, "total_100_faster_rcnn_R_101_FPN_3x_13-06-2023_12-11-22"),
    "total_retinanet_R50":          os.path.join(OUTPUTS_DIR, "total_100_retinanet_R_50_FPN_3x_09-06-2023_10-01-15"),
    "total_retinanet_R101":         os.path.join(OUTPUTS_DIR, "total_100_retinanet_R_101_FPN_3x_10-06-2023_12-01-22"),
    "total_mask_rcnn_R50":          os.path.join(OUTPUTS_DIR, "total_100_mask_rcnn_R_50_FPN_3x_23-06-2023_08-44-11"),
    "total_mask_rcnn_R101":         os.path.join(OUTPUTS_DIR, "total_100_mask_rcnn_R_101_FPN_3x_22-06-2023_19-15-13"),
}


# =============================================================================
# Part 2: Trained model output paths (Roboflow / curated dataset)
# =============================================================================

ROBOFLOW_TRAINED_MODELS = {
    "robo_faster_rcnn_R50":              os.path.join(OUTPUTS_DIR, "final_noaugm_faster_rcnn_R_50_FPN_3x_24-07-2023_19-38-03"),
    "robo_faster_rcnn_R101":             os.path.join(OUTPUTS_DIR, "final_noaugm_faster_rcnn_R_101_FPN_3x_24-07-2023_20-39-39"),
    "robo_retinanet_R50":                os.path.join(OUTPUTS_DIR, "final_noaugm_retinanet_R_50_FPN_3x_24-07-2023_22-06-00"),
    "robo_retinanet_R101":               os.path.join(OUTPUTS_DIR, "final_noaugm_retinanet_R_101_FPN_3x_24-07-2023_22-45-13"),
    "robo_transfer_faster_rcnn_R50":     os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_faster_rcnn_R_50_FPN_3x_25-07-2023_08-38-00"),
    "robo_transfer_faster_rcnn_R101":    os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_faster_rcnn_R_101_FPN_3x_25-07-2023_14-06-36"),
    "robo_transfer_retinanet_R50":       os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_retinanet_R_50_FPN_3x_25-07-2023_07-51-03"),
    "robo_transfer_retinanet_R101":      os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_retinanet_R_101_FPN_3x_25-07-2023_15-10-00"),
    "robo_transfer_lowres_retinanet_R50": os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_retinanet_R_50_FPN_3x_27-08-2023_22-54-12"),
    "robo_transfer_dark_retinanet_R50":  os.path.join(OUTPUTS_DIR, "final_noaugm_transferlearn_dark_retinanet_R_50_FPN_3x_28-08-2023_13-14-01"),
}


# =============================================================================
# Helper: get model_final.pth path from a trained model key
# =============================================================================

def get_model_weights(model_key: str, source: str = "agar") -> str:
    """Return the path to model_final.pth for a trained model.

    Args:
        model_key: Key into AGAR_TRAINED_MODELS or ROBOFLOW_TRAINED_MODELS.
        source: 'agar' or 'roboflow'.
    """
    registry = AGAR_TRAINED_MODELS if source == "agar" else ROBOFLOW_TRAINED_MODELS
    if model_key not in registry:
        raise KeyError(f"Model '{model_key}' not found in {source} registry. "
                       f"Available: {list(registry.keys())}")
    return os.path.join(registry[model_key], "model_final.pth")


def get_predictions_path(model_key: str, source: str = "agar",
                         subfolder: str = "test") -> str:
    """Return the path to coco_instances_results.json for a trained model."""
    registry = AGAR_TRAINED_MODELS if source == "agar" else ROBOFLOW_TRAINED_MODELS
    return os.path.join(registry[model_key], subfolder, "coco_instances_results.json")
