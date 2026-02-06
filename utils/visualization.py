"""
Visualization utilities for microbial colony detection.

Contains helpers for:
- Displaying annotated dataset samples.
- Displaying model predictions.
- Setting consistent visualization colors.
"""

import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from config import CLASS_COLORS


# =============================================================================
# Metadata setup
# =============================================================================

def setup_metadata_colors(dataset_name):
    """Apply consistent class colors to a dataset's metadata.

    Args:
        dataset_name: Registered Detectron2 dataset name.

    Returns:
        Metadata catalog entry with colors applied.
    """
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_colors = CLASS_COLORS
    metadata.stuff_colors = CLASS_COLORS
    return metadata


# =============================================================================
# Dataset visualization
# =============================================================================

def show_dataset_samples(dataset_name, num_samples=5, scale=0.5, seed=None):
    """Display random annotated samples from a registered dataset.

    Args:
        dataset_name: Registered Detectron2 dataset name.
        num_samples: Number of random images to display.
        scale: Image display scale.
        seed: Random seed for reproducibility.
    """
    metadata = setup_metadata_colors(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    if seed is not None:
        random.seed(seed)

    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))

    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 6))
    if len(samples) == 1:
        axes = [axes]

    for ax, d in zip(axes, samples):
        img = cv2.imread(d["file_name"])
        if img is None:
            ax.set_title("Image not found")
            continue

        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out = visualizer.draw_dataset_dict(d)
        ax.imshow(out.get_image())
        ax.axis("off")
        ax.set_title(os.path.basename(d["file_name"]))

    plt.tight_layout()
    plt.show()


def show_specific_image(dataset_name, image_filename, img_dir, scale=0.5):
    """Display a specific image from the dataset with its annotations.

    Args:
        dataset_name: Registered Detectron2 dataset name.
        image_filename: Filename (e.g. '429.jpg').
        img_dir: Base image directory.
        scale: Image display scale.
    """
    metadata = setup_metadata_colors(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    img_path = os.path.join(img_dir, image_filename)

    selected = None
    for d in dataset_dicts:
        if d["file_name"] == img_path:
            selected = d
            break

    if selected is None:
        print(f"Image '{image_filename}' not found in dataset '{dataset_name}'.")
        return

    img = cv2.imread(selected["file_name"])
    visualizer = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
    )
    out = visualizer.draw_dataset_dict(selected)

    plt.figure(figsize=(12, 8))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.title(image_filename)
    plt.show()


# =============================================================================
# Prediction visualization
# =============================================================================

def show_predictions(predictor, dataset_name, num_samples=5,
                     scale=0.5, seed=None):
    """Run inference and display predictions on random test images.

    Args:
        predictor: DefaultPredictor instance.
        dataset_name: Registered test dataset name.
        num_samples: Number of images to display.
        scale: Image display scale.
        seed: Random seed for reproducibility.
    """
    metadata = setup_metadata_colors(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    if seed is not None:
        random.seed(seed)

    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))

    for d in samples:
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Could not read {d['file_name']}")
            continue

        outputs = predictor(img)

        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=scale,
            instance_mode=ColorMode.IMAGE,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(12, 8))
        plt.imshow(out.get_image())
        plt.axis("off")
        plt.title(
            f"{os.path.basename(d['file_name'])} — "
            f"{len(outputs['instances'])} detections"
        )
        plt.show()


def show_single_prediction(predictor, image_path, dataset_name, scale=0.5):
    """Run inference on a single image and display predictions.

    Args:
        predictor: DefaultPredictor instance.
        image_path: Path to the image file.
        dataset_name: Dataset name (for metadata/class names).
        scale: Display scale.
    """
    metadata = setup_metadata_colors(dataset_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read {image_path}")
        return

    outputs = predictor(img)

    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.IMAGE,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.title(
        f"{os.path.basename(image_path)} — "
        f"{len(outputs['instances'])} detections"
    )
    plt.show()
