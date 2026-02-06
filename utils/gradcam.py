"""
Grad-CAM and Grad-CAM++ for Detectron2 models.

Generates class activation maps to visualize which regions of the image
contribute most to a specific object detection.

Usage:
    cam_extractor = Detectron2GradCAM(config_file, model_file)
    image_dict, cam_orig = cam_extractor.get_cam(
        img="path/to/image.jpg",
        target_instance=0,
        layer_name="backbone.bottom_up.res5.2.conv3",
        grad_cam_type="GradCAM",
    )
"""

import cv2
import numpy as np
import torch

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model


# =============================================================================
# Core GradCAM
# =============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for Detectron2 models.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients for CAM computation.

    Args:
        model: Detectron2 GeneralizedRCNN model.
        target_layer_name: Name of the convolutional layer (e.g.
            'backbone.bottom_up.res5.2.conv3').
    """

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self._handles = []
        self._register_hooks()

    def _get_activations_hook(self, module, input, output):
        self.activations = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self._handles.append(
                    module.register_forward_hook(self._get_activations_hook)
                )
                self._handles.append(
                    module.register_backward_hook(self._get_grads_hook)
                )
                return
        raise ValueError(
            f"Layer '{self.target_layer_name}' not found in model. "
            f"Available layers: {[n for n, _ in self.model.named_modules()]}"
        )

    def _release_hooks(self):
        for handle in self._handles:
            handle.remove()

    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H, W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        if cam_orig.max() > 0:
            cam_orig = cam_orig / cam_orig.max()
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_hooks()

    def __call__(self, inputs, target_category):
        """Compute GradCAM for a specific detection instance.

        Args:
            inputs: Dict in Detectron2 model input format.
            target_category: Index of the target instance. If None,
                uses the highest-scoring instance.

        Returns:
            cam: Activation map resized to original image dimensions.
            cam_orig: Raw (un-resized) activation map.
            output: Model output (list of Instance dicts).
        """
        self.model.zero_grad()
        output = self.model.forward([inputs])

        if target_category is None:
            target_category = np.argmax(
                output[0]["instances"].scores.cpu().data.numpy(), axis=-1
            )

        score = output[0]["instances"].scores[target_category]
        score.backward()

        gradient = self.gradient[0].cpu().data.numpy()      # [C, H, W]
        activations = self.activations[0].cpu().data.numpy()  # [C, H, W]
        weight = np.mean(gradient, axis=(1, 2))               # [C]

        cam = activations * weight[:, np.newaxis, np.newaxis]
        cam, cam_orig = self._postprocess_cam(cam, inputs["width"], inputs["height"])

        return cam, cam_orig, output


class GradCamPlusPlus(GradCAM):
    """Grad-CAM++ variant, better for multiple instances of the same class.

    Reference: https://arxiv.org/abs/1710.11063
    """

    def __call__(self, inputs, target_category):
        self.model.zero_grad()
        output = self.model.forward([inputs])

        if target_category is None:
            target_category = np.argmax(
                output[0]["instances"].scores.cpu().data.numpy(), axis=-1
            )

        score = output[0]["instances"].scores[target_category]
        score.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        grads_power_2 = gradient ** 2
        grads_power_3 = grads_power_2 * gradient
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-6

        aij = grads_power_2 / (
            2 * grads_power_2
            + sum_activations[:, None, None] * grads_power_3
            + eps
        )
        aij = np.where(gradient != 0, aij, 0)

        weights = np.maximum(gradient, 0) * aij
        weight = np.sum(weights, axis=(1, 2))

        cam = activations * weight[:, np.newaxis, np.newaxis]
        cam, cam_orig = self._postprocess_cam(cam, inputs["width"], inputs["height"])

        return cam, cam_orig, output


# =============================================================================
# High-level wrapper
# =============================================================================

class Detectron2GradCAM:
    """High-level GradCAM wrapper for Detectron2.

    Handles config setup, model loading, image preprocessing, and
    CAM extraction in one convenient interface.

    Args:
        config_file: Path to Detectron2 model config file.
        model_file: Path to trained model weights (.pth).
    """

    def __init__(self, config_file, model_file):
        self.cfg = self._setup_cfg(config_file, model_file)

    def _setup_cfg(self, config_file, model_file):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        return cfg

    def _get_input_dict(self, original_image):
        height, width = original_image.shape[:2]
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(
            image.astype("float32").transpose(2, 0, 1)
        ).requires_grad_(True)
        return {"image": image, "height": height, "width": width}

    def get_cam(self, img, target_instance, layer_name,
                grad_cam_type="GradCAM"):
        """Generate GradCAM for a specific instance in an image.

        Args:
            img: Path to the input image.
            target_instance: Index of the detection instance to explain.
            layer_name: Target convolutional layer name.
            grad_cam_type: 'GradCAM' or 'GradCAM++'.

        Returns:
            image_dict: Dict with keys 'image', 'cam', 'output', 'label'.
            cam_orig: Raw unprocessed CAM.
        """
        model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        image = read_image(img, format="BGR")
        input_image_dict = self._get_input_dict(image)

        cam_class = GradCAM if grad_cam_type == "GradCAM" else GradCamPlusPlus

        with cam_class(model, layer_name) as cam:
            cam_map, cam_orig, output = cam(input_image_dict, target_instance)

        label = MetadataCatalog.get(
            self.cfg.DATASETS.TRAIN[0]
        ).thing_classes[
            output[0]["instances"].pred_classes[target_instance]
        ]

        return {
            "image": image,
            "cam": cam_map,
            "output": output,
            "label": label,
        }, cam_orig
