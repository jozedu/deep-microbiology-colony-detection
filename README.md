# Deep Learning for Microbial Colony Detection on Agar Plates

Master's Thesis — Object Detection using Detectron2 (Facebook AI Research)

## Overview

This project applies deep learning–based object detection to automate the
detection and counting of microbial colonies on agar plate images. Six model
architectures are trained and evaluated across two datasets, with Weighted
Boxes Fusion (WBF) ensembling and Grad-CAM explainability.

## Project Structure

```
├── config.py                         # Centralized configuration (paths, models, hyperparameters)
├── utils/
│   ├── __init__.py
│   ├── training.py                   # LossEvalHook & MyTrainer (custom Detectron2 trainer)
│   ├── evaluation.py                 # COCO mAP evaluation & colony counting metrics
│   ├── visualization.py              # Dataset & prediction visualization helpers
│   ├── ensemble.py                   # Weighted Boxes Fusion (WBF) & grid search
│   └── gradcam.py                    # Grad-CAM / Grad-CAM++ for Detectron2
├── notebooks/
│   ├── 1_setup.ipynb                 # Environment setup & installation
│   ├── 2_data_exploration.ipynb      # Dataset registration, statistics & visualization
│   ├── 3_train.ipynb                 # Model training (parameterized for any dataset/model)
│   ├── 4_evaluate.ipynb              # COCO mAP & colony counting evaluation
│   ├── 5_ensemble.ipynb              # WBF ensemble methods & grid search
│   ├── 6_gradcam.ipynb               # Grad-CAM visualization
│   └── 7_inference.ipynb             # Inference on new images
├── detectron2.ipynb                  # Original monolithic notebook (archived)
└── README.md
```

## Models

| Key                | Architecture       | Backbone    | Config                                                           |
| ------------------ | ------------------ | ----------- | ---------------------------------------------------------------- |
| `faster_rcnn_R50`  | Faster R-CNN       | ResNet-50   | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`                   |
| `faster_rcnn_R101` | Faster R-CNN       | ResNet-101  | `COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml`                  |
| `retinanet_R50`    | RetinaNet          | ResNet-50   | `COCO-Detection/retinanet_R_50_FPN_3x.yaml`                     |
| `retinanet_R101`   | RetinaNet          | ResNet-101  | `COCO-Detection/retinanet_R_101_FPN_3x.yaml`                    |
| `mask_rcnn_R50`    | Mask R-CNN         | ResNet-50   | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`          |
| `mask_rcnn_R101`   | Mask R-CNN         | ResNet-101  | `COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml`         |

## Datasets

### Part 1 — AGAR Dataset (Public)

- **Images:** 9,851 | **Annotations:** 182,864
- **Classes:** *S. aureus*, *P. aeruginosa*, *E. coli*
- **Subsets:** `total`, `bright`, `dark`, `vague`, `lowres` (by background type)
- **Format:** COCO JSON

### Part 2 — Curated Dataset (Roboflow)

- **Images:** 165 | **Annotations:** 1,801
- **Classes:** *S. aureus*, *P. aeruginosa*, *E. coli* + culture media types
- **Media types:** Blood agar, Chocolate agar, MacConkey, Mannitol salt
- **Format:** COCO JSON

## Experimental Phases

Each dataset part follows three phases:

1. **Base Training** — Train all 6 (or 4) architectures from COCO-pretrained weights.
2. **Transfer Learning** — Fine-tune from best AGAR weights on the curated dataset.
3. **WBF Ensemble** — Combine predictions from multiple models using Weighted Boxes Fusion.

## Reproducibility

### Quick Start (Google Colab)

1. Open `notebooks/1_setup.ipynb` and run all cells to install Detectron2.
2. Upload `config.py` and `utils/` to the Colab working directory.
3. Open `notebooks/2_data_exploration.ipynb` and set your dataset paths.
4. Follow the numbered notebooks in order.

### Configuration

All paths and model definitions are centralized in `config.py`. Update the
base paths at the top of that file to match your Google Drive layout:

```python
AGAR_IMG_DIR = "/content/drive/MyDrive/TESE/AGAR/images"
AGAR_ANN_DIR = "/content/drive/MyDrive/TESE/AGAR/annotations"
OUTPUTS_DIR  = "/content/drive/MyDrive/TESE/RESULTS"
```

### Training Hyperparameters

| Parameter       | AGAR         | Curated      |
| --------------- | ------------ | ------------ |
| Optimizer       | SGD          | SGD          |
| Learning rate   | 0.005        | 0.005        |
| Momentum        | 0.9          | 0.9          |
| Weight decay    | 0.0005       | 0.0005       |
| Batch size      | 8            | 8            |
| Epochs          | 10           | 100          |
| LR decay step   | 70% of iters | 70% of iters |

### Ensemble Parameters (Best)

| Parameter       | Value   |
| --------------- | ------- |
| Method          | WBF     |
| IoU threshold   | 0.75    |
| Skip box thr    | 0.01    |
| Model weights   | [5,5,7,7,5,5] |

## Key Results

### AGAR Dataset (Total subset)

| Model            | AP     | AP50   |
| ---------------- | ------ | ------ |
| Faster R-CNN R101| 62.0%  | —      |
| WBF Ensemble     | **66.4%** | —   |

### Curated Dataset

| Model            | AP     | AP50   |
| ---------------- | ------ | ------ |
| RetinaNet R50    | 52.4%  | —      |
| WBF Ensemble     | **56.3%** | —   |

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.10
- Detectron2 (built from source)
- torchvision
- ensemble_boxes
- pycocotools
- openpyxl
- matplotlib
- OpenCV

## Platform

All experiments were conducted on **Google Colab Pro** with NVIDIA Tesla T4 GPU.
