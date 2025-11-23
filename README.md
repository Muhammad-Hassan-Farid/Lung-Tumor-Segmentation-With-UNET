# Lung-Tumor-Segmentation-With-UNET

## Overview

This Jupyter Notebook performs analysis and evaluation of a **lung tumor segmentation model** on the [Lung Tumor Segmentation dataset](https://www.kaggle.com/datasets/mohammedalqatib/lung-tumor-segmentation) from Kaggle.

The notebook loads preprocessed CT scan slices (`.npy` format) along with their corresponding ground-truth tumor masks, visualizes examples, implements a U-Net-based segmentation model using PyTorch, trains it on GPU (NVIDIA Tesla T4), and evaluates performance using the **Intersection over Union (IoU)** metric.

## Dataset

- **Source**: [Lung Tumor Segmentation - Kaggle Dataset](https://www.kaggle.com/datasets/mohammedalqatib/lung-tumor-segmentation)
- **Structure**:
/train/train/
├── <patient_id>/
│   ├── data/     → Original CT slices (.npy, float64, normalized)
│   └── masks/    → Binary tumor masks (.npy, float64, 0/1 values)

- **Image size**: 256×256 pixels
- **Normalization**: Images are pre-normalized (values roughly in [-0.33, 1.0])

## Key Features

- Exploratory data visualization (original image + mask overlay)
- Custom PyTorch `Dataset` and `DataLoader` for efficient loading
- U-Net architecture with pretrained ResNet-34 encoder (via `segmentation_models_pytorch`)
- Training on Kaggle GPU (NVIDIA Tesla T4)
- Mixed precision training (AMP) for faster training and lower memory usage
- Validation with **Dice Loss** + **BCE** combined loss
- Comprehensive evaluation:
- Per-image IoU calculation
- Mean, median, best, and worst IoU reporting
- Visualization of predictions vs ground truth
- Sample output from a trained model:
  
## Requirements

- Python 3.11+
- Tensorflow
- torchvision
- numpy, matplotlib, seaborn

Install dependencies:
```bash
! pip install numpy, matplotlib, seaborn, tensorflow

**Author**
Muhammad Hassan Farid

(Feel free to modify)
