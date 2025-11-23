# Lung-Tumor-Segmentation-With-UNET

## Overview

This Jupyter Notebook performs analysis and evaluation of a **lung tumor segmentation model** on the [Lung Tumor Segmentation dataset](https://www.kaggle.com/datasets/mohammedalqatib/lung-tumor-segmentation) from Kaggle.

The notebook loads preprocessed CT scan slices (`.npy` format) along with their corresponding ground-truth tumor masks, visualizes examples, implements a U-Net-based segmentation model using tensorflow, trains it on GPU (NVIDIA Tesla T4), and evaluates performance using the **Intersection over Union (IoU)** metric.

## Dataset
- **Source**: [Lung Tumor Segmentation - Kaggle Dataset](https://www.kaggle.com/datasets/mohammedalqatib/lung-tumor-segmentation)
- **Structure**:
#### /train/train/
#### ├── <patient_id>/
#### ├── data/     → Original CT slices (.npy, float64, normalized)
#### └── masks/    → Binary tumor masks (.npy, float64, 0/1 values)

- **Image size**: 256×256 pixels
- **Normalization**: Images are pre-normalized (values roughly in [-0.33, 1.0])

## Key Features
- Exploratory data visualization (original image + mask overlay)
- Load the dataset
- U-Net architecture 
- Training on Kaggle GPU 
- Validation with **Dice Loss** + **BCE** combined loss
- Comprehensive evaluation:
- Per-image IoU calculation
- Mean, median, best, and worst IoU reporting
- Visualization of predictions vs ground truth

## Results
![Alt text for the image](https://github.com/Muhammad-Hassan-Farid/Lung-Tumor-Segmentation-With-UNET/blob/main/Output_Overlay.png?raw=true)

## Requirements
- Python 3.11+
- Tensorflow
- numpy
- matplotlib
- seaborn

Install dependencies:
```bash
! pip install numpy, matplotlib, seaborn, tensorflow
```
**Author**
Muhammad Hassan Farid

(Feel free to modify)
