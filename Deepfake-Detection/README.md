# Deepfake Detection Using Convolutional Neural Networks

A deep learning system for detecting deepfake videos using **EfficientNet-B4** with transfer learning. The system extracts frames from videos, detects faces, classifies each frame as real or fake, and aggregates results into a video-level prediction. Includes a **Flask web interface** for easy video upload and analysis.

> **Runs fully offline** — no external APIs or cloud services required.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Usage](#usage)
6. [Model Architecture](#model-architecture)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Web Interface](#web-interface)
10. [Technologies Used](#technologies-used)

---

## Features

- **Video-only deepfake detection** — frame extraction → face detection → CNN classification → aggregation
- **EfficientNet-B4** with transfer learning from ImageNet for high accuracy
- **Grad-CAM visualization** — highlights manipulated facial regions
- **Two-phase training** — frozen backbone (Phase 1) → full fine-tuning (Phase 2)
- **Comprehensive evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Minimalist Flask web interface** — drag-and-drop upload, real-time results
- **Fully offline operation** — no external APIs or internet required after setup
- **Data augmentation** — rotation, flip, brightness, JPEG compression simulation

---

## Project Structure

```
Deepfake-Detection/
├── app.py                     # Flask web application
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation script
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── model.py               # EfficientNet-B4 CNN architecture
│   ├── dataset.py             # PyTorch dataset & data loaders
│   ├── data_preprocessing.py  # Frame extraction & face detection
│   ├── predict.py             # Video inference module
│   └── utils.py               # Grad-CAM, plotting utilities
├── data/
│   ├── real/                  # Place real videos here
│   ├── fake/                  # Place fake/deepfake videos here
│   ├── processed/             # Preprocessed face frames (auto-generated)
│   └── download_dataset.py    # Dataset setup utility
├── templates/
│   ├── index.html             # Upload page
│   └── result.html            # Results page
├── static/
│   ├── css/style.css          # Dark theme styles
│   └── js/main.js             # Client-side interactions
├── models/                    # Saved model checkpoints
├── uploads/                   # Temporary uploaded files
└── results/                   # Evaluation results & plots
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

### Setup

```bash
# Clone or navigate to the project directory
cd Deepfake-Detection

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

### Option 1: Use Sample Dataset (for testing)

```bash
python data/download_dataset.py --generate-sample --num-real 50 --num-fake 50
```

### Option 2: Use Real Datasets (for training)

Download one or more benchmark datasets:

| Dataset | Videos | Access |
|---------|--------|--------|
| **FaceForensics++** | 1,000 real + 5,000 fake | [GitHub](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF v2** | 590 real + 5,639 fake | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** | 100,000+ videos | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |

After downloading, place videos in:
```
data/real/    ← authentic videos
data/fake/    ← deepfake videos
```

### Preprocess Dataset

Extract face frames from all videos:

```bash
python src/data_preprocessing.py
```

This creates `data/processed/real/` and `data/processed/fake/` with cropped face images.

---

## Usage

### 1. Train the Model

```bash
python train.py
```

Optional arguments:
```bash
python train.py --epochs 30 --batch-size 16 --lr 0.0001
python train.py --resume models/best_model.pth  # Resume training
```

### 2. Evaluate the Model

```bash
python evaluate.py
```

Generates metrics, confusion matrix, and ROC curve in `results/`.

### 3. Predict on a Video

```bash
python src/predict.py path/to/video.mp4
```

### 4. Launch Web Interface

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Model Architecture

```
EfficientNet-B4 (pretrained on ImageNet)
    │
    ├── Feature Extraction (frozen in Phase 1)
    │   └── Convolutional layers (1792-dim features)
    │
    └── Custom Classification Head
        ├── Dropout(0.3)
        ├── Linear(1792 → 512)
        ├── BatchNorm1d(512)
        ├── ReLU
        ├── Dropout(0.2)
        └── Linear(512 → 1) → Sigmoid
```

- **Input**: 224 × 224 × 3 face images
- **Output**: Single probability (0 = Real, 1 = Fake)
- **Parameters**: ~17.5M total, with phased unfreezing for transfer learning

---

## Training Pipeline

### Phase 1: Classification Head Training (5 epochs)
- Backbone weights **frozen** (ImageNet features)
- Only classification head is trained
- Higher learning rate (10× base LR)

### Phase 2: Full Fine-tuning (remaining epochs)
- Entire model is **unfrozen**
- Lower learning rate with ReduceLROnPlateau scheduler
- Early stopping (patience = 7 epochs)
- Gradient clipping (max_norm = 1.0)
- Best model saved based on validation accuracy

### Data Augmentation
- Random horizontal flip, rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Gaussian blur (simulates compression)
- Random erasing (simulates occlusion)
- ImageNet normalization

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True fakes / predicted fakes |
| **Recall** | True fakes / actual fakes |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |
| **Confusion Matrix** | Error pattern visualization |

All metrics and plots are saved to `results/`.

---

## Web Interface

Minimalist dark theme interface:

1. **Upload Page** — Drag-and-drop or click to upload a video
2. **Results Page** — Shows:
   - Verdict (REAL / FAKE) with confidence percentage
   - Frame-by-frame analysis bar chart
   - Video details (resolution, duration, frames analyzed)
   - Most suspicious frame image
   - Grad-CAM heatmap highlighting manipulated regions

---

## Technologies Used

| Category | Technology |
|----------|------------|
| **Deep Learning** | PyTorch, torchvision, EfficientNet-B4 |
| **Computer Vision** | OpenCV (face detection, frame extraction) |
| **Data Processing** | NumPy, pandas, scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Explainability** | Grad-CAM (pytorch-grad-cam) |
| **Language** | Python 3.8+ |

---

## Project Members

- **JOYGANESH BARAT** [120710242004]
- **SAYAN BARAT** [120710242009]

### Project Guide
**Prof. ANUPAM BAIDYA**  
Department of Computer Application, BCREC

---

## License

This project is developed as part of the MCA Major Project at BCREC.
