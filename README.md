# 🎭 Deepfake Detection Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B4-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)

A state-of-the-art deep learning system designed for detecting deepfake videos using **EfficientNet-B4** transfer learning, facial region extraction, **Grad-CAM visual explainability**, and a **Flask web dashboard** for real-time video analysis.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Dataset Preparation](#-dataset-preparation)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Running the Web Application](#-running-the-web-application)
- [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [Project Team](#-project-team)
- [License](#-license)

---

## 🔬 Overview

With the rapid advancement of generative artificial intelligence and deepfake technologies, synthetic media manipulation presents severe security, identity, and misinformation challenges. 

This project implements an end-to-end deep learning framework that analyzes uploaded videos, isolates face frames, evaluates frame-level deepfake probabilities using a fine-tuned **EfficientNet-B4** Convolutional Neural Network (CNN), and aggregates these scores into a reliable video-level authenticity verdict (**REAL** vs. **FAKE**).

---

## ✨ Key Features

- 📹 **Video-to-Frame Extraction**: Automatic frame extraction and facial cropping using OpenCV.
- 🧠 **EfficientNet-B4 Backbone**: High-capacity feature extraction leveraged via PyTorch transfer learning.
- ⚙️ **Two-Phase Phased Fine-Tuning**:
  - *Phase 1*: Classification head training with frozen feature extractor.
  - *Phase 2*: Full network fine-tuning with adaptive learning rate reduction and early stopping.
- 🔍 **Visual Explainability (Grad-CAM)**: Highlights exact spatial regions (eyes, mouth, facial boundary) where digital manipulation artifacts were detected.
- 🌐 **Interactive Flask Web Dashboard**: Drag-and-drop web UI with real-time analysis, interactive confidence bars, and frame breakdown.
- 🔒 **100% Offline & Private**: Processes all videos locally without requiring external APIs or internet services after initial model setup.

---

## 📐 System Architecture

### Data Flow & Layer Breakdown

```text
=============================================================================
                        SYSTEM DESIGN: LAYER-WISE DFD
                        (Deepfake Detection Using CNN)
=============================================================================

            [ PRESENTATION LAYER ]
            .-------------------------------------------------------.
            |  User Interface (Flask Web Dashboard)                |
            |  - Drag-and-Drop Media Upload                         |
            |  - Real-time Authenticity & Confidence Display        |
            '-------------------------------------------------------'
                       |                                 ^
          Raw Video    |                                 |  Final Prediction
         (MP4/AVI/MOV) |                                 |  (REAL / FAKE %)
                       v                                 |
            [ APPLICATION LAYER ]                        |
            .-------------------------------------------------------.
            |  Preprocessing Module                                 |
            |  - Video Frame Extraction (OpenCV)                   |
            |  - Face Detection & Alignment                         |
            |  - Normalization & Resizing (224x224)                 |
            '-------------------------------------------------------'
                       |                                 ^
         Cropped Face  |                                 |  Frame-level
            Frames     |                                 |  Scores
                       v                                 |
            [ MODEL LAYER ]                              |
            .-------------------------------------------------------.
            |  CNN Architecture                                     |
            |  - EfficientNet-B4 Feature Extractor                  |
            |  - Custom Classification Head (Dropout + Dense + Sigmoid)|
            |  - Grad-CAM Heatmap Generation Module                 |
            '-------------------------------------------------------'
                       |                                 ^
                       |                                 | Load Checkpoint
                       v                                 |
            [ DATA LAYER ]                               |
            .-------------------------------------------------------.
            |  Storage & Checkpoints                                |
            |  - Uploaded Media Cache                               |
            |  - Trained PyTorch Checkpoints (best_model.pth)       |
            '-------------------------------------------------------'
```

---

## 📁 Repository Structure

```text
MajorProject/
├── Deepfake-Detection/              # Core Application Source Code
│   ├── app.py                       # Flask Web Server entry point
│   ├── train.py                     # Training script with 2-phase learning
│   ├── evaluate.py                  # Model evaluation & ROC/Confusion matrix generator
│   ├── config.py                    # Hyperparameters & path configurations
│   ├── requirements.txt             # Python package dependencies
│   ├── README.md                    # Module documentation
│   ├── src/                         # Core PyTorch Modules
│   │   ├── model.py                 # EfficientNet-B4 architecture definition
│   │   ├── dataset.py               # Custom Dataset & augmentation pipeline
│   │   ├── data_preprocessing.py    # Frame extraction & face alignment
│   │   ├── predict.py               # Video inference pipeline
│   │   └── utils.py                 # Grad-CAM heatmap visualization utilities
│   ├── data/                        # Datasets (real, fake, processed)
│   │   └── download_dataset.py      # Utility script for sample data generation
│   ├── models/                      # Saved PyTorch checkpoint weights (.pth)
│   ├── static/                      # Web frontend assets (CSS, JS, heatmap outputs)
│   └── templates/                   # HTML Jinja2 templates (index.html, result.html)
├── generate_report.py               # Automated report generation script
├── system_design_dfd.md             # Detailed DFD design documentation
└── .gitignore                       # Git exclusion rules
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.8+ | Primary programming language |
| **Deep Learning** | PyTorch, torchvision | CNN architecture, data loaders, CUDA acceleration |
| **Model Backbone** | EfficientNet-B4 | Pretrained transfer learning feature extractor |
| **Computer Vision** | OpenCV | Video frame extraction, face detection |
| **Web Framework** | Flask | Lightweight backend web server |
| **Explainability** | Grad-CAM | Gradient-weighted Class Activation Mapping |
| **Frontend** | HTML5, CSS3, Vanilla JS | Dark-themed responsive user interface |
| **Data Processing** | NumPy, pandas, scikit-learn | Metrics calculation, confusion matrix, ROC curves |
| **Visualization** | Matplotlib, Seaborn | Training loss curves and ROC plot exports |

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.8+** installed on your system.
- *(Optional)* **NVIDIA GPU with CUDA support** for faster training and inference.

### Step 1: Clone the Repository

```bash
git clone https://github.com/joybaratix/Deepfake-Detection.git
cd Deepfake-Detection/Deepfake-Detection
```

### Step 2: Create & Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Option A: Generate Mock / Sample Dataset (Quick Test)

To quickly test the preprocessing and training scripts:

```bash
python data/download_dataset.py --generate-sample --num-real 50 --num-fake 50
```

### Option B: Using Real Benchmark Datasets

You can use benchmark deepfake datasets such as **FaceForensics++**, **Celeb-DF v2**, or **DFDC**. 

Place your videos in the respective folders:
- `data/real/` — Place authentic original videos here.
- `data/fake/` — Place deepfake/manipulated videos here.

Run the preprocessing script to extract faces:
```bash
python src/data_preprocessing.py
```
This automatically crops face frames into `data/processed/real/` and `data/processed/fake/`.

---

## 🚀 Model Training & Evaluation

### 1. Train the CNN Model

Run the two-phase training process:

```bash
python train.py
```

*Custom training arguments:*
```bash
python train.py --epochs 20 --batch-size 16 --lr 0.0001
```

The best-performing model checkpoint is automatically saved to `models/best_model.pth`.

### 2. Evaluate Model Performance

Calculate Accuracy, Precision, Recall, F1-Score, and ROC-AUC score:

```bash
python evaluate.py
```
*Evaluation outputs, confusion matrices, and ROC curves will be saved under `results/`.*

### 3. Run Command-Line Inference

Predict directly on any video file:

```bash
python src/predict.py path/to/sample_video.mp4
```

---

## 🌐 Running the Web Application

Launch the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
👉 **`http://localhost:5000`**

1. Upload any video (`.mp4`, `.avi`, `.mov`).
2. View real-time processing and frame extraction.
3. Inspect final verdict (**REAL** / **FAKE**), overall probability, confidence score, and **Grad-CAM visual heatmaps**.

---

## 🔍 Explainable AI (Grad-CAM)

Deep learning models are often considered "black boxes." This project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize where the neural network focuses when declaring a frame as synthetic.

- **Warm colors (Red/Yellow)**: High activation areas containing manipulation artifacts (e.g., blending boundaries, unnatural skin texture, eye region anomalies).
- **Cool colors (Blue/Green)**: Normal unmanipulated regions.

---

## 📄 License

This project is developed for educational and academic research purposes as part of the MCA Major Project curriculum.
