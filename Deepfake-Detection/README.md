# 🎭 Deepfake Detection Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B4-green?style=for-the-badge)

A state-of-the-art deep learning system designed for detecting deepfake videos using **EfficientNet-B4** transfer learning, facial region extraction, **Grad-CAM visual explainability**, and a **Flask web dashboard** for real-time video analysis.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Dataset Preparation](#-dataset-preparation)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Running the Web Application](#-running-the-web-application)
- [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [Project Team](#-project-team)

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

### Step 1: Navigate & Setup Virtual Environment

```bash
# Navigate to Deepfake-Detection folder
cd Deepfake-Detection

# Activate Virtual Environment (PowerShell)
.\venv\Scripts\Activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Generate Mock / Sample Dataset (Quick Test)

```bash
python data/download_dataset.py --generate-sample --num-real 50 --num-fake 50
```

### Preprocess Benchmark Datasets (FaceForensics++, Celeb-DF, DFDC)

Place videos in `data/real/` and `data/fake/`, then run:
```bash
python src/data_preprocessing.py
```

---

## 🚀 Model Training & Evaluation

### 1. Train the CNN Model

```bash
python train.py
```

### 2. Evaluate Model Performance

```bash
python evaluate.py
```

### 3. CLI Prediction

```bash
python src/predict.py path/to/sample_video.mp4
```

---

## 🌐 Running the Web Application

Launch the Flask server:

```bash
python app.py
```

Navigate to **`http://localhost:5000`** in your browser.

---

## 👥 Project Team

### Group 20 — MCA Major Project
*Department of Computer Application, Dr. B. C. Roy Engineering College (BCREC)*

- **Joyganesh Barat** (Roll: `120710242004`)
- **Sayan Barat** (Roll: `120710242009`)

**Project Guide:**  
**Prof. Anupam Baidya**  
*Department of Computer Application, BCREC*
