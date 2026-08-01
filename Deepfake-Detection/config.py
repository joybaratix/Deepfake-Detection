"""
Deepfake Detection Using CNN - Configuration
Central configuration file for all project settings.
"""

import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Dataset ─────────────────────────────────────────────
DATASET_REAL_DIR = os.path.join(DATA_DIR, "real")
DATASET_FAKE_DIR = os.path.join(DATA_DIR, "fake")
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ─── Preprocessing ───────────────────────────────────────
IMG_SIZE = 224                  # Input image size for EfficientNet
FRAMES_PER_VIDEO = 5            # Number of frames to extract per video (reduced for speed)
FACE_DETECTION_CONFIDENCE = 0.5
SEQUENCE_LENGTH = 30            # Frames per sequence for temporal analysis

# ─── Model ───────────────────────────────────────────────
MODEL_NAME = "efficientnet-b4"
NUM_CLASSES = 1                 # Binary classification (sigmoid output)
DROPOUT_RATE = 0.3
FEATURE_DIM = 1792              # EfficientNet-B4 feature dimension

# ─── Training ────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7

# ─── Inference ───────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5     # Threshold for fake/real classification
VIDEO_PREDICTION_METHOD = "mean"  # mean, median, or weighted

# ─── Flask App ───────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
MAX_CONTENT_LENGTH = 500 * 1024 * 1024   # 500 MB max upload
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

# ─── Device ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Create directories ─────────────────────────────────
for d in [DATA_DIR, SAMPLE_DATA_DIR, PROCESSED_DIR, MODELS_DIR, UPLOADS_DIR, RESULTS_DIR,
          DATASET_REAL_DIR, DATASET_FAKE_DIR]:
    os.makedirs(d, exist_ok=True)
