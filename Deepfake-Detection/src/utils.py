"""
Utility functions for Deepfake Detection.
Grad-CAM visualization, plotting, and helper functions.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Grad-CAM Implementation ────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights facial regions the model focuses on for its prediction.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default: last convolutional layer of EfficientNet backbone
        if target_layer is None:
            self.target_layer = model.backbone.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Target class index (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) normalized to [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target = output
        else:
            target = output[:, target_class]

        target.backward(retain_graph=True)

        # Pool gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=(config.IMG_SIZE, config.IMG_SIZE),
            mode="bilinear", align_corners=False
        )

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def generate_heatmap_overlay(original_image, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_image: Original image (numpy array, RGB, uint8)
        heatmap: Grad-CAM heatmap (numpy array, float [0,1])
        alpha: Overlay transparency

    Returns:
        overlay: Blended image (numpy array, RGB, uint8)
    """
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ─── Image Preprocessing for Inference ──────────────────
def prepare_image_for_model(image_path_or_array):
    """
    Prepare an image for model inference.

    Args:
        image_path_or_array: File path (str) or numpy array (RGB)

    Returns:
        tensor: Preprocessed tensor (1, 3, 224, 224)
        original: Original image as numpy array (RGB, uint8)
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array).convert("RGB")
    elif isinstance(image_path_or_array, np.ndarray):
        image = Image.fromarray(image_path_or_array)
    else:
        image = image_path_or_array

    original = np.array(image.resize((config.IMG_SIZE, config.IMG_SIZE)))
    tensor = transform(image).unsqueeze(0)

    return tensor, original


# ─── Plotting Utilities ─────────────────────────────────
def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", color="#4fc3f7", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#f44336", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Accuracy", color="#4fc3f7", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", color="#f44336", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2x2 numpy array)
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Real", "Fake"]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under ROC curve
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="#4fc3f7", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#666", linestyle="--", linewidth=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to: {save_path}")
    plt.close()


# ─── File Utilities ──────────────────────────────────────
def is_video_file(filename):
    """Check if a file is a supported video format."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in config.ALLOWED_EXTENSIONS


def get_video_info(video_path):
    """Get basic video information."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info
