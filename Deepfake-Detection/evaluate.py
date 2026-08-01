"""
Evaluation Script for Deepfake Detection Model.
Computes accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.model import load_model
from src.dataset import create_data_loaders
from src.utils import plot_confusion_matrix, plot_roc_curve


@torch.no_grad()
def evaluate_model(model_path=None, data_dir=None):
    """
    Comprehensive model evaluation on the test set.

    Returns:
        Dictionary containing all metrics and predictions.
    """
    print("=" * 60)
    print("  Deepfake Detection — Model Evaluation")
    print("=" * 60)

    # Load model
    model = load_model(model_path)
    model.eval()

    # Load test data
    loaders = create_data_loaders(data_dir)
    test_loader = loaders["test"]

    print(f"\nEvaluating on {len(test_loader.dataset)} test samples...")
    print(f"Device: {config.DEVICE}")
    print()

    # Collect predictions
    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(config.DEVICE)

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Handle single sample batch
        if isinstance(probs, np.float32) or probs.ndim == 0:
            probs = np.array([probs])

        preds = (probs >= config.CONFIDENCE_THRESHOLD).astype(int)

        all_labels.extend(labels.numpy().astype(int))
        all_probs.extend(probs)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # ─── Compute Metrics ─────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        fpr, tpr, thresholds = [0], [0], [0]

    # ─── Print Results ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"{'='*60}")

    print(f"\nConfusion Matrix:")
    print(f"  {'':>10} Pred Real  Pred Fake")
    print(f"  {'Real':>10}  {cm[0][0]:>8d}  {cm[0][1]:>8d}")
    print(f"  {'Fake':>10}  {cm[1][0]:>8d}  {cm[1][1]:>8d}")

    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Real", "Fake"],
        digits=4,
        zero_division=0
    ))

    # ─── Save Results ────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(all_labels),
        "real_samples": int(np.sum(all_labels == 0)),
        "fake_samples": int(np.sum(all_labels == 1)),
    }

    metrics_path = os.path.join(config.RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save confusion matrix plot
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, save_path=cm_path)

    # Save ROC curve plot
    roc_path = os.path.join(config.RESULTS_DIR, "roc_curve.png")
    plot_roc_curve(fpr, tpr, auc, save_path=roc_path)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    parser.add_argument("--model", default=None, help="Path to model weights")
    parser.add_argument("--data-dir", default=None, help="Path to processed data")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, data_dir=args.data_dir)
