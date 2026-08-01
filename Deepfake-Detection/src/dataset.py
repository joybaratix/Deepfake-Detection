"""
PyTorch Dataset and DataLoader for Deepfake Detection.
Handles loading preprocessed face frames with train/val/test splitting.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.
    Loads preprocessed face images from real/ and fake/ directories.
    Label: 0 = Real, 1 = Fake
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label


# ─── Data Augmentation Transforms ────────────────────────
def get_train_transforms():
    """Training transforms with augmentation for robustness."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomGrayscale(p=0.05),
        # Simulate JPEG compression artifacts (common in social media)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Random erasing simulates occlusion
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─── Data Loading Utilities ─────────────────────────────
def load_dataset_paths(data_dir=None):
    """
    Load image paths and labels from the processed dataset directory.

    Expected structure:
        data_dir/
            real/   (preprocessed face images from real videos)
            fake/   (preprocessed face images from fake videos)

    Returns:
        image_paths: list of file paths
        labels: list of labels (0=real, 1=fake)
    """
    if data_dir is None:
        data_dir = config.PROCESSED_DIR

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    image_paths = []
    labels = []

    # Load real images (label = 0)
    if os.path.exists(real_dir):
        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(real_dir, fname))
                labels.append(0)

    # Load fake images (label = 1)
    if os.path.exists(fake_dir):
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(fake_dir, fname))
                labels.append(1)

    print(f"Dataset loaded: {len(image_paths)} images "
          f"({labels.count(0)} real, {labels.count(1)} fake)")

    return image_paths, labels


def create_data_splits(image_paths, labels):
    """
    Create stratified train/val/test splits.

    Returns:
        Dictionary with 'train', 'val', 'test' keys,
        each containing (paths, labels) tuple.
    """
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=labels,
        random_state=42
    )

    # Second split: val vs test
    relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=42
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    print(f"Data splits - Train: {len(train_paths)}, "
          f"Val: {len(val_paths)}, Test: {len(test_paths)}")

    return splits


def create_data_loaders(data_dir=None, batch_size=None):
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    image_paths, labels = load_dataset_paths(data_dir)

    if len(image_paths) == 0:
        raise ValueError(
            "No images found. Run data preprocessing first:\n"
            "  python src/data_preprocessing.py\n"
            "Or generate sample data:\n"
            "  python data/download_dataset.py --generate-sample"
        )

    splits = create_data_splits(image_paths, labels)

    loaders = {}

    # Train loader with augmentation
    train_dataset = DeepfakeDataset(
        *splits["train"], transform=get_train_transforms()
    )
    loaders["train"] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # Val loader (no augmentation)
    val_dataset = DeepfakeDataset(
        *splits["val"], transform=get_val_transforms()
    )
    loaders["val"] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Test loader (no augmentation)
    test_dataset = DeepfakeDataset(
        *splits["test"], transform=get_val_transforms()
    )
    loaders["test"] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return loaders


if __name__ == "__main__":
    """Test data loading."""
    try:
        loaders = create_data_loaders()
        for split_name, loader in loaders.items():
            batch = next(iter(loader))
            images, labels = batch
            print(f"{split_name}: batch shape={images.shape}, labels={labels[:5]}")
    except ValueError as e:
        print(f"Error: {e}")
