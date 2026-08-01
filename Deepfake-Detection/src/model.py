"""
CNN Model Architecture for Deepfake Detection.
EfficientNet-B4 with transfer learning and custom classification head.
Runs fully offline using pre-trained ImageNet weights cached locally.
"""

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DeepfakeDetector(nn.Module):
    """
    Deepfake Detection Model using EfficientNet-B4 backbone.

    Architecture:
        EfficientNet-B4 (pretrained on ImageNet, frozen initially)
        → Adaptive Average Pooling
        → Dropout(0.3)
        → FC(1792 → 512) → BatchNorm → ReLU
        → Dropout(0.2)
        → FC(512 → 1) → Sigmoid

    Output: Single probability value (0 = Real, 1 = Fake)
    """

    def __init__(self, pretrained=True, dropout_rate=None):
        super(DeepfakeDetector, self).__init__()

        if dropout_rate is None:
            dropout_rate = config.DROPOUT_RATE

        # Load EfficientNet-B4 backbone
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

        if pretrained:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b4(weights=weights)
        else:
            self.backbone = efficientnet_b4(weights=None)

        # Get the feature dimension from the backbone
        feature_dim = self.backbone.classifier[1].in_features  # 1792 for B4

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Custom classification head (uses LayerNorm for batch_size=1 compatibility)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(512, 1),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier layers with Xavier uniform."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
        Returns:
            Tensor of shape (batch, 1) — raw logits (use sigmoid for probability)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning Phase 1."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — only classifier head will be trained.")

    def unfreeze_backbone(self, num_layers_to_unfreeze=None):
        """
        Unfreeze backbone for fine-tuning (Phase 2).
        If num_layers_to_unfreeze is None, unfreeze all layers.
        """
        if num_layers_to_unfreeze is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Full backbone unfrozen for fine-tuning.")
        else:
            # Unfreeze last N layers
            layers = list(self.backbone.features.children())
            for layer in layers[-num_layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Last {num_layers_to_unfreeze} backbone layers unfrozen.")

    def get_feature_extractor(self):
        """Return the backbone as a feature extractor (for Grad-CAM)."""
        return self.backbone.features


def load_model(model_path=None, device=None):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model weights (.pth file)
        device: torch.device to load model on

    Returns:
        DeepfakeDetector model in eval mode
    """
    if device is None:
        device = config.DEVICE

    model = DeepfakeDetector(pretrained=False)

    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, "best_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from: {model_path}")
            if "epoch" in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if "val_accuracy" in checkpoint:
                print(f"  Val Accuracy: {checkpoint['val_accuracy']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model weights loaded from: {model_path}")
    else:
        print(f"No saved model found at: {model_path}")
        print("Using fresh model with pre-trained ImageNet weights.")
        model = DeepfakeDetector(pretrained=True)

    model.to(device)
    model.eval()
    return model


def get_model_summary(model):
    """Print model summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*50}")
    print(f"Model: EfficientNet-B4 Deepfake Detector")
    print(f"{'='*50}")
    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")
    print(f"Frozen parameters:    {frozen_params:>12,}")
    print(f"{'='*50}")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


if __name__ == "__main__":
    """Test model creation."""
    model = DeepfakeDetector(pretrained=True)
    model.to(config.DEVICE)
    get_model_summary(model)

    # Test forward pass
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    output = model(dummy_input)
    probs = torch.sigmoid(output)
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Predictions:  {probs.squeeze().detach().cpu().numpy()}")
