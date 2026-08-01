"""
Inference Module for Deepfake Detection.
Handles video prediction with frame-level analysis and aggregation.
All processing runs fully offline.
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import load_model, DeepfakeDetector
from src.data_preprocessing import FaceDetector, extract_frames_from_video
from src.utils import GradCAM, generate_heatmap_overlay


class VideoPredictor:
    """
    Video-level deepfake predictor.
    Extracts frames from video → detects faces → classifies each frame
    → aggregates frame predictions into a video-level decision.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or config.DEVICE
        self.model = load_model(model_path, self.device)
        self.model.eval()
        self.face_detector = FaceDetector()
        self.grad_cam = GradCAM(self.model)

        # Inference transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict_frame(self, frame):
        """
        Predict whether a single frame is real or fake.

        Args:
            frame: BGR numpy array (from OpenCV)

        Returns:
            probability: float (0 = real, 1 = fake)
            face_crop: cropped face image (RGB) or None
        """
        # Detect and crop face
        face = self.face_detector.extract_face(frame)

        if face is not None:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            # Use full frame if no face detected
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        image = Image.fromarray(face_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()

        return probability, face_rgb

    def predict_video(self, video_path, num_frames=None, return_details=False):
        """
        Predict whether a video is real or fake.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze (None = config default)
            return_details: If True, return per-frame details

        Returns:
            dict with:
                - prediction: 'FAKE' or 'REAL'
                - confidence: float (0-100%)
                - probability: float (0-1, probability of being fake)
                - frame_predictions: list of per-frame probabilities
                - num_frames_analyzed: int
                - heatmap: Grad-CAM heatmap of the most suspicious frame
                - suspicious_frame: the frame with highest fake probability
        """
        if num_frames is None:
            num_frames = config.FRAMES_PER_VIDEO

        # Extract frames
        try:
            frames = extract_frames_from_video(video_path, num_frames)
        except Exception as e:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "probability": 0.5,
                "error": str(e),
                "frame_predictions": [],
                "num_frames_analyzed": 0,
            }

        if len(frames) == 0:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "probability": 0.5,
                "error": "No frames could be extracted from video",
                "frame_predictions": [],
                "num_frames_analyzed": 0,
            }

        # Predict each frame
        frame_probs = []
        face_crops = []
        for frame in frames:
            prob, face_crop = self.predict_frame(frame)
            frame_probs.append(prob)
            face_crops.append(face_crop)

        # Aggregate predictions
        probs_array = np.array(frame_probs)

        if config.VIDEO_PREDICTION_METHOD == "median":
            final_prob = float(np.median(probs_array))
        elif config.VIDEO_PREDICTION_METHOD == "weighted":
            # Weight extreme predictions more heavily
            weights = np.abs(probs_array - 0.5) + 0.5
            final_prob = float(np.average(probs_array, weights=weights))
        else:  # mean
            final_prob = float(np.mean(probs_array))

        # Determine prediction
        is_fake = final_prob >= config.CONFIDENCE_THRESHOLD
        prediction = "FAKE" if is_fake else "REAL"
        confidence = final_prob * 100 if is_fake else (1 - final_prob) * 100

        # Generate Grad-CAM for most suspicious frame
        most_suspicious_idx = np.argmax(probs_array)
        suspicious_face = face_crops[most_suspicious_idx]
        heatmap = self._generate_gradcam(suspicious_face)
        heatmap_overlay = generate_heatmap_overlay(suspicious_face, heatmap)

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probability": round(final_prob, 4),
            "frame_predictions": [round(p, 4) for p in frame_probs],
            "num_frames_analyzed": len(frames),
            "heatmap": heatmap,
            "heatmap_overlay": heatmap_overlay,
            "suspicious_frame": suspicious_face,
            "most_suspicious_frame_idx": int(most_suspicious_idx),
        }

        return result

    def _generate_gradcam(self, face_image_rgb):
        """Generate Grad-CAM heatmap for a face image."""
        try:
            image = Image.fromarray(face_image_rgb)
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            heatmap = self.grad_cam.generate(tensor)
            return heatmap
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            return np.zeros((config.IMG_SIZE, config.IMG_SIZE))

    def save_result_images(self, result, output_dir):
        """Save result images (suspicious frame, heatmap) to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save suspicious frame
        if result.get("suspicious_frame") is not None:
            frame_path = os.path.join(output_dir, "suspicious_frame.jpg")
            frame_bgr = cv2.cvtColor(result["suspicious_frame"], cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (config.IMG_SIZE, config.IMG_SIZE))
            cv2.imwrite(frame_path, frame_resized)

        # Save heatmap overlay
        if result.get("heatmap_overlay") is not None:
            heatmap_path = os.path.join(output_dir, "heatmap.jpg")
            overlay_bgr = cv2.cvtColor(result["heatmap_overlay"], cv2.COLOR_RGB2BGR)
            overlay_resized = cv2.resize(overlay_bgr, (config.IMG_SIZE, config.IMG_SIZE))
            cv2.imwrite(heatmap_path, overlay_resized)

        return output_dir


if __name__ == "__main__":
    """Test video prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict deepfake video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", default=None, help="Path to model weights")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames")
    args = parser.parse_args()

    predictor = VideoPredictor(model_path=args.model)
    result = predictor.predict_video(args.video, num_frames=args.frames)

    print(f"\n{'='*50}")
    print(f"Video: {args.video}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Fake Probability: {result['probability']:.4f}")
    print(f"Frames Analyzed: {result['num_frames_analyzed']}")
    print(f"{'='*50}")
