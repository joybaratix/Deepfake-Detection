"""
Data Preprocessing Module
Handles video frame extraction, face detection, cropping, and augmentation.
All processing is done offline using OpenCV — no external API calls needed.
"""

import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Face Detection ──────────────────────────────────────
class FaceDetector:
    """Face detector using OpenCV DNN (works offline, no extra downloads)."""

    def __init__(self):
        # Use OpenCV's built-in Haar cascade for face detection (fully offline)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame, min_face_size=60):
        """
        Detect faces in a frame and return bounding boxes.
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        return faces

    def extract_face(self, frame, margin=0.3):
        """
        Extract the largest face from a frame with a margin.
        Returns cropped face image or None if no face found.
        """
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return None

        # Get the largest face
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        x, y, w, h = faces[largest_idx]

        # Add margin
        h_frame, w_frame = frame.shape[:2]
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_frame, x + w + margin_x)
        y2 = min(h_frame, y + h + margin_y)

        face_crop = frame[y1:y2, x1:x2]
        return face_crop


# ─── Video Frame Extraction ─────────────────────────────
def extract_frames_from_video(video_path, num_frames=None, uniform=True):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (None = all frames)
        uniform: If True, extract uniformly spaced frames

    Returns:
        List of frames (numpy arrays in BGR format)
    """
    if num_frames is None:
        num_frames = config.FRAMES_PER_VIDEO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame indices to extract
    if uniform and total_frames > num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = list(range(min(total_frames, num_frames)))

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def extract_faces_from_video(video_path, num_frames=None):
    """
    Extract face crops from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of face crops (numpy arrays in BGR format)
    """
    detector = FaceDetector()
    frames = extract_frames_from_video(video_path, num_frames)

    face_crops = []
    for frame in frames:
        face = detector.extract_face(frame)
        if face is not None:
            # Resize to standard size
            face_resized = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE))
            face_crops.append(face_resized)

    return face_crops


# ─── Preprocessing Pipeline ─────────────────────────────
def preprocess_frame(frame):
    """
    Preprocess a single frame for model input.
    Converts BGR to RGB, resizes to IMG_SIZE, normalizes to [0, 1].
    """
    if frame is None:
        return None

    # Resize
    frame_resized = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0

    return frame_normalized


def preprocess_video(video_path, num_frames=None):
    """
    Full preprocessing pipeline for a video.
    Extracts frames → detects faces → crops → resizes → normalizes.

    Returns:
        numpy array of shape (N, H, W, 3) with preprocessed face frames
    """
    face_crops = extract_faces_from_video(video_path, num_frames)

    if len(face_crops) == 0:
        # If no faces detected, use full frames
        frames = extract_frames_from_video(video_path, num_frames)
        processed = []
        for frame in frames:
            p = preprocess_frame(frame)
            if p is not None:
                processed.append(p)
        return np.array(processed) if processed else None

    processed = []
    for face in face_crops:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        processed.append(face_normalized)

    return np.array(processed)


# ─── Batch Video Processing ─────────────────────────────
def process_video_directory(video_dir, output_dir, label, num_frames=None):
    """
    Process all videos in a directory and save extracted face frames.

    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save extracted frames
        label: 'real' or 'fake'
        num_frames: Frames per video
    """
    os.makedirs(output_dir, exist_ok=True)
    video_extensions = config.ALLOWED_EXTENSIONS

    video_files = [
        f for f in os.listdir(video_dir)
        if f.split(".")[-1].lower() in video_extensions
    ]

    print(f"\nProcessing {len(video_files)} {label} videos from: {video_dir}")

    detector = FaceDetector()
    processed_count = 0

    for video_file in tqdm(video_files, desc=f"Extracting {label} faces"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        try:
            frames = extract_frames_from_video(video_path, num_frames)

            for i, frame in enumerate(frames):
                face = detector.extract_face(frame)
                if face is not None:
                    face_resized = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE))
                    output_path = os.path.join(
                        output_dir, f"{video_name}_frame_{i:04d}.jpg"
                    )
                    cv2.imwrite(output_path, face_resized)
                    processed_count += 1

        except Exception as e:
            print(f"  Error processing {video_file}: {e}")
            continue

    print(f"  Saved {processed_count} face frames to: {output_dir}")
    return processed_count


if __name__ == "__main__":
    """Run preprocessing on the dataset directories."""
    real_output = os.path.join(config.PROCESSED_DIR, "real")
    fake_output = os.path.join(config.PROCESSED_DIR, "fake")

    # Process real videos
    if os.path.exists(config.DATASET_REAL_DIR):
        process_video_directory(
            config.DATASET_REAL_DIR, real_output, "real", config.FRAMES_PER_VIDEO
        )

    # Process fake videos
    if os.path.exists(config.DATASET_FAKE_DIR):
        process_video_directory(
            config.DATASET_FAKE_DIR, fake_output, "fake", config.FRAMES_PER_VIDEO
        )

    print("\nPreprocessing complete!")
    print(f"  Real frames: {len(os.listdir(real_output)) if os.path.exists(real_output) else 0}")
    print(f"  Fake frames: {len(os.listdir(fake_output)) if os.path.exists(fake_output) else 0}")
