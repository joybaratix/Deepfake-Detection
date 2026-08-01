"""
Dataset Download & Sample Data Generation Utility.
Provides instructions for downloading benchmark deepfake datasets
and generates a sample dataset for quick testing.
"""

import os
import sys
import cv2
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Dataset Download Instructions ───────────────────────
DATASET_INFO = """
╔══════════════════════════════════════════════════════════════╗
║          DEEPFAKE DETECTION — DATASET SETUP GUIDE           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  For training the model, you need real and fake videos.      ║
║  Choose one or more of the following datasets:               ║
║                                                              ║
║  1. FaceForensics++ (Recommended for best results)           ║
║     - URL: https://github.com/ondyari/FaceForensics          ║
║     - Contains 1000 real + 5000 fake videos                  ║
║     - Requires filling out a Google Form for access           ║
║     - Includes multiple manipulation methods:                 ║
║       Deepfakes, Face2Face, FaceSwap, NeuralTextures         ║
║                                                              ║
║  2. Celeb-DF v2                                              ║
║     - URL: https://github.com/yuezunli/celeb-deepfakeforensics║
║     - 590 real + 5639 fake celebrity videos                   ║
║     - Higher quality deepfakes                                ║
║     - Request access via their GitHub page                    ║
║                                                              ║
║  3. DFDC (DeepFake Detection Challenge)                      ║
║     - URL: https://www.kaggle.com/c/deepfake-detection-challenge║
║     - 100,000+ videos from Facebook                           ║
║     - Requires Kaggle account                                 ║
║     - Largest publicly available dataset                      ║
║                                                              ║
║  4. DeeperForensics-1.0                                      ║
║     - URL: https://github.com/EndlessSora/DeeperForensics-1.0 ║
║     - 60,000 videos with hidden test set                      ║
║     - High quality with diverse perturbations                 ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SETUP INSTRUCTIONS:                                         ║
║                                                              ║
║  After downloading, organize videos as:                      ║
║                                                              ║
║    data/                                                     ║
║      real/     ← Place all real/authentic videos here        ║
║      fake/     ← Place all deepfake/manipulated videos here  ║
║                                                              ║
║  Then run preprocessing:                                     ║
║    python src/data_preprocessing.py                          ║
║                                                              ║
║  This will extract face frames from all videos into:         ║
║    data/processed/real/                                       ║
║    data/processed/fake/                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def generate_sample_dataset(num_real=50, num_fake=50, frames_per_video=10,
                             video_duration_sec=3, fps=30):
    """
    Generate a synthetic sample dataset for testing the pipeline.
    Creates simple videos with faces drawn on them.
    Real videos have smooth faces, fake videos have visible artifacts.

    This is ONLY for testing the pipeline — not for real training!
    """
    print("Generating sample dataset for pipeline testing...")
    print(f"  Real videos: {num_real}")
    print(f"  Fake videos: {num_fake}")

    real_dir = config.DATASET_REAL_DIR
    fake_dir = config.DATASET_FAKE_DIR
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    frame_count = int(video_duration_sec * fps)
    size = 320

    # Generate real videos (smooth, natural-looking synthetic faces)
    print("\n  Generating real (authentic) sample videos...")
    for i in range(num_real):
        video_path = os.path.join(real_dir, f"real_{i:04d}.mp4")
        _create_sample_video(
            video_path, frame_count, size, fps,
            is_fake=False, seed=i
        )
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{num_real} real videos created")

    # Generate fake videos (with visible artifacts and inconsistencies)
    print("\n  Generating fake (manipulated) sample videos...")
    for i in range(num_fake):
        video_path = os.path.join(fake_dir, f"fake_{i:04d}.mp4")
        _create_sample_video(
            video_path, frame_count, size, fps,
            is_fake=True, seed=i + 1000
        )
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{num_fake} fake videos created")

    print(f"\n  Sample dataset created!")
    print(f"    Real videos: {real_dir}")
    print(f"    Fake videos: {fake_dir}")
    print(f"\n  Next step: Run preprocessing:")
    print(f"    python src/data_preprocessing.py")


def _create_sample_video(path, frame_count, size, fps, is_fake, seed):
    """Create a single sample video with a synthetic face."""
    np.random.seed(seed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (size, size))

    # Random face parameters
    skin_color = (
        np.random.randint(160, 220),
        np.random.randint(140, 200),
        np.random.randint(120, 180),
    )
    bg_color = (
        np.random.randint(30, 80),
        np.random.randint(30, 80),
        np.random.randint(30, 80),
    )

    for f in range(frame_count):
        frame = np.full((size, size, 3), bg_color, dtype=np.uint8)

        # Draw face (ellipse)
        center_x = size // 2 + int(5 * np.sin(f * 0.05))
        center_y = size // 2 + int(3 * np.cos(f * 0.03))
        face_w = 80 + int(5 * np.sin(f * 0.02))
        face_h = 100 + int(5 * np.cos(f * 0.02))

        cv2.ellipse(frame, (center_x, center_y), (face_w, face_h),
                    0, 0, 360, skin_color, -1)

        # Draw eyes
        eye_y = center_y - 20
        left_eye_x = center_x - 30
        right_eye_x = center_x + 30
        cv2.circle(frame, (left_eye_x, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (left_eye_x, eye_y), 4, (50, 50, 50), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 4, (50, 50, 50), -1)

        # Draw mouth
        mouth_y = center_y + 30
        cv2.ellipse(frame, (center_x, mouth_y), (20, 8 + int(3 * np.sin(f * 0.1))),
                    0, 0, 180, (100, 80, 80), -1)

        if is_fake:
            # Add deepfake artifacts
            # 1. Blending boundary (visible seam around face)
            cv2.ellipse(frame, (center_x, center_y), (face_w + 2, face_h + 2),
                       0, 0, 360, (
                           min(255, skin_color[0] + 40),
                           min(255, skin_color[1] + 40),
                           min(255, skin_color[2] + 40),
                       ), 2)

            # 2. Flickering artifacts (random noise patches)
            if f % 3 == 0:
                noise_x = center_x + np.random.randint(-40, 40)
                noise_y = center_y + np.random.randint(-40, 40)
                noise = np.random.randint(0, 50, (15, 15, 3), dtype=np.uint8)
                y1 = max(0, noise_y - 7)
                y2 = min(size, noise_y + 8)
                x1 = max(0, noise_x - 7)
                x2 = min(size, noise_x + 8)
                h = y2 - y1
                w = x2 - x1
                frame[y1:y2, x1:x2] = cv2.addWeighted(
                    frame[y1:y2, x1:x2], 0.7,
                    noise[:h, :w], 0.3, 0
                )

            # 3. Slight color shift (inconsistent lighting)
            if f % 5 == 0:
                shift = np.random.randint(-15, 15)
                frame = np.clip(frame.astype(int) + shift, 0, 255).astype(np.uint8)

            # 4. Compression artifacts
            if f % 4 == 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(20, 50)]
                _, encoded = cv2.imencode(".jpg", frame, encode_param)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        out.write(frame)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Setup Utility")
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a sample dataset for testing"
    )
    parser.add_argument(
        "--num-real", type=int, default=50,
        help="Number of real sample videos to generate"
    )
    parser.add_argument(
        "--num-fake", type=int, default=50,
        help="Number of fake sample videos to generate"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show dataset download instructions"
    )
    args = parser.parse_args()

    if args.info or not args.generate_sample:
        print(DATASET_INFO)

    if args.generate_sample:
        generate_sample_dataset(
            num_real=args.num_real,
            num_fake=args.num_fake
        )
