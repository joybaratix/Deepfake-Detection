"""
Flask Web Application for Deepfake Detection.
Minimalist interface for uploading videos and viewing detection results.
Runs fully offline — no external API calls.
"""

import os
import sys
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.predict import VideoPredictor
from src.utils import get_video_info

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOADS_DIR

# Global predictor (loaded once)
predictor = None


def get_predictor():
    """Lazy-load the predictor."""
    global predictor
    if predictor is None:
        predictor = VideoPredictor()
    return predictor


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Home page — video upload interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle video upload and run deepfake detection."""
    if "video" not in request.files:
        return render_template("index.html", error="No video file selected.")

    file = request.files["video"]
    if file.filename == "":
        return render_template("index.html", error="No video file selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error=f"Unsupported format. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"{unique_id}_{filename}"
    filepath = os.path.join(config.UPLOADS_DIR, safe_name)
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    file.save(filepath)

    # Get video info
    video_info = get_video_info(filepath)

    # Run prediction
    try:
        pred = get_predictor()
        result = pred.predict_video(filepath)

        # Save heatmap and frame images for display
        result_dir = os.path.join(app.static_folder, "results", unique_id)
        os.makedirs(result_dir, exist_ok=True)

        # Save suspicious frame
        frame_filename = None
        if result.get("suspicious_frame") is not None:
            frame_filename = "frame.jpg"
            frame_bgr = cv2.cvtColor(result["suspicious_frame"], cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (config.IMG_SIZE, config.IMG_SIZE))
            cv2.imwrite(os.path.join(result_dir, frame_filename), frame_resized)

        # Save heatmap overlay
        heatmap_filename = None
        if result.get("heatmap_overlay") is not None:
            heatmap_filename = "heatmap.jpg"
            overlay_bgr = cv2.cvtColor(result["heatmap_overlay"], cv2.COLOR_RGB2BGR)
            overlay_resized = cv2.resize(overlay_bgr, (config.IMG_SIZE, config.IMG_SIZE))
            cv2.imwrite(os.path.join(result_dir, heatmap_filename), overlay_resized)

        return render_template(
            "result.html",
            prediction=result["prediction"],
            confidence=result["confidence"],
            probability=result["probability"],
            num_frames=result["num_frames_analyzed"],
            frame_predictions=result["frame_predictions"],
            filename=filename,
            video_info=video_info,
            frame_image=f"results/{unique_id}/{frame_filename}" if frame_filename else None,
            heatmap_image=f"results/{unique_id}/{heatmap_filename}" if heatmap_filename else None,
        )

    except Exception as e:
        return render_template("index.html", error=f"Error processing video: {str(e)}")

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for deepfake detection."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported video format"}), 400

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(config.UPLOADS_DIR, f"{unique_id}_{filename}")
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    file.save(filepath)

    try:
        pred = get_predictor()
        result = pred.predict_video(filepath)

        return jsonify({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probability": result["probability"],
            "frames_analyzed": result["num_frames_analyzed"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Deepfake Detection — Web Interface")
    print("=" * 50)
    print(f"  Open: http://localhost:{config.FLASK_PORT}")
    print(f"  Device: {config.DEVICE}")
    print("=" * 50 + "\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
    )
