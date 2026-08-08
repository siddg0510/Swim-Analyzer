#!/usr/bin/env python3
"""
Downloads the MediaPipe Pose Landmarker task file needed for biomechanical analysis.
"""
import os
import urllib.request
from pathlib import Path
import sys

def download_pose_model():
    models_dir = Path(__file__).parent.resolve() / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "pose_landmarker.task"
    if model_path.exists():
        if model_path.stat().st_size > 1000000:
            print(f"Model already exists at {model_path} (size: {model_path.stat().st_size} bytes)")
            return True
        else:
            print("Model file exists but seems too small (corrupted/HTML error page). Re-downloading...")
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    print(f"Downloading MediaPipe pose model from {url}...")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Successfully downloaded to {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_pose_model()
    sys.exit(0 if success else 1)
