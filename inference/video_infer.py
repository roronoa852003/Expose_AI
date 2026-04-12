import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor

# Use Hub repo ID + cache_dir — transformers finds the pre-synced files in cache
_REPO_ID = os.environ.get("HF_MODEL_REPO", "vedanthundare/expose_ai")
_HF_TOKEN = os.environ.get("HF_TOKEN")
_CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")

video_model = ViTForImageClassification.from_pretrained(
    _REPO_ID,
    subfolder="models/video_model",
    token=_HF_TOKEN,
    cache_dir=_CACHE_DIR
)
video_extractor = AutoImageProcessor.from_pretrained(
    _REPO_ID,
    subfolder="models/video_model",
    token=_HF_TOKEN,
    cache_dir=_CACHE_DIR
)
video_model.eval()


def video_fake_probability(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_probs = []
    frame_id = 0

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the first frame and then every N frames
        if frame_id == 0 or frame_id % frame_interval == 0:
            # Resize frame to ViT input size immediately to save memory
            frame_resized = cv2.resize(frame, (224, 224))
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            inputs = video_extractor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = video_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                # Frame index 1 is FAKE based on most ViT deepfake models
                frame_probs.append(probs[0][1].item()) 
                
            del inputs, outputs, image, frame_resized

        frame_id += 1

    cap.release()
    return float(np.mean(frame_probs)) if frame_probs else 0.0

def image_fake_probability(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = video_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = video_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return float(probs[0][1].item())  # Fake index based on config
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error extracting image prob: {e}")
        return 0.3