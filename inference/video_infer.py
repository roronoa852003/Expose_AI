import cv2
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor


VIDEO_MODEL_PATH = "/app/models/video_model"

video_model = ViTForImageClassification.from_pretrained(VIDEO_MODEL_PATH)
video_extractor = AutoImageProcessor.from_pretrained(VIDEO_MODEL_PATH)
video_model.eval()


def video_fake_probability(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_probs = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = video_extractor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = video_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                frame_probs.append(probs[0][1].item())  # FAKE prob

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