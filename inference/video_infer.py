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

# Detect which output index corresponds to the FAKE class from the model config.
# Falls back to index 1 if the label map is missing or ambiguous.
_id2label = getattr(video_model.config, "id2label", {})
_FAKE_IDX = next(
    (int(k) for k, v in _id2label.items() if str(v).upper() in ("FAKE", "1", "FORGED", "DEEPFAKE", "MANIPULATED")),
    1  # default assumption: index 1 = FAKE
)
print(f"[video_infer] id2label={_id2label}  →  using FAKE index={_FAKE_IDX}")


def video_fake_probability(video_path, frame_interval=10):
    """Sample frames from a video and return mean FAKE probability.

    frame_interval=10 gives ~3 samples/sec at 30fps — denser than the old 30
    while still being memory-friendly.
    """
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
                # Use the dynamically detected FAKE index from model config
                fake_score = probs[0][_FAKE_IDX].item()
                frame_probs.append(fake_score)

            del inputs, outputs, image, frame_resized

        frame_id += 1

    cap.release()

    if not frame_probs:
        print("[video_infer] WARNING: No frames were extracted from the video.")
        return 0.0

    result = float(np.mean(frame_probs))
    print(f"[video_infer] video_fake_probability={result:.4f}  (frames sampled={len(frame_probs)}, FAKE_IDX={_FAKE_IDX})")
    return result

def image_fake_probability(image_path):
    """Run the ViT model on a single image and return FAKE probability."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = video_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = video_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs = probs[0].tolist()
            fake_score = probs[0][_FAKE_IDX].item()
            print(f"[video_infer] image_fake_probability={fake_score:.4f}  all_probs={all_probs}  FAKE_IDX={_FAKE_IDX}")
            return float(fake_score)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[video_infer] Error extracting image prob: {e}")
        # Return None so fusion can skip this modality rather than anchor at 0.3
        return None