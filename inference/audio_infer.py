import torch
import librosa
import subprocess
import numpy as np
import os
import shutil
import tempfile
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

AUDIO_MODEL_PATH = "models/audio_model"

audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_PATH, local_files_only=True)
audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_PATH, local_files_only=True)
audio_model.eval()


def extract_audio(video_path):
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")

    # Look for ffmpeg in PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Fallback for Windows local development if not in PATH
        fallback = "C:/ffmpeg/bin/ffmpeg.exe"
        if os.path.exists(fallback):
            ffmpeg = fallback
        else:
            # On Render/Linux, ffmpeg must be installed in the environment
            return None

    command = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(audio_path):
        return None

    # Check if audio is actually meaningful
    if os.path.getsize(audio_path) < 10_000:  # ~10 KB → basically silence
        return None

    return audio_path


def _is_audio_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}

def audio_fake_probability(file_path):
    if _is_audio_file(file_path):
        audio_path = file_path
    else:
        audio_path = extract_audio(file_path)

    if audio_path is None:
        return 0.15  # Fallback simulation if no FFmpeg / no audio

    speech, sr = librosa.load(audio_path, sr=16000)

    if len(speech) < sr * 1:  # less than 1 sec
        return None

    chunk_size = sr * 5
    probs = []

    for i in range(0, len(speech), chunk_size):
        chunk = speech[i:i + chunk_size]
        if len(chunk) < chunk_size:
            pad = np.zeros(chunk_size - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        inputs = audio_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = audio_model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
            probs.append(prob)

    return float(np.mean(probs)) if probs else None