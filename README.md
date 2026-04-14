---
title: Expose AI API
emoji: 🕵️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Expose AI - Multimodal Deepfake Detector API

This is the backend for the Expose AI project, hosted as a Docker Space on Hugging Face.

## Deployment Details
- **SDK**: Docker
- **Port**: 7860
- **Logic**: FastAPI + Transformers + FFmpeg

## Environment Variables Required
- `HF_MODEL_REPO`: The Hugging Face Repository ID where model weights are stored.

## Image-Only Evaluation Loop
- Create a CSV file with headers: `path,label`
- Label values can be `0/1` or `real/fake`
- Run threshold sweep:
  - `python scripts/evaluate_image_only.py --csv your_image_eval.csv --start-threshold 0.50 --end-threshold 0.85 --step 0.02`
- The script writes `image_eval_results.json` with confusion metrics at each threshold and the best threshold by F1.
