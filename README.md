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
