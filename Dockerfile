# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set HF cache to a predictable path so sync_models.py and from_pretrained share it
ENV HF_HOME=/app/hf_cache

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Run sync_models.py first, then start Gunicorn with 1 worker and increased timeout
CMD ["sh", "-c", "python sync_models.py && gunicorn -w 1 --timeout 300 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:7860"]
