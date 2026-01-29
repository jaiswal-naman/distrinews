# =============================================================================
# Dockerfile for Hugging Face Spaces Deployment
# =============================================================================
#
# This Dockerfile creates a container that runs the DistriNews inference API.
#
# WHAT THIS DOES:
# 1. Uses a Python base image
# 2. Installs dependencies
# 3. Copies the inference code
# 4. Starts the FastAPI server
#
# =============================================================================

# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference code
COPY inference/ ./inference/
COPY training/model.py ./training/model.py
COPY training/__init__.py ./training/__init__.py

# Copy the trained model checkpoint
# NOTE: You need to upload your trained checkpoint to the Space
COPY checkpoints/ ./checkpoints/

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Set environment variables
ENV MODEL_CHECKPOINT=/app/checkpoints/distilbert_agnews.pt

# Start the server
# HF Spaces expects the app to run on port 7860
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "7860"]
