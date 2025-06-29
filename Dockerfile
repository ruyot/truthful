FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create directory for models
RUN mkdir -p /var/models

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy skeleton model to models directory
COPY results/skeleton_model_v3.pt /var/models/
RUN chmod 644 /var/models/skeleton_model_v3.pt

# Copy cache script and run it to cache model weights in the image
COPY scripts/cache_weights.py /app/scripts/
ENV SKELETON_MODEL_PATH=/var/models/skeleton_model_v3.pt
RUN python scripts/cache_weights.py

# Copy application code
COPY . .

# Remove the original model file to save space
RUN rm -f results/skeleton_model_v3.pt

# Expose port
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]