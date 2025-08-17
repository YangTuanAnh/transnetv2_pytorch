# Use an official PyTorch CUDA image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Install ffmpeg and python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    ffmpeg-python \
    Pillow

# Copy code into container
WORKDIR /tmp
COPY setup.py .
COPY transnetv2_pytorch ./transnetv2_pytorch

# Install as package
RUN pip install -e .

# Default command
ENTRYPOINT ["transnetv2_pytorch"]
