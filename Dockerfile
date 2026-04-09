# Watcher + capstone pipeline (GPU-enabled)
# Requires NVIDIA Container Toolkit on the host for GPU access.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages once at build time (not at container startup)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

CMD ["python3", "IncomingFileEventHandler.py", \
     "--watch-path", "/srv/FITSfileDropFolder", \
     "--outdir", "/app/outputs_capstone", \
     "--device", "gpu"]
