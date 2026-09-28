FROM ghcr.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:33fd41e37b822b119b838baa2df58981274e7b5cb1e37932d27621c0ae4f8b1d

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        procps \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .

ENV PYTHONPATH=/workspace
CMD ["bash"]
