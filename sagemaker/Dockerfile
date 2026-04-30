FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git wget ninja-build libglib2.0-0 libgl1-mesa-glx && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python
# Ensure pip/python point to the same 3.12
RUN python -m pip install --upgrade pip

# PyTorch (CUDA 11.8)
RUN python -m pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify torch is importable before building detectron2
RUN python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

# Core deps (from sfada env)
RUN python -m pip install \
    omegaconf==2.3.0 hydra-core==1.3.2 pyyaml \
    timm==1.0.25 einops==0.8.2 fairscale==0.4.13 \
    pycocotools==2.0.11 opencv-python==4.11.0.86 \
    scipy pillow==12.0.0 tqdm tabulate termcolor \
    fvcore iopath cloudpickle pybind11 \
    matplotlib tensorboard accelerate peft \
    sagemaker-training boto3

# Copy and build detectron2 from source
# --no-build-isolation: setup.py does `import torch` at top level
# FORCE_CUDA=1: no GPU during docker build, but we need CUDA extensions
# TORCH_CUDA_ARCH_LIST: explicit arch list since no GPU to auto-detect
COPY external/detrex/detectron2 /tmp/detectron2
RUN cd /tmp/detectron2 && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    python -m pip install --no-build-isolation . && rm -rf /tmp/detectron2

# Copy and build detrex from source
COPY external/detrex /tmp/detrex
RUN cd /tmp/detrex && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9" \
    python -m pip install --no-build-isolation . && rm -rf /tmp/detrex/build

# Verify both packages are importable
RUN python -c "import detectron2; print(f'detectron2={detectron2.__version__}')" && \
    python -c "import detrex; print('detrex OK')"

# Copy project code
ENV WORKDIR=/opt/ml/code
COPY src/ ${WORKDIR}/src/
COPY baselines/ ${WORKDIR}/baselines/
COPY configs/ ${WORKDIR}/configs/
COPY sagemaker/sagemaker_entry.py ${WORKDIR}/sagemaker_entry.py
COPY sagemaker/s3_sync.py ${WORKDIR}/s3_sync.py

# detrex_adapter.py resolves paths relative to src/models -> ../../external/detrex
# Copy projects dir directly (symlinks can cause issues at runtime)
RUN mkdir -p ${WORKDIR}/external/detrex && \
    cp -r /tmp/detrex/projects ${WORKDIR}/external/detrex/projects && \
    ls ${WORKDIR}/external/detrex/projects/dino/configs/dino-resnet/dino_r50_4scale_12ep.py

ENV SAGEMAKER_PROGRAM=sagemaker_entry.py
ENV PYTHONPATH=${WORKDIR}
WORKDIR ${WORKDIR}
