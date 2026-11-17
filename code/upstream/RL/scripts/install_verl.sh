#!/bin/bash
set -euxo pipefail

echo "========================================"
echo "Create stable RL environment"
echo "========================================"

# ---------------------------------------
# 1. Upgrade pip
# ---------------------------------------

# python -m pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1

# ---------------------------------------
# 2. Install PyTorch stack
# ---------------------------------------

pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    torchdata==0.11.0 \
    tensordict==0.6.2

# ---------------------------------------
# 3. Install vLLM + inference stack
# ---------------------------------------

pip install --no-cache-dir \
    vllm==0.8.5.post1 \
    ray==2.52.1 \
    xformers==0.0.29.post2 \
    triton==3.2.0

# ---------------------------------------
# 4. Install FlashAttention
# ---------------------------------------

wget -nv \
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install --no-cache-dir \
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ---------------------------------------
# 5. Install FlashInfer (IMPORTANT)
# ---------------------------------------

pip install --no-cache-dir \
flashinfer-python==0.2.5 \
--extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/

# ---------------------------------------
# 6. Core training packages
# ---------------------------------------

pip install --no-cache-dir \
    transformers==4.51.1 \
    accelerate==1.12.0 \
    datasets==4.4.1 \
    peft==0.18.0 \
    huggingface-hub==0.36.0 \
    hf_transfer==0.1.9 \
    pyarrow==22.0.0 \
    pandas==2.3.3 \
    hydra-core==1.3.2 \
    wandb==0.23.0 \
    dill==0.4.0 \
    pybind11==3.0.1 \
    liger_kernel==0.6.4 \
    mathruler==0.1.0 \
    blobfile==3.0.0 \
    xgrammar==0.1.18 \
    codetiming==1.4.0 \
    qwen-vl-utils==0.0.14

# ---------------------------------------
# 7. CUDA / runtime compatibility
# ---------------------------------------

pip install --no-cache-dir \
    cupy-cuda12x==13.6.0 \
    nvidia-ml-py==13.580.82 \
    grpcio==1.76.0 \
    protobuf==6.33.1 \
    fastapi==0.123.0 \
    uvicorn==0.38.0 \
    pydantic==2.12.4 \
    optree==0.18.0

# ---------------------------------------
# 8. OpenCV fix
# ---------------------------------------

pip install opencv-python==4.12.0.88
pip install opencv-python-headless==4.12.0.88
pip install opencv-fixer==0.2.5

python -c "from opencv_fixer import AutoFix; AutoFix()"

# ---------------------------------------
# 9. Install verl
# ---------------------------------------

pip install --no-deps -e verl/

echo "========================================"
echo "Environment installation completed"
echo "========================================"