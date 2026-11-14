#!/usr/bin/env bash
# Rebuild everything on a fresh GPU pod: Python deps, llama.cpp (CUDA), Ripple-65B GGUF.
# Tested on: RTX 6000 Ada 48 GB, CUDA 13.2, driver 595.91.07, Ubuntu / Python 3.12.
# Needs ~42 GB free disk (model 38.8 GB + llama.cpp build).
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/root/models/ripple-65b}
LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
LLAMA_COMMIT=9183a552e5fdcbdb5243f187f3a63204eca7dba9    # first build known to load Ripple (arch "ricursive")
HF_REPO=AtomicChat/Ripple-65B-Preview-GGUF
HF_REV=df550a7178a87bbfb6ee2bdbd37603c553d536ee
MODEL_FILE=Ripple-65B-Preview-Q4_K_M.gguf
MODEL_SHA256=51de49f0833201d82e624c1273a0316dca6c281616371770213f0c6d277b8652
CUDA_ARCH=${CUDA_ARCH:-89}   # 89 = Ada (RTX 6000 Ada / 4090). 86 = Ampere A6000, 90 = H100.

echo "== Python deps"
pip install -q -r "$(dirname "$0")/../requirements.txt"

echo "== llama.cpp @ ${LLAMA_COMMIT:0:7} (CUDA arch $CUDA_ARCH)"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT" 2>/dev/null || git -C "$LLAMA_DIR" fetch origin
git -C "$LLAMA_DIR" checkout -q "$LLAMA_COMMIT"
export PATH=/usr/local/cuda/bin:$PATH
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" --target llama-cli llama-server > /dev/null
echo "built $LLAMA_DIR/build/bin/llama-server"

echo "== Model: $HF_REPO/$MODEL_FILE (38.8 GB, resumable)"
mkdir -p "$MODEL_DIR"
curl -L -C - --fail -o "$MODEL_DIR/$MODEL_FILE" \
  "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$MODEL_FILE"
echo "== Verifying sha256 (takes a minute or two)"
echo "$MODEL_SHA256  $MODEL_DIR/$MODEL_FILE" | sha256sum -c -

echo "Done. Next: scripts/start_server.sh, then restore your master key (see README)."
