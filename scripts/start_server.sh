#!/usr/bin/env bash
# Serve Ripple-65B on the GPU for the watermarking backend (ripple_fingerprint/llamacpp.py).
#
# --swa-full is REQUIRED for speed: Ripple uses sliding-window attention on 64 of
# 80 layers. Without the full SWA cache, llama-server can't rewind it exactly and
# recomputes the last ~6 tokens on every per-token request (~120 ms vs ~48 ms).
# Costs ~2.7 GB extra VRAM at -c 8192. Total VRAM used: ~40 GB of 48.
set -euo pipefail
MODEL=${MODEL:-/root/models/ripple-65b/Ripple-65B-Preview-Q4_K_M.gguf}
LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
CTX=${CTX:-8192}
PORT=${PORT:-8080}
exec "$LLAMA_DIR/build/bin/llama-server" -m "$MODEL" -ngl 999 -c "$CTX" --parallel 1 --swa-full \
     --host 127.0.0.1 --port "$PORT"
