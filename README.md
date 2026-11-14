# Ripple-Watermark

Keyed, distortion-free watermarking for LLM output, running on **Ripple-65B-Preview**
(Ricursive, Apache-2.0) through llama.cpp. Every generated token is chosen by a keyed
exponential-minimum race, so the output distribution is unchanged for anyone without the
key, and the key holder can detect the text and say *which model* wrote it, with an exact p-value.

## Layout

```
ripple_fingerprint/
  core.py          Watermarker: keyed sampling + detection (Gamma(n,1) test)
  attribution.py   ModelRegistry: per-model keys from one master key, "which model wrote this?", span finding
  llamacpp.py      Backend for llama-server (per-token probabilities -> watermarked sampling in Python)
  mlx.py           Apple MLX backend
  hf.py            ⚠ currently an exact duplicate of core.py: the real Hugging Face backend is missing
demo.py            One watermarked + one plain generation, then detect/attribute
scripts/
  setup_pod.sh     Fresh pod -> deps, llama.cpp (pinned commit, CUDA), model download + sha256 check
  start_server.sh  llama-server with the required --swa-full flag
  eval_detection.py  Story + code, watermarked vs plain, detection + attribution + speed
  cache_probe.py   Shows how many prompt tokens llama-server recomputes per request
tests/
  test_sampler_equivalence.py  Optimized sampler picks the identical token as the original (2000 cases)
```

## Restore after a pod crash

```bash
git clone git@github.com:tejaDhulipala/Ripple-Watermark.git /workspace && cd /workspace
bash scripts/setup_pod.sh                  # ~42 GB disk; model download is resumable
nohup scripts/start_server.sh > /root/llama-server.log 2>&1 &
curl -s --retry 60 --retry-delay 2 --retry-all-errors http://127.0.0.1:8080/health

python3 scripts/eval_detection.py          # ~1.5 min
```

### The key

The master key is committed at `keys/demo-master.key` **on purpose, because this is a demo.**
Anyone with the key can detect the watermark *and forge it*, so a public key gives no real
protection. For real use, generate a private key, keep it out of git (`.gitignore` blocks other
`*.key` files), and point the scripts at it with `RIPPLE_KEY=/path/to/private.key`. Text
watermarked under one key is only detectable with that same key.

### Environment it was built on

| | |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, 48 GB (driver 595.91.07, CUDA 13.2) |
| Model | `AtomicChat/Ripple-65B-Preview-GGUF` @ `df550a7`, `Ripple-65B-Preview-Q4_K_M.gguf`, 38,798,820,576 bytes, sha256 `51de49f0…7b8652` |
| llama.cpp | `9183a552e5fd` (2026-11-14), built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` |
| Python | 3.12.3, numpy 2.5.4, scipy 1.18.1 |
| VRAM in use | ~40 GB (weights + 8K context with `--swa-full`) |

Model architecture (`general.architecture = ricursive`): 80 layers, 5120 hidden, 80 heads / 8 KV
heads, vocab 100,352, sliding window 1024 on 4 of every 5 layers, ChatML template
(`<|im_start|>` … `<|im_end|>`). It sometimes emits a `<think>…</think>` block before answering.

## How the llama.cpp backend works

For each token, `llamacpp.py` asks llama-server for the **top 1000 raw next-token log-probs**
(`n_predict=1`, `n_probs=1000`, `post_sampling_probs=false`, prompt cache on). It applies
temperature / top-p in Python and lets `Watermarker.sample` pick the token. One HTTP request per
token over a kept-alive connection.

```python
from ripple_fingerprint import ModelRegistry
from ripple_fingerprint.llamacpp import LlamaServer, generate

srv = LlamaServer("http://127.0.0.1:8080")
reg = ModelRegistry(master_key_bytes)
wm  = reg.register("ripple-65b-preview", encode=srv.tokenize)
text = generate(srv, "Write a haiku.", wm, temperature=1.0, top_p=0.95)
reg.attribute(text).best          # -> "ripple-65b-preview"
```

## Results so far (2026-11-14)

### Speed

| | tokens/sec |
|---|---|
| Plain Ripple (`llama-cli`, no watermark) | ~21 |
| Watermarked, first version | 5-6 |
| Watermarked, now | **16-17.5** |

The fixes:
1. **`--swa-full` (the big one).** Without it, llama-server recomputed the last ~6 tokens on
   every request because the sliding-window cache can't be rewound (120-138 ms/request vs 48 ms).
   See `scripts/cache_probe.py`.
2. `Watermarker.sample` races only tokens with nonzero probability (~1000) instead of all
   100k. Verified identical picks in `tests/test_sampler_equivalence.py`. The scheme and keys
   are unchanged.
3. 1000 candidates instead of 2000, plus a persistent HTTP connection.

### Detection (one sample each, temperature 1.0, top-p 0.95, ≤400 tokens, fpr = 1e-5)

| Text | Watermarked? | p-value | Detected | Attributed to Ripple |
|---|---|---|---|---|
| Story | yes | 1e-22 to 3e-25 | ✅ | ✅ |
| Story | no | 0.35 / 0.54 | ✅ not flagged | ✅ none |
| Code | yes | 3e-6 / 3e-7 | ✅ | ⚠ first run just missed (adj. p 1.08e-5), second passed |
| Code | no | 0.08 / 0.025 | ✅ not flagged | ✅ none |

A decoy key never matched anything. Detection works from the decoded text (re-tokenized),
not just the original token ids.

## Open issues / next steps

- **Code is low-entropy** (~0.3 bits/token vs ~1.1 for prose), so it needs roughly 600+ tokens
  for reliable attribution at fpr 1e-5.
- **Plain code scored 1.105 and 1.109 mean token score** (null expectation 1.0, about 2σ high
  both times). Probably chance, but it could indicate a bias that inflates false positives.
  Needs the larger evaluation.
- **`<think>` blocks:** if a UI hides the reasoning, detection relies on the visible answer
  only. Decide whether to disable thinking in the template or detect on the visible text only.
- **Proper evaluation:** 20-50 prompts per category, detection rate vs text length,
  false-positive rate on plain text.
- **`hf.py`** needs the real Hugging Face backend (the current file is a copy of `core.py`).
- Possible further speedup: sample inside llama.cpp (Python bindings / C API) to remove the
  per-token HTTP round trip and get close to 21 tok/s.
