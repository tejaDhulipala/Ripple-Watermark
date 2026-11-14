"""Apple MLX backend (needs ``mlx-lm``). Runs MLX models such as LM Studio's.

    from mlx_lm import load
    model, tok = load("/path/to/Some-Model-MLX-8bit")
    text = generate(model, tok, "Write a haiku.", wm)

Detection only needs the tokenizer: use ``Watermarker.detect(tok.encode(text))``
or ``ripple_fingerprint.hf.detect_text`` with a Hugging Face tokenizer loaded
from the same folder.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .core import Watermarker


def warp(logits: np.ndarray, temperature: float, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    """Temperature / top-k / top-p in numpy. Returns a float64 distribution."""
    z = logits.astype(np.float64) / temperature
    if top_k and 0 < top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        drop = order[cum - p[order] > top_p]
        p[drop] = 0.0
        p /= p.sum()
    return p


def generate_ids(
    model,
    prompt_ids: list[int],
    watermarker: Watermarker,
    eos_ids: set[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    watermark: bool = True,
    on_step=None,
) -> list[int]:
    """Token-by-token generation. ``watermark=False`` gives plain sampling with the same settings.

    ``on_step(probs, history, token, state)`` runs after each choice, before the
    token is appended to ``history`` (used by the demo's visualizer).
    """
    if temperature <= 0:
        raise ValueError("greedy decoding has no randomness to watermark; use temperature > 0")
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)[0, -1]
    history = list(prompt_ids)
    state = watermarker.new_state()
    out: list[int] = []
    for _ in range(max_new_tokens):
        p = warp(np.array(logits.astype(mx.float32)), temperature, top_k, top_p)
        if watermark:
            tok = watermarker.sample(p, history, state)
        else:
            tok = int(state.rng.choice(p.size, p=p))
            state.last_mode = "plain"
        if on_step is not None:
            on_step(p, history, tok, state)
        if tok in eos_ids:
            break
        history.append(tok)
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
    return out


def generate(model, tokenizer, prompt: str, watermarker: Watermarker, chat: bool = True, **kw) -> str:
    """Prompt in, (watermarked) completion text out. ``tokenizer`` comes from ``mlx_lm.load``."""
    if chat and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False
        )
    else:
        ids = tokenizer.encode(prompt)
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    return tokenizer.decode(generate_ids(model, ids, watermarker, eos, **kw))