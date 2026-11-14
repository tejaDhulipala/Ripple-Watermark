"""llama.cpp backend: watermarked sampling through a running ``llama-server``.

The server only supplies next-token probabilities (``n_probs`` top candidates,
taken *before* its own sampler runs); the token is chosen here, in Python, by
``Watermarker.sample``. One HTTP request per generated token, with the prompt
cache keeping each request incremental.

    srv = LlamaServer("http://127.0.0.1:8080")
    text = generate(srv, "Write a haiku.", wm)
    detect_text(srv, wm, text)

The distribution is truncated to the server's top ``n_probs`` tokens (default
1000), i.e. an implicit top-k. On Ripple-65B the top 1000 hold ~99.5% of the
mass at temperature 1, and top-p < 1 normally cuts well inside that.

Start the server with ``--swa-full``. Ripple uses sliding-window attention, and
without the full SWA cache llama-server can't rewind it exactly, so it
recomputes the last ~6 tokens on every request (~2.5x slower here).

    llama-server -m Ripple-65B-Preview-Q4_K_M.gguf -ngl 999 -c 8192 --parallel 1 --swa-full
"""

from __future__ import annotations

import http.client
import json
import urllib.parse
from typing import Sequence

import numpy as np

from .core import DetectionResult, Watermarker


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 300.0):
        self.url = url.rstrip("/")
        u = urllib.parse.urlsplit(self.url)
        self._host, self._port = u.hostname, u.port or 80
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None  # kept alive across requests
        self._eos: set[int] | None = None

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if body is not None else {}
        for attempt in (0, 1):  # reconnect once if the kept-alive connection was dropped
            if self._conn is None:
                self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
            try:
                self._conn.request(method, path, body=body, headers=headers)
                r = self._conn.getresponse()
                data = r.read()
            except (http.client.HTTPException, ConnectionError, OSError):
                self._conn.close()
                self._conn = None
                if attempt:
                    raise
                continue
            if r.status != 200:
                raise RuntimeError(f"llama-server {method} {path} -> HTTP {r.status}: {data[:300]!r}")
            return json.loads(data)
        raise AssertionError("unreachable")

    def _post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, payload)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    # ------------------------------------------------------------ tokenizer
    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = False) -> list[int]:
        out = self._post("/tokenize", {"content": text, "add_special": add_special, "parse_special": parse_special})
        return [int(t) for t in out["tokens"]]

    def detokenize(self, ids: Sequence[int]) -> str:
        return self._post("/detokenize", {"tokens": [int(t) for t in ids]})["content"]

    def chat_prompt_ids(self, prompt: str, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self._post("/apply-template", {"messages": msgs})["prompt"]
        return self.tokenize(text, add_special=True, parse_special=True)

    def eos_ids(self) -> set[int]:
        if self._eos is None:
            props = self._get("/props")
            ids: set[int] = set()
            for t in {props.get("eos_token"), "<|im_end|>", "<|endoftext|>"} - {None, ""}:
                tok = self.tokenize(t, parse_special=True)
                if len(tok) == 1:
                    ids.add(tok[0])
            self._eos = ids
        return self._eos

    # --------------------------------------------------------- distribution
    def next_token_logprobs(self, ids: Sequence[int], n_probs: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Top ``n_probs`` (token ids, raw log-probs) for the token after ``ids``."""
        out = self._post(
            "/completion",
            {
                "prompt": [int(t) for t in ids],
                "n_predict": 1,
                "n_probs": n_probs,
                "cache_prompt": True,
                "post_sampling_probs": False,  # raw model distribution, pre-sampler
                "top_k": 1,  # the server's own pick is discarded
            },
        )
        top = out["completion_probabilities"][0]["top_logprobs"]
        return (
            np.array([t["id"] for t in top], dtype=np.int64),
            np.array([t["logprob"] for t in top], dtype=np.float64),
        )


def warp(
    tok_ids: np.ndarray, logprobs: np.ndarray, temperature: float, top_k: int = 0, top_p: float = 1.0
) -> np.ndarray:
    """Dense distribution over ids ``0..max(tok_ids)`` after temperature / top-k / top-p."""
    z = logprobs / temperature
    if top_k and 0 < top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        p[order[cum - p[order] > top_p]] = 0.0
        p /= p.sum()
    dense = np.zeros(int(tok_ids.max()) + 1, dtype=np.float64)
    dense[tok_ids] = p
    return dense


def generate_ids(
    server: LlamaServer,
    prompt_ids: list[int],
    watermarker: Watermarker,
    eos_ids: set[int] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    n_probs: int = 1000,
    watermark: bool = True,
    on_step=None,
) -> list[int]:
    """Token-by-token generation. ``watermark=False`` gives plain sampling with the same settings.

    ``on_step(probs, history, token, state)`` runs after each choice, as in the MLX backend.
    """
    if temperature <= 0:
        raise ValueError("greedy decoding has no randomness to watermark; use temperature > 0")
    eos = server.eos_ids() if eos_ids is None else eos_ids
    history = list(prompt_ids)
    state = watermarker.new_state()
    out: list[int] = []
    for _ in range(max_new_tokens):
        ids, lp = server.next_token_logprobs(history, n_probs)
        p = warp(ids, lp, temperature, top_k, top_p)
        if watermark:
            tok = watermarker.sample(p, history, state)
        else:
            tok = int(state.rng.choice(p.size, p=p))
            state.last_mode = "plain"
        if on_step is not None:
            on_step(p, history, tok, state)
        if tok in eos:
            break
        history.append(tok)
        out.append(tok)
    return out


def generate(
    server: LlamaServer, prompt: str, watermarker: Watermarker, chat: bool = True, system: str | None = None, **kw
) -> str:
    """Prompt in, (watermarked) completion text out."""
    ids = server.chat_prompt_ids(prompt, system) if chat else server.tokenize(prompt, add_special=True)
    return server.detokenize(generate_ids(server, ids, watermarker, **kw))


def detect_text(server: LlamaServer, watermarker: Watermarker, text: str) -> DetectionResult:
    """Re-tokenize ``text`` with Ripple's tokenizer and test it for this key's watermark."""
    return watermarker.detect(server.tokenize(text))
