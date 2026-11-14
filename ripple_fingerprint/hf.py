"""Distortion-free, edit-robust watermarking for LLM sampling.

Scheme (exponential-minimum / Gumbel-max sampling with a keyed PRF):

* At every generation step the model gives a next-token distribution ``p``
  (after temperature / top-k / top-p, whatever you normally use).
* A secret key and the previous ``context_width`` token ids are hashed into a
  seed.  The seed expands (SHAKE-256) into one uniform ``u_v`` in (0, 1) per
  vocabulary entry.
* The chosen token is ``argmax_v  u_v ** (1 / p_v)``.

If the ``u_v`` are uniform, ``P(choose v) = p_v`` exactly (exponential race),
so to anyone without the key each watermarked token is an exact sample from
the model's own distribution. Nothing in the output distribution changes.

With the key, the detector recomputes ``u`` for each observed token and sums
``-log(1 - u_x)``. For text that wasn't watermarked with this key the terms
are i.i.d. Exp(1), so the sum is Gamma(n, 1) and gives an exact p-value.
Watermarked tokens get systematically large ``u``.

The seed depends only on the *local* context window, not on the absolute
position, so substitutions, insertions and deletions only spoil the
``context_width + 1`` tokens around each edit. Every untouched stretch still
counts toward detection.

Two refinements keep the output distribution exact even across a response:

* **Repeated-context masking.** If a context window already occurred earlier
  in the same response, its ``u`` values would be reused, which correlates
  tokens. Those steps use ordinary true-random sampling, and the detector skips
  them.
* **Entropy gate.** Real randomness is used until the response has
  accumulated ``entropy_threshold`` bits of surprisal. This stops two
  generations from the same prompt collapsing onto the same deterministic text.
  Without it, asking the same question twice and getting identical answers
  would give the watermark away.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import gammaincc

_DOMAIN = b"ripple-fingerprint/v1"
_U53 = float(2**53)


def _seed(key: bytes, context: Sequence[int]) -> bytes:
    msg = np.asarray(context, dtype="<i8").tobytes()
    return hmac.new(key, _DOMAIN + msg, hashlib.sha256).digest()


def keyed_uniforms(key: bytes, context: Sequence[int], n: int) -> np.ndarray:
    """First ``n`` keyed uniforms in (0, 1) for this context (SHAKE-256 XOF)."""
    raw = hashlib.shake_256(_seed(key, context)).digest(8 * n)
    ints = np.frombuffer(raw, dtype="<u8") >> np.uint64(11)  # 53 bits
    return (ints.astype(np.float64) + 0.5) / _U53


@dataclass
class GenerationState:
    """Per-response state. Make a new one for every generated response."""

    seen_contexts: set = field(default_factory=set)
    surprisal_bits: float = 0.0
    active: bool = False
    # How the last token was chosen: "keyed" (watermarked), "warmup" (entropy gate),
    # "repeat" (context seen before) or "start" (not enough context yet).
    last_mode: str = ""
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(secrets.randbits(128))
    )


@dataclass
class DetectionResult:
    p_value: float
    score: float
    num_scored: int
    num_tokens: int
    is_watermarked: bool

    @property
    def z_score(self) -> float:
        """Normal approximation, for reporting only. The p-value is exact."""
        n = max(self.num_scored, 1)
        return (self.score - n) / np.sqrt(n)


class Watermarker:
    def __init__(
        self,
        key: bytes | str,
        context_width: int = 3,
        entropy_threshold: float = 8.0,
        fpr: float = 1e-5,
    ):
        if isinstance(key, str):
            key = key.encode()
        if len(key) < 16:
            raise ValueError("use a secret key of at least 16 bytes")
        if context_width < 1:
            raise ValueError("context_width must be >= 1")
        self.key = key
        self.context_width = context_width
        self.entropy_threshold = entropy_threshold
        self.fpr = fpr

    # ------------------------------------------------------------ generation
    def new_state(self) -> GenerationState:
        return GenerationState()

    def sample(
        self,
        probs: np.ndarray,
        history: Sequence[int],
        state: GenerationState,
    ) -> int:
        """Choose the next token.

        ``probs``: the model's final next-token distribution (1-D, sums to 1).
        ``history``: all token ids so far (the prompt plus the generated tokens).
        """
        p = np.asarray(probs, dtype=np.float64)
        p = np.clip(p, 0.0, None)
        p = p / p.sum()

        h = self.context_width
        ctx = tuple(int(t) for t in history[-h:]) if len(history) >= h else None

        if ctx is None or not state.active or ctx in state.seen_contexts:
            tok = int(state.rng.choice(len(p), p=p))
            state.last_mode = "start" if ctx is None else ("warmup" if not state.active else "repeat")
        else:
            state.last_mode = "keyed"
            u = keyed_uniforms(self.key, ctx, len(p))
            # argmax u^(1/p) == argmin  log(-log u) - log p   (log space: no overflow)
            with np.errstate(divide="ignore"):
                race = np.log(-np.log(u)) - np.log(p)
            tok = int(np.argmin(race))

        if ctx is not None:
            state.seen_contexts.add(ctx)
        if not state.active:
            state.surprisal_bits += -np.log2(max(p[tok], 1e-300))
            state.active = state.surprisal_bits >= self.entropy_threshold
        return tok

    # ------------------------------------------------------------- detection
    def token_scores(self, token_ids: Sequence[int]) -> list[float | None]:
        """Per-token score ``-log(1 - u)``. ``None`` marks tokens the test skips.

        Under the null hypothesis each score is Exp(1), with mean 1. Watermarked
        tokens average noticeably higher.
        """
        ids = [int(t) for t in token_ids]
        h = self.context_width
        seen: set = set()
        out: list[float | None] = [None] * min(h, len(ids))
        for i in range(h, len(ids)):
            ctx = tuple(ids[i - h : i])
            tok = ids[i]
            if ctx in seen or tok < 0:
                out.append(None)
                continue
            seen.add(ctx)
            u = keyed_uniforms(self.key, ctx, tok + 1)[tok]
            out.append(float(-np.log1p(-u)))
        return out

    def detect(self, token_ids: Sequence[int]) -> DetectionResult:
        """Test whether ``token_ids`` carry this key's watermark."""
        ids = list(token_ids)
        scores = [s for s in self.token_scores(ids) if s is not None]
        total, n = float(sum(scores)), len(scores)
        p_value = float(gammaincc(n, total)) if n > 0 else 1.0
        return DetectionResult(
            p_value=p_value,
            score=total,
            num_scored=n,
            num_tokens=len(ids),
            is_watermarked=n > 0 and p_value < self.fpr,
        )