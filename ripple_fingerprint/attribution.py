"""Which model wrote this? Per-model keys and attribution.

Every model gets its own watermark key, derived from one master secret and a
stable model label::

    key_A = HMAC-SHA256(master, "ripple-fingerprint/model-key/v1" || label_A)

Under any key other than the one used to sample, the keyed uniforms are
independent of the text, so the detector's score is exactly Gamma(n, 1). Text
from model A therefore looks like ordinary text to model B's detector, and it
can't produce a false attribution beyond the usual false-positive rate.

``ModelRegistry.attribute(text)`` re-tokenizes the text with each registered
model's own tokenizer and runs two tests per model:

* **Whole text.** The usual detector, Bonferroni-corrected across the ``K``
  models and the two tests: ``adjusted_p = min(1, 2K * p)``.
* **Spans.** Stretches of text carrying one model's watermark, so a document
  stitched together from several models (and human writing) can be split by
  source. Each span's p-value is corrected for every possible segment of
  every model (``2K * m(m+1)/2`` tests for ``m`` scored tokens).

A model matches if either test is below ``fpr``. The chance that *any* model
is wrongly named for text it didn't write stays below ``fpr``. A span needs
more evidence than the whole text, but it catches a short passage diluted by
a long document.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.special import gammaincc

from .core import DetectionResult, Watermarker

_KEY_DOMAIN = b"ripple-fingerprint/model-key/v1\x00"


def derive_key(master_key: bytes | str, label: str) -> bytes:
    """The watermark key for one model. Same master + label always gives the same key."""
    if isinstance(master_key, str):
        master_key = master_key.encode()
    if len(master_key) < 16:
        raise ValueError("use a master key of at least 16 bytes")
    if not label:
        raise ValueError("model label must be non-empty")
    return hmac.new(master_key, _KEY_DOMAIN + label.encode(), hashlib.sha256).digest()


# ------------------------------------------------------------------ results
@dataclass
class ModelScore:
    name: str
    result: DetectionResult
    adjusted_p: float  # whole-text p, Bonferroni-corrected (models x 2 tests)
    is_match: bool  # whole-text test or a significant span
    token_ids: list[int] = field(repr=False, default_factory=list)
    token_scores: list[float | None] = field(repr=False, default_factory=list)
    offsets: list[tuple[int, int]] | None = field(repr=False, default=None)


@dataclass
class Span:
    model: str
    token_start: int  # in that model's tokenization, end exclusive
    token_end: int
    char_start: int | None  # in the input text, end exclusive; None without offsets
    char_end: int | None
    num_scored: int
    p_value: float  # corrected for every segment of every model (and both tests)


@dataclass
class AttributionResult:
    models: list[ModelScore]  # most significant first
    spans: list[Span]  # in text order
    fpr: float

    @property
    def matches(self) -> list[str]:
        """Every model whose watermark is present (more than one for mixed text)."""
        return [m.name for m in self.models if m.is_match]

    @property
    def best(self) -> str | None:
        """The most likely source, or ``None`` if no model's watermark is present."""
        return self.models[0].name if self.models and self.models[0].is_match else None

    @property
    def is_attributed(self) -> bool:
        return self.best is not None

    def __getitem__(self, name: str) -> ModelScore:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


# ----------------------------------------------------------------- registry
@dataclass
class _Entry:
    name: str
    watermarker: Watermarker
    tokenizer: object | None
    encode: Callable[[str], Sequence[int]] | None


class ModelRegistry:
    """Per-model watermark keys, plus "which model wrote this?".

    ``master_key`` is the only secret. Register each model under a **stable
    label**: the label decides the key, so renaming a model (or registering a
    new checkpoint path under a new label) changes its key. Pass ``key_id`` to
    keep the key while the display name or path changes.
    """

    def __init__(self, master_key: bytes | str, fpr: float = 1e-5, **watermarker_kwargs):
        derive_key(master_key, "check")  # validate early
        self.master_key = master_key.encode() if isinstance(master_key, str) else master_key
        self.fpr = fpr
        self.watermarker_kwargs = watermarker_kwargs
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        name: str,
        tokenizer=None,
        *,
        encode: Callable[[str], Sequence[int]] | None = None,
        key_id: str | None = None,
    ) -> Watermarker:
        """Add a model and return its ``Watermarker`` (use it for that model's sampling).

        ``tokenizer``: a Hugging Face (or mlx-lm) tokenizer, needed to attribute
        raw text. With a fast tokenizer, spans also come back as character ranges.
        ``encode``: alternatively, any ``text -> token ids`` function.
        """
        if name in self._entries:
            raise ValueError(f"model {name!r} is already registered")
        wm = Watermarker(derive_key(self.master_key, key_id or name), fpr=self.fpr, **self.watermarker_kwargs)
        self._entries[name] = _Entry(name, wm, tokenizer, encode)
        return wm

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def watermarker(self, name: str) -> Watermarker:
        return self._entries[name].watermarker

    # -------------------------------------------------------------- attribution
    def attribute(self, text: str, models: Sequence[str] | None = None) -> AttributionResult:
        """Test ``text`` against every registered model (each with its own tokenizer)."""
        entries = self._select(models)
        tokenized = {}
        for e in entries:
            if e.tokenizer is None and e.encode is None:
                raise ValueError(f"model {e.name!r} has no tokenizer; register one or use attribute_ids")
            tokenized[e.name] = _encode(e, text)
        return self._attribute(entries, tokenized)

    def attribute_ids(self, token_ids: Sequence[int], models: Sequence[str] | None = None) -> AttributionResult:
        """Like ``attribute``, for models that all share one tokenizer and already-tokenized input."""
        ids = [int(t) for t in token_ids]
        entries = self._select(models)
        return self._attribute(entries, {e.name: (ids, None) for e in entries})

    def _select(self, models):
        if not self._entries:
            raise ValueError("no models registered")
        names = self.names if models is None else list(models)
        for n in names:
            if n not in self._entries:
                raise KeyError(f"unknown model {n!r}")
        return [self._entries[n] for n in names]

    def _attribute(self, entries, tokenized) -> AttributionResult:
        k = 2 * len(entries)  # two tests per model: whole text and spans
        scores_out: list[ModelScore] = []
        spans: list[Span] = []
        for e in entries:
            ids, offsets = tokenized[e.name]
            wm = e.watermarker
            tok_scores = wm.token_scores(ids)
            scored = [s for s in tok_scores if s is not None]
            total, n = float(sum(scored)), len(scored)
            p = float(gammaincc(n, total)) if n > 0 else 1.0
            res = DetectionResult(p, total, n, len(ids), n > 0 and p < wm.fpr)
            adj = min(1.0, p * k)
            found = _segments(tok_scores, k, self.fpr)
            is_match = (n > 0 and adj < self.fpr) or bool(found)
            scores_out.append(ModelScore(e.name, res, adj, is_match, ids, tok_scores, offsets))
            for i, j, m, seg_p in found:
                lo = max(0, i - wm.context_width)  # the context tokens belong to the span too
                cs = ce = None
                if offsets:
                    cs, ce = offsets[lo][0], offsets[j - 1][1]
                spans.append(Span(e.name, lo, j, cs, ce, m, seg_p))
        best_span = {}
        for sp in spans:
            best_span[sp.model] = min(sp.p_value, best_span.get(sp.model, 1.0))
        scores_out.sort(key=lambda s: (not s.is_match, min(s.adjusted_p, best_span.get(s.name, 1.0)), -s.result.z_score))
        spans.sort(key=lambda s: (s.char_start if s.char_start is not None else s.token_start, s.model))
        return AttributionResult(scores_out, spans, self.fpr)


# ------------------------------------------------------------------ helpers
def _encode(e: _Entry, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    if e.encode is not None:
        return [int(t) for t in e.encode(text)], None
    tok = e.tokenizer
    if type(tok).__name__ == "TokenizerWrapper":  # mlx-lm wraps a HF tokenizer
        tok = tok._tokenizer
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except (TypeError, NotImplementedError, ValueError, KeyError):
        pass  # slow tokenizer: no offsets
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
    return [int(t) for t in ids], None


def _segments(
    tok_scores: Sequence[float | None],
    num_tests: int,
    fpr: float,
    drift: float = 1.15,
    min_gain: float = 3.0,
) -> list[tuple[int, int, int, float]]:
    """Disjoint stretches where this key's evidence piles up: ``(start, end, n_scored, p)``.

    Repeatedly takes the maximum-sum run of ``score - drift`` (unwatermarked
    tokens average 1, so they pull the sum down; watermarked ones average
    more), then searches the pieces left and right of it. Only runs whose
    corrected p-value is below ``fpr`` are kept.
    """
    x = np.array([0.0 if s is None else s - drift for s in tok_scores])
    raw = np.array([0.0 if s is None else s for s in tok_scores])
    counted = np.array([s is not None for s in tok_scores], dtype=np.int64)
    m = int(counted.sum())
    if m == 0:
        return []
    n_tests = num_tests * m * (m + 1) / 2
    cum_raw = np.concatenate([[0.0], np.cumsum(raw)])
    cum_cnt = np.concatenate([[0], np.cumsum(counted)])
    out = []
    stack = [(0, len(x))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 1:
            continue
        i, j, gain = _max_subarray(x, lo, hi)
        if gain < min_gain:
            continue
        # Sharpen the edges. For a run whose scores average mu, the likelihood
        # ratio Exp(mean mu) vs Exp(1) favours the watermark exactly when
        # score > log(mu) / (1 - 1/mu). Re-cut the run with that break-even
        # point, so a strong watermark doesn't drag in neighbouring text.
        for _ in range(3):
            n_seg = int(cum_cnt[j] - cum_cnt[i])
            mu = (cum_raw[j] - cum_raw[i]) / max(n_seg, 1)
            if mu <= drift + 1e-9:
                break
            cut = max(drift, float(np.log(mu) / (1 - 1 / mu)))
            xs = np.where(counted[i:j] > 0, raw[i:j] - cut, 0.0)
            di, dj, _ = _max_subarray(xs, 0, j - i)
            if (di, dj) == (0, j - i):
                break
            i, j = i + di, i + dj
        n_seg = int(cum_cnt[j] - cum_cnt[i])
        p = min(1.0, float(gammaincc(n_seg, cum_raw[j] - cum_raw[i])) * n_tests)
        if p < fpr:
            out.append((i, j, n_seg, p))
        stack.append((lo, i))
        stack.append((j, hi))
    return sorted(out)


def _max_subarray(x: np.ndarray, lo: int, hi: int) -> tuple[int, int, float]:
    """Kadane on ``x[lo:hi]``: returns ``(start, end_exclusive, sum)``."""
    best, best_i, best_j = -np.inf, lo, lo
    cur, cur_i = 0.0, lo
    for t in range(lo, hi):
        if cur <= 0:
            cur, cur_i = 0.0, t
        cur += x[t]
        if cur > best:
            best, best_i, best_j = cur, cur_i, t + 1
    return best_i, best_j, float(best)