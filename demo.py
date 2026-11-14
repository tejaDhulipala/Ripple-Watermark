"""Watermark Ripple-65B output and check that detection/attribution works.

Usage: python3 -I demo.py ["prompt"] [max_new_tokens]
Needs llama-server running on 127.0.0.1:8080 with Ripple loaded.
"""

import os
import secrets
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ripple_fingerprint import ModelRegistry  # noqa: E402
from ripple_fingerprint.llamacpp import LlamaServer, detect_text, generate_ids  # noqa: E402

# Demo key, committed on purpose (public = anyone can detect/forge this watermark).
# Override with RIPPLE_KEY=/path/to/other.key for a private key.
KEY_PATH = os.environ.get("RIPPLE_KEY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys", "demo-master.key"))
LABEL = "ripple-65b-preview"


def master_key() -> bytes:
    if not os.path.exists(KEY_PATH):
        os.makedirs(os.path.dirname(KEY_PATH), mode=0o700, exist_ok=True)
        fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
        print(f"created new master key at {KEY_PATH}")
    return bytes.fromhex(open(KEY_PATH).read().strip())


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a short blog post about why people should learn to cook at home."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 250

    srv = LlamaServer()
    reg = ModelRegistry(master_key())
    wm = reg.register(LABEL, encode=srv.tokenize)
    reg.register("decoy-model", encode=srv.tokenize)  # a second key: must NOT match

    ids = srv.chat_prompt_ids(prompt)
    for marked in (True, False):
        modes = Counter()
        t0 = time.time()
        out = generate_ids(srv, ids, wm, max_new_tokens=n, watermark=marked,
                           on_step=lambda p, h, t, s: modes.update([s.last_mode]))
        dt = time.time() - t0
        text = srv.detokenize(out)
        det_ids = wm.detect(out)
        det_txt = detect_text(srv, wm, text)
        att = reg.attribute(text)
        print("=" * 70)
        print(f"{'WATERMARKED' if marked else 'PLAIN (no watermark)'}: {len(out)} tokens in {dt:.1f}s "
              f"({len(out) / dt:.1f} tok/s), modes={dict(modes)}")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print(f"-- detect(generated ids): p={det_ids.p_value:.3g} z={det_ids.z_score:.1f} "
              f"scored={det_ids.num_scored} -> {det_ids.is_watermarked}")
        print(f"-- detect(re-tokenized text): p={det_txt.p_value:.3g} z={det_txt.z_score:.1f} -> {det_txt.is_watermarked}")
        print(f"-- attribution: best={att.best} matches={att.matches} "
              + ", ".join(f"{m.name}: adj_p={m.adjusted_p:.3g}" for m in att.models))


if __name__ == "__main__":
    main()
