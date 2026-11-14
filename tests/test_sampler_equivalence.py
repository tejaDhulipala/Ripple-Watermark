"""Old full-vocab race vs new support-only race: must pick the identical token."""
import sys
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ripple_fingerprint.core import Watermarker, keyed_uniforms

wm = Watermarker(b'k' * 32)
rng = np.random.default_rng(0)
V = 100352
mismatch = 0
for trial in range(2000):
    p = np.zeros(V)
    k = int(rng.integers(1, 1500))
    idx = rng.choice(V, size=k, replace=False)
    p[idx] = rng.dirichlet(np.full(k, rng.choice([0.05, 0.5, 5.0])))
    ctx = tuple(int(x) for x in rng.integers(0, V, size=3))
    # original implementation
    u = keyed_uniforms(wm.key, ctx, len(p))
    with np.errstate(divide='ignore'):
        old = int(np.argmin(np.log(-np.log(u)) - np.log(p)))
    # new implementation, via the real sample() (state forced into keyed mode)
    st = wm.new_state(); st.active = True
    new = wm.sample(p, list(ctx), st)
    assert st.last_mode == 'keyed'
    mismatch += old != new
print(f"2000 trials, mismatches: {mismatch}")
