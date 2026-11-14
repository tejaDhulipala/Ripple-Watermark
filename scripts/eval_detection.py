"""Generate story + code with and without the watermark, then detect and attribute.

Needs scripts/start_server.sh running. Uses keys/demo-master.key (or $RIPPLE_KEY).
"""
import sys, time, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ripple_fingerprint import ModelRegistry
from ripple_fingerprint.llamacpp import LlamaServer, generate_ids, detect_text
key = bytes.fromhex(open(os.environ.get('RIPPLE_KEY', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'keys', 'demo-master.key'))).read().strip())
srv = LlamaServer(); reg = ModelRegistry(key)
wm = reg.register('ripple-65b-preview', encode=srv.tokenize); reg.register('decoy-model', encode=srv.tokenize)
PROMPTS = {"story": "Write an imaginative short story (about 300 words) about a lighthouse keeper who finds a strange object on the beach.",
           "code": "Write a Python module implementing an LRU cache class with get/put, docstrings, and a few unit tests."}
for name, prompt in PROMPTS.items():
  ids = srv.chat_prompt_ids(prompt)
  for marked in (True, False):
      ent = []
      t0 = time.time()
      out = generate_ids(srv, ids, wm, max_new_tokens=400, temperature=1.0, top_p=0.95, watermark=marked,
          on_step=lambda p, h, t, s: ent.append(-(p[p > 0] * np.log2(p[p > 0])).sum()))
      dt = time.time() - t0
      text = srv.detokenize(out)
      full = list(ids) + out
      sc = wm.token_scores(full)[len(ids):]
      d1 = wm.detect(full[len(ids) - 3:])  # include the 3 context tokens before the reply
      d2 = detect_text(srv, wm, text)
      att = reg.attribute(text)
      s = [x for x in sc if x is not None]
      print('=' * 60)
      print(name.upper(), 'WATERMARKED' if marked else 'PLAIN', f'{len(out)} tok, {len(out)/dt:.1f} tok/s, mean entropy {np.mean(ent):.2f} bits, mean token score {np.mean(s):.3f} (null=1.0)')
      print(text[:300].replace('\n', ' '), '...')
      print(f'detect(ids): p={d1.p_value:.3g} z={d1.z_score:.1f} -> {d1.is_watermarked} | detect(text): p={d2.p_value:.3g} z={d2.z_score:.1f} -> {d2.is_watermarked}')
      print('attribution:', att.best, att.matches, [(m.name, f'{m.adjusted_p:.3g}') for m in att.models], f'spans={len(att.spans)}')
