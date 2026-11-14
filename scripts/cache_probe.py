"""How much of the prompt does llama-server recompute per request?"""
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ripple_fingerprint.llamacpp import LlamaServer

srv = LlamaServer()
base = srv.chat_prompt_ids("Write a haiku about rain.")


def ask(ids, **extra):
    out = srv._post('/completion', {"prompt": ids, "n_predict": 1, "n_probs": 5, "cache_prompt": True,
                                    "top_k": 1, **extra})
    t = out["timings"]
    picked = out["completion_probabilities"][0]["id"]
    return t.get("cache_n"), t["prompt_n"], round(t["prompt_ms"], 1), picked


print("len(base) =", len(base))
print("A same prompt, 1st:        cache_n, prompt_n, ms, picked =", ask(base))
c, n, ms, picked = ask(base)
print("B same prompt again:       ", (c, n, ms, picked))
print("C + server's own token:    ", ask(base + [picked]))
print("D + a DIFFERENT token:     ", ask(base + [picked, 13]))
print("E + 1 more token:          ", ask(base + [picked, 13, 14]))
print("F + 1 more, n_predict=0:   ", ask(base + [picked, 13, 14, 15], n_predict=0))
print("G + 1 more after F:        ", ask(base + [picked, 13, 14, 15, 16]))
