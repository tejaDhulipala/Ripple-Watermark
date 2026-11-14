"""ripple_fingerprint: keyed, distortion-free watermarking (core + attribution from /workspace, plus a llama.cpp backend)."""
from .core import Watermarker, DetectionResult, GenerationState, keyed_uniforms
from .attribution import ModelRegistry, derive_key
