"""
Cosine similarity — from scratch, matching the JS implementation.

Formula:
                 a · b
  cos(a, b) = -----------
              ||a|| ||b||

Why cosine (not Euclidean / dot alone):
  TF-IDF vectors have different lengths (chunk size, query length). Cosine
  measures *angle* — how aligned the term-weight directions are — so a long
  chunk and a short query can still score highly if they share the same
  distinctive terms.

Implementation notes that match JS exactly:
  - Sparse dicts: only iterate keys of vec_a for the dot product (terms absent
    from either vector contribute 0 anyway).
  - Zero-magnitude vector → similarity 0 (avoid ZeroDivisionError).
  - Clamp result into [0, 1]. With non-negative TF-IDF weights, cos is already
    in [0, 1]; the clamp is defensive parity with Math.min/Math.max in JS.
"""

from __future__ import annotations

import math


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Return clamped cosine similarity of two sparse TF-IDF vectors."""
    # Dot product: sum a[t]*b[t] over shared terms only.
    dot = 0.0
    for term, wa in vec_a.items():
        wb = vec_b.get(term)
        if wb is not None:
            dot += wa * wb

    # Euclidean magnitudes.
    mag_a = math.sqrt(sum(w * w for w in vec_a.values()))
    mag_b = math.sqrt(sum(w * w for w in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    raw = dot / (mag_a * mag_b)
    return min(1.0, max(0.0, raw))
