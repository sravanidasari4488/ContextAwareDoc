"""
From-scratch TF-IDF — intentional port of the JS math (NOT sklearn).

Formulas used (identical to ContextAwareDocQABot.jsx):

  Tokenize(text):
      lower → replace non [a-z0-9] with space → split → drop empties & stopwords

  TF(t, d)  = count(t, d) / |d|
      Raw relative frequency. Dividing by |d| stops long chunks from dominating
      just because they contain more tokens.

  DF(t)     = |{ chunks that contain t at least once }|
      Set-based: a term counted once per chunk even if it repeats inside it.
      That is the classic document-frequency definition.

  IDF(t)    = log(1 + N / max(df(t), 1))
      Smoothed IDF. The leading +1 inside the log avoids log(0) pathologies and
      softens extreme rarity. max(df,1) guards empty DF. N = number of chunks.
      NOTE: this is NOT sklearn's default idf = log((1+N)/(1+df)) + 1.

  w(t, d)   = TF(t, d) * IDF(t)
      Sparse dict {term: weight}. Query terms absent from corpus IDF are skipped
      — you cannot weight a term the index never saw.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .stopwords import STOPWORDS

# Same as JS: /[^a-z0-9]+/g → space
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Turn free text into comparable tokens.

    Why this exact pipeline:
      Case folding + stripping punctuation puts "Nano-Tube" and "nanotube" nearer
      (hyphen becomes a split, which is what the JS regex does). Stopword removal
      is applied here so every downstream count (TF, DF) already excludes noise.
    """
    lowered = str(text).lower()
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return [w for w in cleaned.split() if w and w not in STOPWORDS]


def term_frequency_map(tokens: list[str]) -> dict[str, int]:
    """Raw term counts inside one document/chunk."""
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


def document_frequency(chunks: list[dict[str, Any]]) -> dict[str, int]:
    """
    DF across the corpus.

    Critical detail for interviews: we use a *set* of tokens per chunk.
    Repeating "graphene" 10× in one chunk still contributes only +1 to DF.
    IDF measures "how many documents know this word", not "how often it appears".
    """
    df: dict[str, int] = {}
    for chunk in chunks:
        seen = set(tokenize(chunk["text"]))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    return df


def inverse_document_frequency(df: dict[str, int], n: int) -> dict[str, float]:
    """
    IDF(t) = log(1 + N / max(df, 1))

    Rare terms → large IDF → high weight when they do appear.
    Ubiquitous terms → small IDF → down-weighted even if TF is high.
    """
    idf: dict[str, float] = {}
    for term, d in df.items():
        idf[term] = math.log(1.0 + n / max(d, 1))
    return idf


def text_to_tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """
    Build a sparse TF-IDF vector for one string (chunk or query).

    Query and document vectors MUST share the same IDF table — that is what
    makes cosine similarity compare weights in the same feature space.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}
    tf = term_frequency_map(tokens)
    total = len(tokens)
    vec: dict[str, float] = {}
    for term, cnt in tf.items():
        if term not in idf:
            # OOV relative to the indexed corpus — skip (same as `idf[term] == null` in JS).
            continue
        vec[term] = (cnt / total) * idf[term]
    return vec


def build_tfidf_index_for_chunks(all_chunks: list[dict[str, Any]]) -> dict[str, float]:
    """
    Fit corpus IDF and attach `.tfidf` sparse vectors onto each chunk (in place).

    Returns the IDF map so queries can be vectorized with the same weights later.
    Mutating chunks mirrors the JS version, which wrote `chunk.tfidf = ...`.
    """
    n = len(all_chunks)
    if n == 0:
        return {}
    df = document_frequency(all_chunks)
    idf = inverse_document_frequency(df, n)
    for chunk in all_chunks:
        chunk["tfidf"] = text_to_tfidf_vector(chunk["text"], idf)
    return idf
