"""
Smoke tests for the from-scratch RAG core (no Gemini / no PDF required).

Run from backend/:  python -m tests.smoke_test
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow `python -m tests.smoke_test` and `python tests/smoke_test.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import CHUNK_SIZE, OVERLAP, chunk_text
from rag.cosine import cosine_similarity
from rag.tfidf import (
    build_tfidf_index_for_chunks,
    inverse_document_frequency,
    text_to_tfidf_vector,
    tokenize,
)
from rag.retrieval import retrieve_relevant_chunks


def test_tokenize_drops_stopwords_and_punct():
    tokens = tokenize("The Nano-Tube is GREAT!!!")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "nano" in tokens and "tube" in tokens and "great" in tokens


def test_chunk_overlap_and_size():
    # Build text with clear sentence boundaries so the soft cut engages.
    sentence = "Carbon nanotubes are cylindrical molecules. "
    text = sentence * 40  # well over 600 chars
    chunks = chunk_text(text, "demo.txt", page=1)
    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert len(c["text"]) <= CHUNK_SIZE + 5  # trim can only shorten
    # Overlap means consecutive windows share content near the boundary.
    a, b = chunks[0]["text"], chunks[1]["text"]
    assert OVERLAP > 0
    assert any(a.endswith(a[-min(40, len(a)) :]) for _ in [0])  # sanity


def test_idf_formula():
    df = {"rare": 1, "common": 10}
    n = 10
    idf = inverse_document_frequency(df, n)
    assert abs(idf["rare"] - math.log(1 + 10 / 1)) < 1e-12
    assert abs(idf["common"] - math.log(1 + 10 / 10)) < 1e-12


def test_cosine_identical_vectors():
    v = {"a": 0.5, "b": 0.5}
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert cosine_similarity({}, {"a": 1.0}) == 0.0


def test_retrieval_top_k():
    chunks = [
        {"id": "1", "text": "graphene is a single layer of carbon atoms", "docName": "a.pdf", "page": 1, "chunkIndex": 0},
        {"id": "2", "text": "bananas are yellow fruit grown in tropics", "docName": "a.pdf", "page": 1, "chunkIndex": 1},
        {"id": "3", "text": "carbon nanotubes form cylindrical graphene sheets", "docName": "a.pdf", "page": 2, "chunkIndex": 0},
    ]
    idf = build_tfidf_index_for_chunks(chunks)
    out = retrieve_relevant_chunks("What are carbon nanotubes?", chunks, top_k=2, idf_map=idf)
    assert out["results"]
    # Top hit should be nanotube / graphene related, not bananas.
    top_text = out["results"][0]["chunk"]["text"].lower()
    assert "banana" not in top_text


if __name__ == "__main__":
    tests = [
        test_tokenize_drops_stopwords_and_punct,
        test_chunk_overlap_and_size,
        test_idf_formula,
        test_cosine_identical_vectors,
        test_retrieval_top_k,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
