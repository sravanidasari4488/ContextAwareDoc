"""
Retrieval — top-7 TF-IDF cosine search with the same expansions as the JS app.

Pipeline for one query:
  1. (optional) rewrite follow-ups using topics from the last assistant reply
  2. normalize whitespace / case
  3. fuzzy vocab expansion (4-char prefix) for typos / morphology
  4. domain keyword expansion ("types", "nanotube", …)
  5. vectorize query with corpus IDF
  6. score every chunk with cosine similarity
  7. drop scores < MIN_RETRIEVAL_SIMILARITY (0.02)
  8. keep topK=7, then expand with ±1 neighbor chunks (capped at 12)
  9. confidence = average of top-3 similarities → High / Medium / Low
"""

from __future__ import annotations

import re
from typing import Any

from .cosine import cosine_similarity
from .tfidf import text_to_tfidf_vector, tokenize

MIN_RETRIEVAL_SIMILARITY = 0.02

FOLLOW_UP_PHRASES = [
    "elaborate",
    "tell me more",
    "explain further",
    "what did you mean",
    "can you explain",
    "more detail",
    "go on",
    "continue",
    "and then",
    "what about",
    "how about",
    "why is that",
    "how so",
]


def confidence_from_top_scores(top_scores: list[float]) -> dict[str, Any]:
    """
    Map retrieval strength to a human label.

    Why average of top-3 (not just #1):
      A single lucky keyword hit can inflate the best score. Averaging the best
      few rewards queries that consistently match several relevant chunks.
    Thresholds (same as JS): >0.3 High, >=0.1 Medium, else Low.
    """
    k = min(3, len(top_scores))
    if k == 0:
        return {
            "avg": 0.0,
            "label": "Low confidence — answer may not be in document",
        }
    avg = sum(top_scores[:k]) / k
    if avg > 0.3:
        return {"avg": avg, "label": "High confidence"}
    if avg >= 0.1:
        return {"avg": avg, "label": "Medium confidence"}
    return {
        "avg": avg,
        "label": "Low confidence — answer may not be in document",
    }


def is_follow_up_question(query: str) -> bool:
    lower = str(query).lower()
    return any(phrase in lower for phrase in FOLLOW_UP_PHRASES)


def get_last_assistant_message(messages: list[dict[str, str]]) -> dict[str, str] | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m
    return None


def extract_retrieval_topics_from_assistant_message(content: str) -> str:
    """
    Follow-ups like "tell me more" share almost no content words with the doc.
    Steal topic tokens from the start of the previous answer so TF-IDF still
    lands near the same passages.
    """
    excerpt = str(content).replace("**", "")[:200].strip()
    tokens = tokenize(excerpt)
    if tokens:
        return " ".join(tokens)
    return excerpt


def resolve_retrieval_query(
    user_query: str,
    chat_history: list[dict[str, str]],
) -> dict[str, Any]:
    if not is_follow_up_question(user_query):
        return {"retrievalQuery": user_query, "isFollowUp": False}
    last = get_last_assistant_message(chat_history)
    if not last:
        return {"retrievalQuery": user_query, "isFollowUp": False}
    topics = extract_retrieval_topics_from_assistant_message(last["content"])
    if not topics.strip():
        return {"retrievalQuery": user_query, "isFollowUp": False}
    return {"retrievalQuery": topics, "isFollowUp": True}


def preprocess_query_for_retrieval(query: str) -> str:
    return re.sub(r"\s+", " ", str(query).lower().strip())


def expand_query_with_fuzzy_vocabulary_tokens(
    query: str,
    vocabulary_idf: dict[str, float],
) -> str:
    """
    Typo / stem bridge via shared 4-character prefixes.

    Example: query has "nanotub" (missing from IDF) → add corpus terms that
    start with "nano" like "nanotube", "nanotubes". This is cheap approximate
    matching without edit-distance libraries.
    """
    tokens = tokenize(query)
    if not tokens:
        return query

    added: set[str] = set()
    for word in tokens:
        if word in vocabulary_idf:
            continue
        if len(word) < 4:
            continue
        prefix = word[:4]
        for term in vocabulary_idf:
            if term != word and term.startswith(prefix):
                added.add(term)
    if not added:
        return query
    return f"{query} {' '.join(added)}"


def expand_query_for_retrieval(query: str) -> str:
    """
    Lightweight intent expansion used in the original demo corpus.

    "types" questions need enumeration vocabulary; nanotube questions benefit
    from common CNT aliases that appear in research PDFs.
    """
    base = str(query)
    lower = base.lower()
    extras: list[str] = []
    if re.search(r"\btypes?\b", lower):
        extras.append("kinds categories list describe definition")
    if re.search(r"nanotube|\bcnts?\b", lower, flags=re.I):
        extras.append(
            "SWCNT MWCNT single-walled multi-walled single walled "
            "multi walled graphene cylindrical"
        )
    return f"{base} {' '.join(extras)}" if extras else base


def expand_retrieved_with_neighbors(
    results: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    max_extra: int = 8,
) -> list[dict[str, Any]]:
    """
    Pull adjacent chunks on the same page (±1 chunkIndex).

    Why: a heading may score high while the definition lives in the next window.
    Neighbor similarity is discounted by 0.92 so true hits still rank above them.
    """
    seen = {r["chunk"]["id"] for r in results}
    extras: list[dict[str, Any]] = []
    for r in results:
        chunk = r["chunk"]
        doc_name, page, chunk_index = chunk["docName"], chunk["page"], chunk["chunkIndex"]
        for delta in (-1, 1):
            if len(extras) >= max_extra:
                break
            sibling = next(
                (
                    c
                    for c in all_chunks
                    if c["docName"] == doc_name
                    and c["page"] == page
                    and c["chunkIndex"] == chunk_index + delta
                ),
                None,
            )
            if sibling and sibling["id"] not in seen:
                seen.add(sibling["id"])
                extras.append(
                    {
                        "chunk": sibling,
                        "similarity": r["similarity"] * 0.92,
                    }
                )
    merged = results + extras
    merged.sort(key=lambda x: x["similarity"], reverse=True)
    return merged


def retrieve_relevant_chunks(
    query: str,
    all_chunks: list[dict[str, Any]],
    top_k: int = 7,
    idf_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Core retrieval entry point — mirrors retrieveRelevantChunks(..., 7, idf).

    Returns:
      {
        "results": [{ "chunk": {...}, "similarity": float }, ...]  # ≤ 12
        "confidence": { "avg": float, "label": str }
      }
    """
    if not idf_map or not all_chunks:
        return {"results": [], "confidence": confidence_from_top_scores([])}

    normalized = preprocess_query_for_retrieval(query)
    fuzzy = expand_query_with_fuzzy_vocabulary_tokens(normalized, idf_map)
    search_query = expand_query_for_retrieval(fuzzy)
    q_vec = text_to_tfidf_vector(search_query, idf_map)

    scored = [
        {
            "chunk": chunk,
            "similarity": cosine_similarity(q_vec, chunk.get("tfidf") or {}),
        }
        for chunk in all_chunks
    ]
    filtered = [s for s in scored if s["similarity"] >= MIN_RETRIEVAL_SIMILARITY]
    filtered.sort(key=lambda x: x["similarity"], reverse=True)

    confidence = confidence_from_top_scores(
        [r["similarity"] for r in filtered[:3]]
    )
    core = filtered[:top_k]
    results = expand_retrieved_with_neighbors(core, all_chunks)[:12]
    return {"results": results, "confidence": confidence}
