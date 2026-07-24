"""
Sentence-aware overlapping chunking — exact port of JS chunkText().

Constants (must match the original for identical retrieval behavior):
  CHUNK_SIZE = 600   # target characters per chunk
  OVERLAP    = 120   # characters shared with the next chunk

Why overlapping windows:
  A fact that sits on a chunk boundary would otherwise be split across two
  vectors and never match a query that needs both halves. Overlap of 120
  (~20% of 600) keeps boundary context in both neighbors.

Why sentence-aware (not blind every-600-chars):
  Cutting mid-sentence destroys TF for important terms and sends Gemini
  incomplete clauses. We search backward for the last '.!?' in a window so
  chunks usually end on a sentence boundary.
"""

from __future__ import annotations

import math
import re
import uuid
from typing import Any

CHUNK_SIZE = 600
OVERLAP = 120

# Same pattern as JS: /[.!?](?:\\s|$)/g — punctuation that ends a sentence,
# only counted when followed by whitespace or end-of-string.
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")


def chunk_text(text: str, doc_name: str, page: int) -> list[dict[str, Any]]:
    """
    Split `text` into overlapping, sentence-preferring chunks.

    Algorithm walkthrough (interview-ready):
      1. Place a window [start, start+600].
      2. If we are not at EOF, look for sentence ends inside
         [start + 0.35*600, start+600] = [start+210, end].
         Prefer the *last* such boundary so the chunk is as long as possible
         without crossing mid-sentence at the cut.
      3. Emit the trimmed slice with metadata (doc, page, char offset).
      4. Advance: next_start = end - 120. If that would not move forward,
         force start+1 to avoid an infinite loop on tiny/empty slices.
    """
    chunks: list[dict[str, Any]] = []
    if not text:
        return chunks

    start = 0
    chunk_index = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)

        # Only search for a softer cut when more text remains after this window.
        if end < n:
            # Don't snap to a sentence in the first 35% — that would make
            # tiny chunks and waste the 600-char budget.
            search_from = max(start + math.floor(CHUNK_SIZE * 0.35), start)
            best = -1
            for m in _SENTENCE_END_RE.finditer(text, start, end + 1):
                # boundaryEnd = index of the punctuation + 1 (include the .!?)
                boundary_end = m.start() + 1
                if boundary_end <= search_from:
                    continue
                if boundary_end > end:
                    break
                best = boundary_end  # keep last valid end in the window
            if best != -1:
                end = best

        slice_ = text[start:end].strip()
        if not slice_:
            # Whitespace-only window — nudge forward instead of emitting empty junk.
            if end >= n:
                break
            start = max(start + 1, end)
            continue

        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "text": slice_,
                "docName": doc_name,
                "page": page,
                "chunkIndex": chunk_index,
                "startChar": start,
            }
        )
        chunk_index += 1

        if end >= n:
            break

        next_start = end - OVERLAP
        if next_start <= start:
            # Degenerate case: end barely moved past start (e.g. tiny remaining text).
            next_start = start + 1
        start = next_start

    return chunks
