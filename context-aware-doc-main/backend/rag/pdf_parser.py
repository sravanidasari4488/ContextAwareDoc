"""
PDF / TXT parsing — Python replacement for PDF.js in the browser.

Recommendation: pdfplumber (chosen) over pypdf
----------------------------------------------
Why pdfplumber for this project:
  - The original PDF.js path reconstructed lines from glyph Y positions so a
    heading and its definition stayed on adjacent lines (Math.abs(y - lastY) > 4).
  - pdfplumber exposes character-level bounding boxes and a line-aware extract,
    so we can recreate that "new line when Y jumps" behavior.
  - It handles multi-column and table-ish layouts more gracefully than a naive
    string dump.

Why not pypdf for the default:
  - pypdf.PdfReader(...).extract_text() is lighter and fine for simple linear PDFs,
    but it often flattens columns / ignores visual line breaks, which breaks the
    "definition stays with heading" property the chunker relies on.
  - Keep pypdf in mind as a fallback if you need zero extra layout deps.

Output shape matches the JS ingest pipeline:
  [{ "page": 1, "text": "..." }, ...]
"""

from __future__ import annotations

import io
import re
import uuid
from typing import Any

import pdfplumber

from .chunking import chunk_text


def _chars_to_text_like_pdfjs(chars: list[dict[str, Any]]) -> str:
    """
    Rebuild page text the way pdfPageItemsToText did in JS.

    Interview intuition:
      PDFs store glyphs, not paragraphs. Two glyphs on the same visual line share
      roughly the same Y. When Y jumps more than ~4 units, the eye moved to a new
      line — insert '\\n'. Otherwise insert a space between tokens so words don't glue.
    """
    if not chars:
        return ""

    # Sort reading-order: top→bottom (larger Y first in PDF coords), then left→right.
    # pdfplumber uses bottom-left origin; higher 'top' ≈ higher on the page.
    ordered = sorted(
        chars,
        key=lambda c: (-round(float(c.get("top", 0)), 1), float(c.get("x0", 0))),
    )

    text = ""
    last_y: float | None = None
    for ch in ordered:
        s = ch.get("text") or ""
        if not s:
            continue
        y = float(ch.get("top", 0))
        if last_y is not None and abs(y - last_y) > 4:
            # Same threshold as JS: vertical jump ⇒ hard line break.
            text += "\n"
        elif text and not text.endswith("\n") and not text.endswith(" "):
            # Same glyph row: separate tokens with a single space.
            text += " "
        text += s
        last_y = y

    # Collapse runaway blank lines the way JS did: \\n{3,} → \\n\\n
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_pdf_bytes(data: bytes) -> list[dict[str, Any]]:
    """
    Parse a PDF from raw bytes into per-page text segments.

    Why bytes (not a path): FastAPI/Flask receive uploads as in-memory file
    objects — this mirrors file.arrayBuffer() in the browser.
    """
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            chars = page.chars or []
            if chars:
                text = _chars_to_text_like_pdfjs(chars)
            else:
                # Fallback when a page has no char stream (scanned image, etc.)
                text = (page.extract_text() or "").strip()
            pages.append({"page": i, "text": text})
    return pages


def parse_txt_bytes(data: bytes, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """TXT is treated as a single logical page — same as parseTxtFile in JS."""
    full_text = data.decode(encoding, errors="replace")
    return [{"page": 1, "text": full_text}]


def ingest_file(
    filename: str,
    data: bytes,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    Full Step 1+2 of ingest for one file: parse → chunk → attach endChar.

    Returns a Document dict compatible with the JS builtDocs shape:
      { id, name, chunks: [{ id, text, docName, page, chunkIndex, startChar, endChar }] }
    """
    is_pdf = (content_type == "application/pdf") or filename.lower().endswith(".pdf")
    page_segments = parse_pdf_bytes(data) if is_pdf else parse_txt_bytes(data)

    all_raw_chunks: list[dict[str, Any]] = []
    for seg in page_segments:
        all_raw_chunks.extend(chunk_text(seg["text"], filename, seg["page"]))

    chunks = [
        {**c, "endChar": c["startChar"] + len(c["text"])}
        for c in all_raw_chunks
    ]

    return {
        "id": str(uuid.uuid4()),
        "name": filename,
        "chunks": chunks,
        "page_count": len(page_segments),
    }
