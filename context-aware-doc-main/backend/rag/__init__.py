"""
RAG core package — Python port of the browser-side ContextAwareDocQABot pipeline.

Pipeline order (same as the original React app):
  1. parse PDF/TXT  →  page segments
  2. chunk          →  overlapping sentence-aware windows
  3. TF-IDF index   →  sparse vectors per chunk + corpus IDF
  4. retrieve       →  cosine similarity, top-7 (+ neighbors)
  5. generate       →  Gemini with citations + confidence
"""

from .chunking import chunk_text, CHUNK_SIZE, OVERLAP
from .cosine import cosine_similarity
from .pdf_parser import parse_pdf_bytes, parse_txt_bytes, ingest_file
from .retrieval import retrieve_relevant_chunks, resolve_retrieval_query
from .tfidf import build_tfidf_index_for_chunks, text_to_tfidf_vector, tokenize
from .gemini_client import answer_question
from .pipeline import DocumentStore

__all__ = [
    "CHUNK_SIZE",
    "OVERLAP",
    "DocumentStore",
    "answer_question",
    "build_tfidf_index_for_chunks",
    "chunk_text",
    "cosine_similarity",
    "ingest_file",
    "parse_pdf_bytes",
    "parse_txt_bytes",
    "resolve_retrieval_query",
    "retrieve_relevant_chunks",
    "text_to_tfidf_vector",
    "tokenize",
]
