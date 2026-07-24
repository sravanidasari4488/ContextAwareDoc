"""
Flask wrapper around the same DocumentStore as fastapi_app.py.

Why Flask is still a solid choice:
  - Smaller conceptual surface — easy to explain line-by-line in an interview
  - Sync request handlers map cleanly onto answer_question_sync
  - Familiar to many teams already running Python microservices

Tradeoff vs FastAPI:
  - No built-in async / OpenAPI schema generation
  - Manual JSON validation (or add marshmallow/pydantic yourself)

Run:
  flask --app flask_app run --debug
  # or:  python flask_app.py
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from rag.pipeline import DocumentStore

load_dotenv()

app = Flask(__name__)
CORS(app)

store = DocumentStore()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/stats")
def stats():
    return jsonify(store.stats())


@app.delete("/documents")
def clear_documents():
    store.clear()
    return jsonify({"status": "cleared"})


@app.post("/ingest")
def ingest():
    """
    multipart/form-data with one or more files under the field name `files`.
    Example:  curl -F files=@paper.pdf http://localhost:5000/ingest
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"detail": "No files uploaded"}), 400

    ingested = []
    for f in files:
        data = f.read()
        if not data:
            continue
        meta = store.add_file(f.filename or "upload.bin", data, f.mimetype)
        ingested.append(meta)

    if not ingested:
        return jsonify({"detail": "All uploads were empty"}), 400

    return jsonify({"ingested": ingested, "stats": store.stats()})


@app.post("/ask")
def ask():
    body: dict[str, Any] = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"detail": "question is required"}), 400
    if not store.flat_chunks:
        return jsonify({"detail": "No documents indexed. POST /ingest first."}), 400

    history = body.get("chat_history") or []
    top_k = int(body.get("top_k") or 7)
    result = store.ask_sync(question, history, top_k=top_k)
    return jsonify(result)


@app.post("/retrieve")
def retrieve_only():
    body: dict[str, Any] = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"detail": "question is required"}), 400
    if not store.flat_chunks:
        return jsonify({"detail": "No documents indexed."}), 400

    top_k = int(body.get("top_k") or 7)
    result = store.retrieve(question, top_k=top_k)
    slim = [
        {
            "similarity": r["similarity"],
            "docName": r["chunk"]["docName"],
            "page": r["chunk"]["page"],
            "chunkIndex": r["chunk"]["chunkIndex"],
            "text": r["chunk"]["text"],
        }
        for r in result["results"]
    ]
    return jsonify({"results": slim, "confidence": result["confidence"]})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
