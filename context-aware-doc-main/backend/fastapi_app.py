"""
FastAPI wrapper around the shared DocumentStore.

Why FastAPI shines here:
  - Native async → Gemini waits don't block the event loop
  - Automatic OpenAPI docs at /docs (great for demos / interviews)
  - UploadFile + Pydantic models give typed request/response contracts

Run:
  uvicorn fastapi_app:app --reload --app-dir backend
  # or from backend/:  uvicorn fastapi_app:app --reload
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.pipeline import DocumentStore

load_dotenv()

app = FastAPI(
    title="Context-Aware Document Q&A",
    description="Python port of the TF-IDF + Gemini RAG bot",
    version="1.0.0",
)

# Browser React app on Vite (:5173) needs CORS to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store — one process = one corpus (swap for Redis/DB in production).
store = DocumentStore()


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=7, ge=1, le=20)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return store.stats()


@app.delete("/documents")
def clear_documents() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """
    Upload one or more PDF/TXT files → parse → chunk → rebuild TF-IDF index.
    Mirrors ingestFilesToBuiltDocs in the React app.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    ingested = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        meta = store.add_file(f.filename or "upload.bin", data, f.content_type)
        ingested.append(meta)

    if not ingested:
        raise HTTPException(status_code=400, detail="All uploads were empty")

    return {"ingested": ingested, "stats": store.stats()}


@app.post("/ask")
async def ask(body: AskRequest) -> dict[str, Any]:
    """Retrieve top-k chunks and generate a cited Gemini answer."""
    if not store.flat_chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. POST /ingest first.",
        )
    history = [m.model_dump() for m in body.chat_history]
    return await store.ask(body.question, history, top_k=body.top_k)


@app.post("/retrieve")
async def retrieve_only(body: AskRequest) -> dict[str, Any]:
    """Debug endpoint: similarity search without calling Gemini."""
    if not store.flat_chunks:
        raise HTTPException(status_code=400, detail="No documents indexed.")
    result = store.retrieve(body.question, top_k=body.top_k)
    # Drop heavy tfidf payloads from the response.
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
    return {"results": slim, "confidence": result["confidence"]}
