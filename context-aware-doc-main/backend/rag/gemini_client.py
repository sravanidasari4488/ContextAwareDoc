"""
Gemini answer generation — Python SDK port of the browser fetch() path.

Behavior preserved from the JS app:
  - Prompt forces answers from excerpts only + "According to Source N…" citations
  - Last 6 chat turns included for conversational context
  - Follow-up instruction injected when isFollowUp=True
  - Models: preferred → fallbacks (flash-lite, then flash)
  - Min interval between calls + retries on rate limits
  - temperature=0.2, maxOutputTokens=1000

Uses the current official SDK: `google-genai` (`from google import genai`).
Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable

from google import genai
from google.genai import types

GEMINI_MODEL_FALLBACKS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
GEMINI_MIN_INTERVAL_S = 5.0
GEMINI_RATE_LIMIT_RETRY_S = 5.0
GEMINI_MAX_RETRIES = 4

_last_call_at = 0.0
_queue_lock = asyncio.Lock()
_client: genai.Client | None = None


def _models_to_try() -> list[str]:
    preferred = os.getenv("GEMINI_MODEL") or os.getenv("VITE_GEMINI_MODEL")
    if preferred:
        rest = [m for m in GEMINI_MODEL_FALLBACKS if m != preferred]
        return [preferred, *rest]
    return list(GEMINI_MODEL_FALLBACKS)


def _get_client() -> genai.Client:
    """
    Lazy-init the GenAI client once.

    Why a Client (new SDK) instead of genai.configure (old SDK):
      google-generativeai is deprecated; google-genai is the supported package.
      Client(api_key=...) matches how the REST key was passed in the browser URL.
    """
    global _client
    if _client is not None:
        return _client
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("VITE_GEMINI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY / VITE_GEMINI_API_KEY)."
        )
    _client = genai.Client(api_key=api_key)
    return _client


def _is_rate_limit(err: BaseException) -> bool:
    msg = str(err).lower()
    return "429" in msg or "resource exhausted" in msg or "rate" in msg


def _is_daily_quota(err: BaseException) -> bool:
    msg = str(err).lower()
    return "quota" in msg and ("day" in msg or "daily" in msg or "free_tier" in msg)


def _is_model_not_found(err: BaseException) -> bool:
    msg = str(err).lower()
    return "404" in msg or "not found" in msg or "is not found" in msg


def build_prompt(
    question: str,
    relevant_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    is_follow_up: bool = False,
) -> str:
    """
    Assemble the grounded prompt.

    Why labeled sources:
      Numbered [Source i | doc | page] blocks give the model stable handles so
      citations in the answer map back to UI source chips. Separators (---)
      reduce cross-chunk bleed in the model's attention.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source {i + 1} | {c['docName']} | Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(relevant_chunks)
    )

    follow_up_instruction = (
        "The user is asking a follow-up question about your previous response. "
        "Use the conversation history to understand context and elaborate on "
        "what was previously discussed.\n\n"
        if is_follow_up
        else ""
    )

    system_prompt = f"""You are a document analysis assistant. Answer questions using ONLY the document excerpts below.

{follow_up_instruction}RULES:
- Read every excerpt carefully. If any excerpt contains facts, definitions, measurements, or lists that answer the question (even partially), you MUST include them in your answer.
- When the question asks for "types", list each type named in the excerpts with its description from the text.
- Cite sources (e.g., "According to Source 2...").
- Say "I couldn't find information about this in the uploaded document(s)." ONLY if no excerpt contains any substantive content related to the question.
- Do not claim the excerpts only "mention" a topic without details if the text actually includes definitions or specifications.
- Never use outside knowledge.

DOCUMENT EXCERPTS:
{context_text}"""

    history_msgs = [
        m
        for m in chat_history
        if m.get("role") in ("user", "assistant")
    ]
    history_block = "\n".join(
        f"{m['role']}: {m['content']}" for m in history_msgs[-6:]
    )
    return (
        f"{system_prompt}\n\nConversation history:\n{history_block}"
        f"\n\nUser question: {question}"
    )


def _generate_once(model_name: str, prompt: str) -> str:
    """Single SDK call — low temperature keeps answers faithful to excerpts."""
    client = _get_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.2,
        ),
    )
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    # Empty / blocked — same user-facing fallback as JS SAFETY/RECITATION handling.
    return (
        "I couldn't produce an answer for that question based on the document "
        "content. Try rephrasing your question."
    )


async def answer_question(
    question: str,
    relevant_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    *,
    is_follow_up: bool = False,
    on_wait: Callable[[str], None] | None = None,
) -> str:
    """
    Async wrapper with the same rate-limit queue semantics as enqueueGeminiRequest.
    """
    global _last_call_at

    prompt = build_prompt(question, relevant_chunks, chat_history, is_follow_up)

    async with _queue_lock:
        elapsed = time.monotonic() - _last_call_at
        if _last_call_at > 0 and elapsed < GEMINI_MIN_INTERVAL_S:
            if on_wait:
                on_wait("Thinking...")
            await asyncio.sleep(GEMINI_MIN_INTERVAL_S - elapsed)

        saw_daily = False
        saw_rate = False

        for model_name in _models_to_try():
            for attempt in range(GEMINI_MAX_RETRIES + 1):
                _last_call_at = time.monotonic()
                try:
                    # SDK call is sync — offload so FastAPI's event loop stays free.
                    return await asyncio.to_thread(_generate_once, model_name, prompt)
                except Exception as err:
                    if _is_model_not_found(err):
                        break
                    if _is_rate_limit(err):
                        if _is_daily_quota(err):
                            saw_daily = True
                            break
                        saw_rate = True
                        if attempt < GEMINI_MAX_RETRIES:
                            if on_wait:
                                on_wait("Just a moment...")
                            await asyncio.sleep(GEMINI_RATE_LIMIT_RETRY_S)
                            continue
                        break
                    break

        if saw_daily and not saw_rate:
            return (
                "Today's free limit for the main Gemini model on this API key is "
                "used up. I tried alternate models — if this persists, wait until "
                "tomorrow (quota resets daily) or create a new key in Google AI Studio."
            )
        if saw_daily or saw_rate:
            return (
                "I'm having a little trouble right now — please try your question "
                "again in a minute."
            )
        return "I couldn't generate an answer just now. Please try again shortly."


def answer_question_sync(
    question: str,
    relevant_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    *,
    is_follow_up: bool = False,
) -> str:
    """Sync helper for Flask (or scripts) — runs the async path to completion."""
    return asyncio.run(
        answer_question(
            question,
            relevant_chunks,
            chat_history,
            is_follow_up=is_follow_up,
        )
    )
