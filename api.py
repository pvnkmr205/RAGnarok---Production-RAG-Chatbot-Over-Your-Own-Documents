"""
Checkpoint 7 - FastAPI wrapper, caching, and token cost tracking

Wraps the same retrieve() / build_prompt() functions already validated by
evaluate_retrieval.py and test_refusal.py in an actual HTTP API -- nothing
about the underlying pipeline changes here, this is purely a delivery layer
on top of it.

Run with:
    uvicorn api:app --reload

Then open http://localhost:8000/docs for FastAPI's auto-generated,
interactive API docs -- try requests straight from the browser.
"""

import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from Query import retrieve, build_prompt, extract_citations, client, chunks, format_source_label

app = FastAPI(
    title="RAGnarok API",
    description="Ask questions about whatever document set was last ingested.",
)

# In-memory cache: question (normalized) -> full response dict.
# Resets whenever the server restarts -- fine for a demo/resume project,
# a real deployment would use Redis or similar for persistence.
_cache: dict[str, dict] = {}


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool
    citation_warning: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/corpus")
def corpus_info():
    """What document set is currently active -- same info query.py prints on startup."""
    unique_sources = sorted(set(c["source"] for c in chunks))
    return {
        "num_source_files": len(unique_sources),
        "num_chunks": len(chunks),
        "source_files": unique_sources[:20],
        "truncated": len(unique_sources) > 20,
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    start = time.time()
    cache_key = req.question.strip().lower()

    if cache_key in _cache:
        cached = _cache[cache_key]
        return AskResponse(**cached, cached=True, latency_seconds=round(time.time() - start, 4))

    retrieved = retrieve(req.question)
    prompt = build_prompt(req.question, retrieved)

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    answer = response.choices[0].message.content
    usage = response.usage

    citation_check = extract_citations(answer, len(retrieved))
    citation_warning = None
    if citation_check["invalid"]:
        citation_warning = f"Cites nonexistent source(s): {citation_check['invalid']}"
    elif not citation_check["is_refusal"] and not citation_check["has_citation"]:
        citation_warning = "Answer has no citations and isn't a refusal"

    result = {
        "answer": answer,
        "sources": [format_source_label(c) for c in retrieved],
        "citation_warning": citation_warning,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    _cache[cache_key] = result

    return AskResponse(**result, cached=False, latency_seconds=round(time.time() - start, 4))