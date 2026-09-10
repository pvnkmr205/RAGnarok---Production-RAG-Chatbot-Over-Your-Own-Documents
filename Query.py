"""
Production pipeline: pure vector search (Checkpoint 4's winning config).

Checkpoint 5 (hybrid BM25 + vector search + cross-encoder re-ranking) was
fully built and rigorously tested -- see query_hybrid.py -- but measured
worse than this simpler pipeline (86.7% vs 96.7% hit-rate on the 30
question eval set) and introduced a fragile, heavily version-pinned
Windows ML dependency chain in exchange for that regression. Keeping
this simpler pipeline as production was a deliberate decision based on
that measurement, not an oversight. See README "Key decisions".

What this does, in order:
  1. Embeds your question
  2. Compares it against every stored chunk vector by cosine similarity,
     keeps the top_k closest
  3. Stuffs those chunks into a prompt as context
  4. Asks the LLM to answer using only that context

Run ingest.py first, or this will have nothing to search.
"""

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOP_K = 5
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def reload_corpus():
    """Re-read chunks.json and embeddings.npy from disk into memory, and
    print the active-corpus banner. Called once at import time, and again
    by app.py after a fresh upload gets ingested -- so an already-running
    process picks up the new corpus without needing a restart.
    """
    global chunks, embedding_matrix
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embedding_matrix = np.load(EMBEDDINGS_FILE)

    unique_sources = sorted(set(c["source"] for c in chunks))
    print("=" * 55)
    print(f"ACTIVE CORPUS: {len(unique_sources)} source file(s), {len(chunks)} chunks")
    for s in unique_sources[:5]:
        print(f"  - {s}")
    if len(unique_sources) > 5:
        print(f"  ... and {len(unique_sources) - 5} more")
    print("=" * 55)


reload_corpus()


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(
        model=os.environ["EMBED_MODEL"],
        input=[question],
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Find the top_k chunks whose embeddings are most similar to the question."""
    query_vector = embed_query(question)
    similarities = embedding_matrix @ query_vector
    top_indices = np.argsort(-similarities)[:top_k]

    retrieved = []
    for idx in top_indices:
        chunk = chunks[idx]
        retrieved.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page"),  # None for formats with no page concept, or older chunks.json
                "score": float(similarities[idx]),
            }
        )
    return retrieved


def format_source_label(c: dict) -> str:
    """'quant.pdf (page 34)' when a page number is known, else just the filename."""
    label = c["source"]
    if c.get("page"):
        label += f" (page {c['page']})"
    return label


def build_prompt(question: str, chunks_: list[dict]) -> str:
    # Numbered sources -- the model cites by number, not by repeating the
    # filename, which is easier to verify programmatically afterward.
    context = "\n\n---\n\n".join(
        f"[{i + 1}] Source: {format_source_label(c)}\n{c['text']}" for i, c in enumerate(chunks_)
    )
    return f"""Answer the question using ONLY the context below. Cite the sources you used inline with bracketed numbers matching the list below, like [1] or [2][3]. Every factual claim should have at least one citation.

If the context does not contain the answer, say EXACTLY "I don't have enough information to answer that." and cite nothing.

Context:
{context}

Question: {question}

Answer (with inline citations):"""


def extract_citations(answer: str, num_sources: int) -> dict:
    """Pull citation numbers like [1] or [2][3] out of the answer and check
    they're all within the range of sources actually given to the model.
    A citation number outside that range is a real red flag -- the model
    referencing a source that was never retrieved for this question.
    """
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    is_refusal = "i don't have enough information" in answer.lower()
    invalid = {n for n in cited if n < 1 or n > num_sources}
    return {
        "cited": sorted(cited),
        "invalid": sorted(invalid),
        "is_refusal": is_refusal,
        "has_citation": len(cited) > 0,
    }


def ask(question: str) -> str:
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    answer = response.choices[0].message.content

    citation_check = extract_citations(answer, len(retrieved))
    if citation_check["invalid"]:
        print(f"\n  WARNING: answer cites source(s) {citation_check['invalid']} that don't exist in the retrieved context -- possible fabrication.")
    elif not citation_check["is_refusal"] and not citation_check["has_citation"]:
        print("\n  WARNING: answer isn't a refusal but has no citations at all.")

    print("\n--- Retrieved chunks (closest first, numbered for citations) ---")
    for i, c in enumerate(retrieved):
        print(f"  [{i + 1}] {format_source_label(c)}  (similarity: {c['score']:.4f})")

    print("\n--- Answer ---")
    print(answer)
    return answer


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question about FastAPI (or 'quit'): ")
        if q.lower() == "quit":
            break
        try:
            ask(q)
        except Exception:
            import traceback
            print("\n--- Something went wrong ---")
            traceback.print_exc()