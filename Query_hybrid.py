"""
Checkpoint 5 - Hybrid retrieval + LLM-based re-ranking (final version)

What this does, in order:
  1. Embeds your question (vector search candidates -- meaning-based)
  2. ALSO scores your question against every chunk with BM25 (keyword
     search candidates -- word-based). Vector search and BM25 catch
     different things, so we take the union of both.
  3. Sends the combined shortlist to the LLM in SMALL BATCHES (5 at a
     time, not all ~30 at once), asking it to score each chunk's
     relevance to the question directly. This is the same core idea as a
     cross-encoder re-ranker -- score the question and a chunk TOGETHER,
     not separately.
  4. Stuffs the final top_k into a prompt as context
  5. Asks the LLM to answer using only that context

Why an LLM instead of a local cross-encoder model: sentence-transformers
(which needs torch) failed on this machine across six different cascading
Windows dependency conflicts -- a DLL init crash, a too-old torch build,
a numpy/transformers version mismatch, and finally an incompatible
numpy/scipy/sklearn combination. Each was a real, fixable-in-isolation
problem, but the pattern itself (this is a well-known fragility of the
Python ML packaging ecosystem on Windows) made it not worth further time.
This LLM-based approach needs nothing but the packages already working
fine for the rest of the project.

Two bugs already found and fixed in this approach, worth knowing about:
  - Passages must be sent in FULL, not truncated -- FastAPI's docs often
    build up to their point rather than leading with it, so truncating
    can hide the actual answer from the scorer.
  - Batches must be SMALL (~5 items) -- asking the model to score ~26-30
    passages in one call made it collapse to near-uniform 0 scores
    instead of genuinely differentiating relevant passages.

Run ingest.py first, or this will have nothing to search.
"""

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

TOP_K = 5
CANDIDATE_POOL = 15    # how many each of BM25 and vector search contribute, before re-ranking
RERANK_BATCH_SIZE = 5  # score this many candidates per LLM call -- keeps each call's judgment reliable
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
embedding_matrix = np.load(EMBEDDINGS_FILE)  # shape: (num_chunks, dim), rows already unit-normalized


def tokenize(text: str) -> list[str]:
    """Very simple tokenizer: lowercase, keep only alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", text.lower())


print("Building BM25 index...")
bm25_corpus = [tokenize(c["text"]) for c in chunks]
bm25_index = BM25Okapi(bm25_corpus)
print("Ready. [query.py version: LLM small-batch reranking, full-text, gpt-4o-mini]")


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(
        model=os.environ["EMBED_MODEL"],
        input=[question],
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _score_batch(question: str, batch_idx: list[int]) -> list[tuple[int, float]]:
    """Score one small batch of candidates. Returns (chunk_index, score) pairs."""
    passages = "\n\n".join(
        f"[{i}] {chunks[idx]['text']}"
        for i, idx in enumerate(batch_idx)
    )
    prompt = f"""Score how well each passage answers the question, from 0 (irrelevant) to 10 (directly and completely answers it). Use the full 0-10 range -- don't default to 0 unless a passage is genuinely unrelated to the question.

Question: {question}

Passages:
{passages}

Respond with ONLY a JSON array covering every passage above, like:
[{{"index": 0, "score": 7}}, {{"index": 1, "score": 2}}]
No other text, no markdown formatting."""

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        scored = json.loads(raw)
        return [(batch_idx[item["index"]], float(item["score"])) for item in scored]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        # If a batch fails to parse, keep those candidates with a neutral
        # score rather than losing them or crashing the whole query.
        return [(idx, 0.0) for idx in batch_idx]


def rerank_llm(question: str, candidate_idx: list[int]) -> list[tuple[int, float]]:
    all_results = []
    for batch_start in range(0, len(candidate_idx), RERANK_BATCH_SIZE):
        batch = candidate_idx[batch_start:batch_start + RERANK_BATCH_SIZE]
        all_results.extend(_score_batch(question, batch))
    all_results.sort(key=lambda x: -x[1])
    return all_results


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid retrieval: BM25 + vector search each contribute candidates,
    the LLM re-ranks the combined shortlist (in small batches), and we
    return the final top_k by that re-ranked order.
    """
    query_vector = embed_query(question)
    vector_scores = embedding_matrix @ query_vector
    vector_top_idx = np.argsort(-vector_scores)[:CANDIDATE_POOL]

    bm25_scores = bm25_index.get_scores(tokenize(question))
    bm25_top_idx = np.argsort(-bm25_scores)[:CANDIDATE_POOL]

    candidate_idx = sorted(set(vector_top_idx.tolist()) | set(bm25_top_idx.tolist()))

    ranked = rerank_llm(question, candidate_idx)[:top_k]

    retrieved = []
    for idx, score in ranked:
        retrieved.append(
            {
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": score,
            }
        )
    return retrieved


def build_prompt(question: str, chunks_: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks_
    )
    return f"""Answer the question using ONLY the context below. If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str) -> str:
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    answer = response.choices[0].message.content

    print("\n--- Retrieved chunks (re-ranked) ---")
    for c in retrieved:
        print(f"  {c['source']}  (rerank score: {c['score']:.1f}/10)")

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