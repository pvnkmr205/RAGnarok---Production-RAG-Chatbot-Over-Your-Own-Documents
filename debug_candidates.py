"""
Diagnostic, not part of the pipeline: shows exactly which chunks make it
into the candidate pool (BM25 top-N union vector top-N) for a specific
question, BEFORE re-ranking touches anything.

This answers one question: did retrieval find the right chunk at all?
If not, no re-ranker -- LLM-based or a real cross-encoder -- can fix it,
since re-ranking only reorders candidates that already made the list.
"""

import numpy as np
from Query import embed_query, bm25_index, tokenize, embedding_matrix, chunks, CANDIDATE_POOL


def normalize_source(path):
    normalized = path.replace("\\", "/")
    marker = "docs/en/docs/"
    idx = normalized.find(marker)
    return normalized[idx + len(marker):] if idx != -1 else normalized


QUESTION = "Besides plain functions, what else can be used as a FastAPI dependency?"
TARGET_FILE = "tutorial/dependencies/classes-as-dependencies.md"


def main():
    query_vector = embed_query(QUESTION)
    vector_scores = embedding_matrix @ query_vector
    vector_top_idx = np.argsort(-vector_scores)[:CANDIDATE_POOL]

    bm25_scores = bm25_index.get_scores(tokenize(QUESTION))
    bm25_top_idx = np.argsort(-bm25_scores)[:CANDIDATE_POOL]

    print("Vector search top candidates:")
    for idx in vector_top_idx:
        marker = " <-- TARGET" if normalize_source(chunks[idx]["source"]) == TARGET_FILE else ""
        print(f"  {normalize_source(chunks[idx]['source']):<55} (score: {vector_scores[idx]:.4f}){marker}")

    print("\nBM25 top candidates:")
    for idx in bm25_top_idx:
        marker = " <-- TARGET" if normalize_source(chunks[idx]["source"]) == TARGET_FILE else ""
        print(f"  {normalize_source(chunks[idx]['source']):<55} (score: {bm25_scores[idx]:.4f}){marker}")

    candidate_idx = sorted(set(vector_top_idx.tolist()) | set(bm25_top_idx.tolist()))
    found = any(normalize_source(chunks[i]["source"]) == TARGET_FILE for i in candidate_idx)

    print(f"\nTotal unique candidates going into re-ranking: {len(candidate_idx)}")
    print(f"Target file '{TARGET_FILE}' reached the candidate pool: {found}")


if __name__ == "__main__":
    main()