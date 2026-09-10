"""
Diagnostic, not part of the pipeline: shows the RAW LLM response from the
reranking call, and checks whether every candidate actually got scored --
large batch "score all N items and return correct indices" prompts are a
known weak spot for LLMs (miscounting, dropping items, index drift),
which is different from (and in addition to) the earlier truncation bug.
"""

import os
import json
import numpy as np
from Query import (
    client, chunks, embed_query, bm25_index, tokenize,
    embedding_matrix, CANDIDATE_POOL,
)

QUESTION = "Besides plain functions, what else can be used as a FastAPI dependency?"
TARGET_FILE = "tutorial/dependencies/classes-as-dependencies.md"


def normalize_source(path):
    normalized = path.replace("\\", "/")
    marker = "docs/en/docs/"
    idx = normalized.find(marker)
    return normalized[idx + len(marker):] if idx != -1 else normalized


def main():
    query_vector = embed_query(QUESTION)
    vector_scores = embedding_matrix @ query_vector
    vector_top_idx = np.argsort(-vector_scores)[:CANDIDATE_POOL]

    bm25_scores = bm25_index.get_scores(tokenize(QUESTION))
    bm25_top_idx = np.argsort(-bm25_scores)[:CANDIDATE_POOL]

    candidate_idx = sorted(set(vector_top_idx.tolist()) | set(bm25_top_idx.tolist()))
    print(f"Candidate pool size: {len(candidate_idx)}\n")

    target_positions = [
        i for i, idx in enumerate(candidate_idx)
        if normalize_source(chunks[idx]["source"]) == TARGET_FILE
    ]
    print(f"Target file sits at prompt position(s): {target_positions}\n")

    passages = "\n\n".join(
        f"[{i}] {chunks[idx]['text']}"
        for i, idx in enumerate(candidate_idx)
    )
    prompt = f"""Score how well each passage answers the question, from 0 (irrelevant) to 10 (directly and completely answers it).

Question: {QUESTION}

Passages:
{passages}

Respond with ONLY a JSON array covering every passage above, like:
[{{"index": 0, "score": 7}}, {{"index": 1, "score": 2}}]
No other text, no markdown formatting."""

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    print("=== RAW LLM RESPONSE ===")
    print(raw)
    print("=== END RAW RESPONSE ===\n")

    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        scored = json.loads(cleaned)
        print(f"Parsed {len(scored)} scored items (expected {len(candidate_idx)})")
        scored_positions = {item["index"] for item in scored}
        missing = set(range(len(candidate_idx))) - scored_positions
        print(f"Positions the LLM never scored: {sorted(missing)}")
        for pos in target_positions:
            if pos in scored_positions:
                score = next(item["score"] for item in scored if item["index"] == pos)
                print(f"Target at position {pos}: SCORED {score}")
            else:
                print(f"Target at position {pos}: MISSING from LLM response entirely")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"JSON parsing FAILED: {e}")
        print("This means rerank_llm() silently fell back to unscored candidate order.")


if __name__ == "__main__":
    main()