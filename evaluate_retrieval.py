"""
Checkpoint 3 - Baseline retrieval measurement

Runs every question in eval_questions.json through retrieve() and checks
whether the chunk we know contains the answer (source_file) actually shows
up in the top-k results. This is "retrieval hit-rate" -- the single most
important number in a RAG project, because if retrieval doesn't find the
right chunk, the LLM has no chance of answering correctly no matter how
good the prompt is.

Run this after ingest.py. Produces a console report and eval_results.json
(a saved snapshot so we can compare against Checkpoint 4/5 changes later).
"""

import json
from Query import retrieve

TOP_K = 5


def normalize_source(path: str) -> str:
    """Convert a stored chunk source path into the same relative-path format
    used in eval_questions.json's source_file field, regardless of OS.
    e.g. 'fastapi_docs/docs/en/docs\\tutorial\\path-params.md' -> 'tutorial/path-params.md'
    """
    normalized = path.replace("\\", "/")
    marker = "docs/en/docs/"
    idx = normalized.find(marker)
    if idx != -1:
        return normalized[idx + len(marker):]
    return normalized


def main():
    with open("eval_questions.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    hits = 0
    topic_stats = {}

    for item in eval_set:
        retrieved = retrieve(item["question"], top_k=TOP_K)
        retrieved_sources = [normalize_source(c["source"]) for c in retrieved]
        hit = item["source_file"] in retrieved_sources
        rank = retrieved_sources.index(item["source_file"]) + 1 if hit else None

        if hit:
            hits += 1

        topic = item["topic"]
        topic_stats.setdefault(topic, {"hits": 0, "total": 0})
        topic_stats[topic]["total"] += 1
        if hit:
            topic_stats[topic]["hits"] += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_source": item["source_file"],
            "hit": hit,
            "rank": rank,
            "retrieved_sources": retrieved_sources,
        })

        status = f"HIT (rank {rank})" if hit else "MISS"
        print(f"[{item['id']:>2}] {status:<12} {item['question']}")

    hit_rate = hits / len(eval_set)
    print(f"\n{'=' * 55}")
    print(f"Overall hit-rate: {hits}/{len(eval_set)} = {hit_rate:.1%}")
    print(f"{'=' * 55}")

    print("\nBy topic:")
    for topic, stats in sorted(topic_stats.items()):
        rate = stats["hits"] / stats["total"]
        print(f"  {topic:<18} {stats['hits']}/{stats['total']}  ({rate:.0%})")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"hit_rate": hit_rate, "results": results}, f, indent=2)
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    main()