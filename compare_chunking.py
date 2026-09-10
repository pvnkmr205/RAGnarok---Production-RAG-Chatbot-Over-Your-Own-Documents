"""
Checkpoint 4 - Chunking strategy comparison

Builds the vector index from scratch under several different chunk_size /
overlap settings, and measures retrieval hit-rate for each against the
same 30-question eval set. This tells us whether chunking is actually the
lever worth pulling, rather than guessing.

This does NOT touch your existing chunks.json / embeddings.npy (the ones
query.py uses) -- everything here runs in memory so your working pipeline
stays untouched until you decide to adopt a winning config.

Cost note: each config re-embeds the full ~365K-token corpus, so 3 configs
costs about 3x a normal ingest.py run -- still a fraction of a cent total.
"""

import os
import glob
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCS_PATH = "fastapi_docs/docs/en/docs"
TOP_K = 5

CONFIGS = [
    {"name": "baseline (1000/200)", "chunk_size": 1000, "overlap": 200},
    {"name": "smaller (500/100)", "chunk_size": 500, "overlap": 100},
    {"name": "larger (1500/300)", "chunk_size": 1500, "overlap": 300},
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def load_markdown_files(docs_path):
    files = glob.glob(f"{docs_path}/**/*.md", recursive=True)
    documents = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            documents.append({"source": filepath, "text": text})
    return documents


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def normalize_source(path):
    normalized = path.replace("\\", "/")
    marker = "docs/en/docs/"
    idx = normalized.find(marker)
    return normalized[idx + len(marker):] if idx != -1 else normalized


def embed_texts(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=os.environ["EMBED_MODEL"], input=batch)
        embeddings.extend([item.embedding for item in resp.data])
        print(f"    embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return embeddings


def build_index(documents, chunk_size, overlap):
    all_chunks, all_sources = [], []
    for doc in documents:
        for chunk in chunk_text(doc["text"], chunk_size, overlap):
            all_chunks.append(chunk)
            all_sources.append(doc["source"])

    embeddings = np.array(embed_texts(all_chunks), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return all_chunks, all_sources, embeddings


def evaluate(eval_set, sources, embedding_matrix, top_k=TOP_K):
    hits = 0
    miss_ids = []
    for item in eval_set:
        resp = client.embeddings.create(model=os.environ["EMBED_MODEL"], input=[item["question"]])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec)

        similarities = embedding_matrix @ q_vec
        top_indices = np.argsort(-similarities)[:top_k]
        retrieved_sources = [normalize_source(sources[i]) for i in top_indices]

        if item["source_file"] in retrieved_sources:
            hits += 1
        else:
            miss_ids.append(item["id"])

    return hits / len(eval_set), miss_ids


def main():
    documents = load_markdown_files(DOCS_PATH)
    with open("eval_questions.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"{len(documents)} documents, {len(eval_set)} eval questions\n")

    results = []
    for cfg in CONFIGS:
        print(f"--- {cfg['name']} ---")
        chunks, sources, embeddings = build_index(documents, cfg["chunk_size"], cfg["overlap"])
        print(f"  {len(chunks)} chunks created, now evaluating...")
        hit_rate, miss_ids = evaluate(eval_set, sources, embeddings)
        print(f"  hit-rate: {hit_rate:.1%}  (misses: {miss_ids})\n")
        results.append({
            "config": cfg["name"],
            "num_chunks": len(chunks),
            "hit_rate": hit_rate,
            "miss_ids": miss_ids,
        })

    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    for r in results:
        print(f"  {r['config']:<22} chunks={r['num_chunks']:<6} hit-rate={r['hit_rate']:.1%}  misses={r['miss_ids']}")

    with open("chunking_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to chunking_comparison.json")


if __name__ == "__main__":
    main()