# RAGnarok---Production-RAG-Chatbot-Over-Your-Own-Documents
## Status

- [x] Checkpoint 0 — repo scaffolded, environment configured
- [x] Checkpoint 1 — naive pipeline working end-to-end
- [x] Checkpoint 2 — 30-question evaluation set written (`eval_questions.json`)
- [x] Checkpoint 3 — baseline retrieval hit-rate: **86.7% (26/30)**
- [ ] Checkpoint 4 — chunking strategy comparison
- [ ] Checkpoint 5 — hybrid retrieval + re-ranking
- [ ] Checkpoint 6 — citations + refusal handling
- [ ] Checkpoint 7 — FastAPI wrapper + cost tracking
- [ ] Checkpoint 8 — UI, deploy, demo video

## What it is

A chatbot that answers FastAPI questions strictly from FastAPI's own docs rather than general model knowledge, so answers stay grounded in the actual documented behavior and every answer traces back to a real source file. For an engineer: a retrieval-augmented generation pipeline — chunked docs embedded into vectors, retrieved by cosine similarity at query time, and fed to an LLM as grounding context.

## Architecture

Two pipelines:

**Ingestion (run once)**
`fastapi_docs/*.md` → chunk (1000 chars, 200 overlap) → embed (OpenRouter, `text-embedding-3-small`) → store (`chunks.json` + `embeddings.npy`)

**Query (every question)**
question → embed → cosine similarity search (top 5) → build prompt with retrieved context → LLM answers (OpenRouter, `gpt-4o-mini`)

## Setup

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git fastapi_docs
cd fastapi_docs && git sparse-checkout set docs/en/docs && cd ..

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
cp .env.example .env         # then fill in OPENROUTER_API_KEY
```

## Running

```bash
python ingest.py    # builds chunks.json and embeddings.npy -- run once, or after changing chunking config
python query.py     # interactive Q&A loop
```

## Evaluation set

`eval_questions.json` — 30 hand-written questions spanning 16 topic areas of the FastAPI docs (overview, path/query params, request bodies, dependencies, security, middleware, background tasks, testing, websockets, lifecycle events, settings, deployment). Each entry has:

- `question` — the test question
- `expected_answer` — the correct answer, written independently of the pipeline's output
- `source_file` — the doc file (relative to `docs/en/docs/`) that actually contains the answer, used to check whether retrieval finds the right chunk

This is the ground truth Checkpoint 3 measures retrieval against.

**Baseline result (fixed 1000-char chunks, 200 overlap):** 86.7% hit-rate (26/30), measured by `evaluate_retrieval.py`, saved to `eval_results.json`. All misses land in topics where FastAPI's docs split one concept across several near-duplicate sibling files (testing has 5 separate files, dependencies has 6) rather than in obviously bad chunks — under investigation before deciding whether Checkpoint 4 (chunking) or Checkpoint 5 (hybrid search / re-ranking) is the better fix.

## Key decisions

| Decision | Considered | Chose | Why | Gave up |
|---|---|---|---|---|
| Vector store | Chroma, Qdrant, pgvector, plain NumPy | Plain NumPy array + JSON | At ~1,900 chunks, brute-force cosine similarity takes under 1ms -- a dedicated vector DB solves a scale problem this project doesn't have. Also sidesteps an unresolved Windows-native crash in Chroma's query engine. | Metadata filtering, incremental updates, graceful scaling past ~100k+ vectors |
| LLM/embedding provider | OpenAI direct, Anthropic direct, OpenRouter | OpenRouter | One API key covers both chat and embeddings; models are swappable via `.env` without touching code | Slightly higher per-call latency than calling providers directly |

## Roadmap

See Status above. Full project spec: The Resume Project Vault 2026, Project D1 — Production RAG Chatbot.

## Credits

- Source documents: [fastapi/fastapi](https://github.com/fastapi/fastapi), `docs/en/docs`
