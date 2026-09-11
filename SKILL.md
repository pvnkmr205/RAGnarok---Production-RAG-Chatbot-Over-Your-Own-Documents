---
project: RAGnarok
repo: https://github.com/pvnkmr205/RAGnarok---Production-RAG-Chatbot-Over-Your-Own-Documents
live: https://ragnarok---rag-chatbot-over-your-own-documents.streamlit.app/
---

# 1. What this project is
For a non-technical friend: you upload any document you have — a PDF, a Word file, a webpage — and then ask it questions in plain English, and it answers using only what's actually in that document, telling you exactly where it found the answer.
For an engineer: a retrieval-augmented generation (RAG) pipeline — documents are extracted to text, chunked, embedded into vectors, retrieved by cosine similarity at query time, and fed to an LLM as grounding context, with programmatic citation validation and a measured refusal fallback to prevent hallucination.

# 2. Problem it solves
Reading a long document to find one specific fact is slow, and general-purpose chatbots will confidently answer questions about a document they've never actually seen. RAGnarok only answers from what you actually gave it, and says "I don't have enough information" rather than guessing — which I didn't just assume, I measured: 100% correct refusal across 10 deliberately unanswerable test questions, and I manually verified a specific synthesized claim from a 94-page test PDF against the raw source text to confirm the system wasn't fabricating anything.

# 3. Architecture
See `architecture.svg` in this repo for the diagram. ASCII version:

```
INGESTION (ingest.py) -- run once per document
  any file (PDF/DOCX/HTML/MD/TXT)
       |
       v
  extract text (page-boundary-aware for PDFs)
       |
       v
  chunk (1500 chars, 300 overlap)
       |
       v
  embed (OpenRouter, text-embedding-3-small)
       |
       v
  store: chunks.json + embeddings.npy

QUERY (query.py) -- every question
  question -> embed -> cosine similarity search (top 5)
       |
       v
  build prompt (numbered sources, citation + refusal rules)
       |
       v
  LLM answer (OpenRouter, gpt-4o-mini)
       |
       v
  validate citations against what was actually retrieved

Three interfaces call the same core functions:
  CLI (query.py)  |  FastAPI (api.py: /ask /corpus /health)  |  Streamlit UI (app.py, deployed)
```

Components:
- `ingest.py` -> multi-format extraction, chunking, embedding, storage -> `pypdf`/`python-docx`/`beautifulsoup4` chosen specifically for being lightweight with minimal native-dependency risk, after two separate native-library crises elsewhere in the project made that risk very real
- `query.py` -> retrieval, prompt construction, citation validation (the shared core every interface calls) -> plain NumPy cosine similarity instead of a vector DB, because Chroma crashed unrecoverably on Windows and at this corpus scale (1-2K chunks) brute-force search is sub-millisecond anyway
- `api.py` -> HTTP API, response caching, token cost tracking -> reuses `query.py`'s functions rather than duplicating logic
- `app.py` -> upload + chat UI -> calls the core functions in-process rather than over HTTP to `api.py`, so it's one process and simpler to containerize
- `query_hybrid.py` -> BM25 + vector search + cross-encoder re-ranking (preserved, not production) -> built and rigorously tested, measured worse than the simpler pipeline, kept as documented engineering history rather than deleted

# 4. Key decisions and trade-offs
| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Vector store | Chroma, Qdrant, pgvector, plain NumPy | Plain NumPy array + JSON | At ~1,300-1,900 chunks, brute-force cosine similarity takes under 1ms -- a dedicated vector DB solves a scale problem I don't have. Also sidesteps an unresolved Windows-native crash in Chroma's query engine (confirmed via matching GitHub issues) | Metadata filtering, incremental updates, graceful scaling past ~100k+ vectors |
| Chunk size/overlap | 1000/200, 500/100, 1500/300 | 1500/300 | Measured, not guessed: 96.7% vs 90.0% hit-rate on a 30-question eval set I wrote myself | Slightly more tokens sent to the LLM per query |
| Re-ranker (hybrid retrieval) | Local cross-encoder, LLM-based scoring, none | None -- pure vector search stayed in production | Built and fully tested both alternatives. The cross-encoder measured worse (86.7% vs 96.7%) even after resolving 5 sequential Windows dependency conflicts to get it running. LLM-based scoring worked once two of its own bugs were fixed, but was superseded before a full evaluation run | A theoretically more sophisticated pipeline that measurably underperformed the simpler one |
| Document scope | FastAPI docs only vs. general-purpose | General-purpose (any PDF/DOCX/HTML/MD/TXT) | Matches the project's actual goal; validated against two genuinely different real documents (FastAPI's technical docs, a 94-page economics paper) | True multi-corpus support -- only one document set is active at a time |
| LLM/embedding provider | OpenAI direct, Anthropic direct, OpenRouter | OpenRouter | One API key for both chat and embeddings, swappable via `.env` without code changes | Free-tier chat models proved unreliable (stale model slugs, daily rate limits) -- settled on a cheap paid model with explicitly capped `max_tokens` instead |
| UI-to-core wiring | Streamlit calling `api.py` over HTTP vs. calling core functions directly | Direct, in-process | Single service, one process, simpler to containerize and deploy | `api.py` and `app.py` don't share a running process or its cache -- each independently imports the same core modules |

# 5. Skills demonstrated
- [x] REST API design -- evidence: `api.py` (`/ask`, `/corpus`, `/health`, auto-documented at `/docs`)
- [x] Embeddings and vector similarity search, implemented from scratch -- evidence: `query.py` `retrieve()` (cosine similarity via NumPy, no vector-DB library)
- [x] Controlled experimentation / evaluation methodology -- evidence: `eval_questions.json`, `evaluate_retrieval.py`, `compare_chunking.py`
- [x] Hybrid retrieval and re-ranking, including knowing when NOT to ship it -- evidence: `query_hybrid.py`, README "Hybrid retrieval experiment"
- [x] Prompt engineering for grounding and citation enforcement -- evidence: `query.py` `build_prompt()`, `extract_citations()`
- [x] LLM refusal/hallucination testing -- evidence: `refusal_questions.json`, `test_refusal.py` (100% pass rate)
- [x] Cost and latency control -- evidence: `api.py` `max_tokens` caps, in-memory response cache, per-request token tracking
- [x] Multi-format document parsing -- evidence: `ingest.py` `EXTRACTORS` (pypdf, python-docx, beautifulsoup4)
- [x] Diagnosing native-dependency conflicts on Windows -- evidence: `requirements-hybrid.txt` version pins, README "Key decisions" re-ranker row
- [x] Containerization -- evidence: `Dockerfile`, `.dockerignore`
- [x] Cloud deployment -- evidence: live Streamlit Community Cloud URL (see frontmatter)

# 6. Numbers I measured
| Metric | Before | After | How I measured it |
|---|---|---|---|
| Retrieval hit-rate (chunking) | 90.0% (1000/200 chunks) | 96.7% (1500/300 chunks) | `evaluate_retrieval.py` against my 30-question eval set |
| Retrieval hit-rate (hybrid + re-rank) | 96.7% (pure vector, production) | 86.7% (hybrid + cross-encoder) | Same eval set -- hybrid regressed; documented, not shipped |
| Refusal rate on unanswerable questions | untested | 100% (10/10) | `test_refusal.py` against 10 deliberately unanswerable questions |
| Tokens per grounded, cited answer | -- | ~2,024 (1,989 prompt + 35 completion) | Live `/ask` response, measured via OpenRouter's usage data |
| Corpus scale tested | -- | 155 files / 1,293 chunks (FastAPI docs); 1 file / 205 chunks (94-page PDF) | `ingest.py` output on two genuinely different real document sets |

# 7. Things that broke and how I fixed them
1. Symptom: `query.py` returned silently to the shell prompt with zero output after asking a question -- no error, no traceback.
   Cause: Chroma's native HNSW query engine crashes on Windows below the level Python's `try/except` can catch (confirmed against multiple matching open GitHub issues with the identical symptom).
   Fix: Replaced Chroma with a plain NumPy array + JSON for storage -- brute-force cosine similarity, sub-millisecond at this scale.
   Lesson: When a compiled/native dependency fails silently on Windows, check whether the project's actual scale even needs that dependency's complexity before debugging deeper into it.

2. Symptom: An LLM-based re-ranker scored the one chunk I knew was correct at 0/10, tied with nearly every other candidate.
   Cause: Asking the model to score ~26 full passages in a single call caused it to collapse to near-uniform low scores instead of genuinely differentiating -- confirmed by inspecting the raw response, not just the parsed output.
   Fix: Split the same scoring task into small batches of 5 candidates per call.
   Lesson: Large-batch LLM judgment tasks are a distinct, real failure mode from prompt bugs -- verify against raw model output before assuming the logic is wrong.

3. Symptom: Installing a local cross-encoder reranker triggered a five-step cascade: a native DLL crash, then a torch-version mismatch, then a numpy ABI incompatibility, then a scipy/scikit-learn ABI conflict with that numpy version.
   Cause: A precise, undocumented set of mutually-compatible package versions was required on this machine; each fix attempted introduced a new incompatibility one layer down.
   Fix: Diagnosed and resolved each in sequence (VC++ Redistributable, forcing then reverting a torch version, pinning `numpy<2`, pinning `scipy<1.14` and `scikit-learn<1.5`), then pinned every exact version in `requirements-hybrid.txt`.
   Lesson: After all that effort, the resulting pipeline measurably underperformed the simpler one it replaced (86.7% vs 96.7%) -- so it wasn't adopted. Fighting for a dependency doesn't obligate you to ship the thing it enables.

4. Symptom: After re-ingesting a new document, the system kept answering questions from the *previous* document as if nothing had changed.
   Cause: `ingest.py`'s failure path (0 documents found) returned early without ever touching `chunks.json`/`embeddings.npy` -- so a silently failed re-ingestion looked identical to a successful one from the outside.
   Fix: Made 0-document ingestion fail loudly and explicitly, and added an "ACTIVE CORPUS" banner printed at every entry point's startup, computed live from `chunks.json` rather than a separate metadata file that could drift out of sync.
   Lesson: A script that fails safely (doesn't corrupt existing state) can still fail *misleadingly* -- visibility into current state matters as much as correctness of the happy path.

# 8. What I would do differently at 100x scale
- Vector store: NumPy brute-force cosine similarity is correct for ~2K chunks but would not hold at 200K+; I'd move to a real ANN index (pgvector or Qdrant) -- and re-run the same hit-rate measurement methodology to confirm it actually helps, rather than assume a fancier tool automatically wins, having already learned that lesson once with hybrid retrieval.
- Multi-tenancy: right now only one document set is active at a time (re-ingesting overwrites the previous corpus). At real scale with multiple concurrent users, I'd need per-session or per-user corpus isolation, almost certainly backed by a real database instead of flat JSON/NumPy files.
- Caching: the current in-memory cache resets on every restart and lives in a single process. At scale I'd move it to Redis so it survives restarts and works consistently across multiple server instances.

# 9. Interview answers I have rehearsed
Q: How do you know your RAG system is good? Give me a number and tell me how you measured it.
A: I wrote a 30-question evaluation set myself, with known correct answers and known source files, so I could check programmatically whether retrieval actually found the right chunk -- not just whether the final answer looked plausible. My baseline hit-rate was 90%, which I improved to 96.7% by testing three chunk-size configurations and keeping the one that measured best. I also built and fully tested a more sophisticated hybrid-search-plus-reranking pipeline, but it measured worse at 86.7%, so I made the deliberate decision not to ship it, and documented exactly why. Beyond retrieval, I tested refusal behavior on 10 deliberately unanswerable questions and got 100%, and I manually verified a specific synthesized claim in one answer against the raw source PDF to directly confirm the system wasn't fabricating anything.

Q: The model confidently answers a question your documents do not cover. How did you stop that?
A: I built an explicit refusal instruction into the prompt telling the model to say exactly "I don't have enough information" when retrieved context doesn't contain the answer, then tested it deliberately with 10 questions I knew weren't covered -- including a false-premise trick question assuming FastAPI has a CEO. It refused all 10. I also added programmatic citation validation: every answer has to cite a numbered source that was actually retrieved, and if it cites a number that doesn't exist, my code flags it as a possible fabrication instead of trusting the output blindly.

# 10. Honest limitations
- Only one document set is active at a time -- ingesting a new file overwrites the previous corpus; there's no real multi-document or multi-session support.
- The in-memory API cache resets on every server restart and isn't shared across multiple server instances.
- Citation validation confirms a cited source was actually retrieved, but does not verify the cited text truly supports the specific claim made -- it catches fabricated sources, not subtly misattributed ones.
- Chunking is fixed-size and character-based with no structural awareness (headings, paragraphs, code blocks). This was a measured, deliberate choice (see the chunking experiment) rather than an oversight, but it's still a real limit on highly structured documents.
- PDF page citations attribute a chunk to whichever page its first character falls on; a chunk spanning a page boundary is only attributed to its starting page.
- No authentication or rate limiting on the deployed app -- anyone with the URL can use it and consume API credits.

# 11. How to run it
```bash
git clone <repo> && cd RAGnarok
cp .env.example .env   # fill in the values listed below
pip install -r requirements.txt
python ingest.py                            # default: FastAPI docs reference corpus
python ingest.py path/to/your/document.pdf  # or point it at any document instead
streamlit run app.py
# open http://localhost:8501
```
Or with Docker:
```bash
docker build -t ragnarok .
docker run -p 8501:8501 -e OPENROUTER_API_KEY=... -e CHAT_MODEL=openai/gpt-4o-mini -e EMBED_MODEL=openai/text-embedding-3-small ragnarok
```
Required environment variables: `OPENROUTER_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`

# 12. Credits
- Reference documents: [fastapi/fastapi](https://github.com/fastapi/fastapi) (`docs/en/docs`)
- Test document: *"What Would it Cost to End Extreme Poverty?"* (Sahoo, Blumenstock, Niehaus, Selker, Wager)
- Project structure and methodology: The Resume Project Vault 2026 (@pratham.codes)
- Libraries: OpenAI Python SDK (via OpenRouter), NumPy, rank-bm25, pypdf, python-docx, beautifulsoup4, FastAPI, Streamlit
