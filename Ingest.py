"""
Ingestion pipeline: any folder (or single file) of your own documents ->
searchable chunks.

Supports .md, .txt, .pdf, .docx, .html/.htm, read recursively from
whatever folder you point it at, or a single file directly.

Usage:
    python ingest.py                       # uses DOCS_PATH default below
    python ingest.py path/to/your/folder   # ingest a different folder instead
    python ingest.py path/to/one/file.pdf  # or a single file directly

Re-running this OVERWRITES chunks.json and embeddings.npy -- only one
document set is "active" at a time. To switch back to a previous corpus,
just re-run ingest.py pointed at that folder again.

The core pipeline (ingest_path) is also imported directly by api.py's
/ingest endpoint, so a document can be uploaded over HTTP without a
server restart -- see that file for how it reloads query.py's in-memory
state afterward.
"""

import os
import sys
import glob
import json
import bisect
import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Config ---
DOCS_PATH = sys.argv[1] if len(sys.argv) > 1 else "fastapi_docs/docs/en/docs"
CHUNK_SIZE = 1500       # characters per chunk -- chosen in Checkpoint 4's comparison
CHUNK_OVERLAP = 300     # characters shared between consecutive chunks
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
CORPUS_INFO_FILE = "corpus_info.json"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def read_text_file(filepath: str) -> tuple[str, list[int] | None]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(), None


def read_pdf(filepath: str) -> tuple[str, list[int]]:
    """Returns the full text AND a list of character offsets marking where
    each page starts in that text -- this is what lets us cite an actual
    page number later instead of just repeating the filename.
    """
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    page_texts = [page.extract_text() or "" for page in reader.pages]

    page_offsets = []
    offset = 0
    for text in page_texts:
        page_offsets.append(offset)
        offset += len(text) + 2  # +2 accounts for the "\n\n" separator added below

    full_text = "\n\n".join(page_texts)
    return full_text, page_offsets


def read_docx(filepath: str) -> tuple[str, list[int] | None]:
    import docx
    doc = docx.Document(filepath)
    return "\n\n".join(p.text for p in doc.paragraphs), None


def read_html(filepath: str) -> tuple[str, list[int] | None]:
    from bs4 import BeautifulSoup
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n"), None


def offset_to_page(offset: int, page_offsets: list[int] | None) -> int | None:
    """Given a character offset into a document's full text, return the
    1-indexed page number it falls on. None if the format has no page
    concept (md/txt/docx/html) -- citations just fall back to filename only.
    """
    if not page_offsets:
        return None
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return max(idx, 0) + 1


EXTRACTORS = {
    ".md": read_text_file,
    ".txt": read_text_file,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".html": read_html,
    ".htm": read_html,
}


def load_documents(docs_path: str) -> list[dict]:
    """Read every supported file under docs_path (recursively), or a
    single file if docs_path points directly at one.
    Returns a list of {source: filepath, text: extracted text}.
    """
    if os.path.isfile(docs_path):
        all_paths = [docs_path]
    else:
        all_paths = glob.glob(f"{docs_path}/**/*", recursive=True)
    documents = []
    skipped = 0

    for filepath in all_paths:
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text, page_offsets = EXTRACTORS[ext](filepath)
        except Exception as e:
            print(f"  Skipping {filepath}: {e}")
            skipped += 1
            continue

        if text.strip():
            documents.append({"source": filepath, "text": text, "page_offsets": page_offsets})
        else:
            skipped += 1

    print(f"Loaded {len(documents)} documents ({skipped} skipped/empty)")
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """Fixed-size character chunking with overlap. Returns (chunk_text,
    start_offset) pairs -- the start_offset is what offset_to_page() uses
    to figure out which page each chunk begins on.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((text[start:end], start))
        start = end - overlap
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenRouter's embeddings endpoint in batches."""
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=os.environ["EMBED_MODEL"],
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    return embeddings


def ingest_path(docs_path: str) -> dict:
    """Run the full ingestion pipeline against docs_path and OVERWRITE the
    active corpus (chunks.json / embeddings.npy / corpus_info.json).

    Returns a summary dict on success, or {"error": "..."} if nothing
    could be ingested -- in the error case, nothing on disk is touched,
    so the previous corpus stays fully intact.

    This is the single reusable core called by both main() (the CLI) and
    api.py's /ingest endpoint, so there's exactly one ingestion code path
    to keep correct.
    """
    documents = load_documents(docs_path)
    if not documents:
        return {"error": f"No supported documents found at: {docs_path}"}

    all_chunks, all_sources, all_pages, all_ids = [], [], [], []
    for doc in documents:
        for idx, (chunk, start_offset) in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            all_chunks.append(chunk)
            all_sources.append(doc["source"])
            all_pages.append(offset_to_page(start_offset, doc.get("page_offsets")))
            all_ids.append(f"{doc['source']}::{idx}")

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

    embeddings = embed_texts(all_chunks)
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / norms
    np.save(EMBEDDINGS_FILE, embedding_matrix)

    records = [
        {"id": i, "source": s, "text": c, "page": p}
        for i, s, c, p in zip(all_ids, all_sources, all_chunks, all_pages)
    ]
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f)

    unique_sources = sorted(set(all_sources))
    corpus_info = {
        "ingested_from": docs_path,
        "ingested_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_source_files": len(unique_sources),
        "num_chunks": len(all_chunks),
        "source_files": unique_sources[:20],
        "source_files_truncated": len(unique_sources) > 20,
    }
    with open(CORPUS_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus_info, f, indent=2)

    return corpus_info


def main():
    print("[ingest.py version: single-file support enabled]")
    print(f"Ingesting from: {DOCS_PATH}")
    print(f"  (resolved as a file: {os.path.isfile(DOCS_PATH)}, as a folder: {os.path.isdir(DOCS_PATH)})\n")

    result = ingest_path(DOCS_PATH)

    if "error" in result:
        print("\n" + "=" * 55)
        print("INGESTION FAILED -- " + result["error"])
        print("Your PREVIOUS corpus (if any) is still active and untouched,")
        print("since nothing here got the chance to overwrite it.")
        print("=" * 55)
        return

    print(f"\nStored {result['num_chunks']} chunks in {CHUNKS_FILE} and {EMBEDDINGS_FILE}")
    print(f"Wrote {CORPUS_INFO_FILE} -- {result['num_source_files']} source file(s)")


if __name__ == "__main__":
    main()