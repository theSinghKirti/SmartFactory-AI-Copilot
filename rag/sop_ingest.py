"""
rag/sop_ingest.py

Ingestion pipeline for safety SOP documents into the safety_sops ChromaDB collection.

Features:
- Loads .txt and .pdf files from knowledge_base/safety_sops/
- Splits long documents into overlapping chunks
- Stores source filename, category, and chunk index as metadata
- Idempotent: re-ingesting the same file does not create duplicate vectors
  (IDs are derived from filename + chunk index)
- Supports standalone retrieval via retrieve_sop_context()

Dependencies:
- chromadb (via rag.ingest._get_client)
- PyMuPDF (fitz) for PDF extraction — already in requirements.txt
"""

import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SOP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base", "safety_sops",
)

COLLECTION_NAME = "safety_sops"
CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between adjacent chunks


# ------------------------------------------------------------------
# Text extraction helpers
# ------------------------------------------------------------------

def _extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_text_from_pdf(path: str) -> str:
    """Use PyMuPDF (fitz) to extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed — cannot extract PDF text.")
        return ""
    except Exception as exc:
        logger.error("PDF extraction failed for %s: %s", path, exc)
        return ""


def _extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return _extract_text_from_txt(path)
    elif ext == ".pdf":
        return _extract_text_from_pdf(path)
    else:
        logger.warning("Unsupported file type: %s — skipping.", path)
        return ""


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping character-level chunks.
    Returns a list of string chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ------------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------------

def ingest_sop_documents(sop_dir: str = SOP_DIR) -> int:
    """
    Load all .txt and .pdf files from sop_dir and upsert them into the
    safety_sops ChromaDB collection.

    Returns the total number of chunks upserted.
    """
    if not os.path.isdir(sop_dir):
        logger.warning("SOP directory not found: %s — skipping ingestion.", sop_dir)
        return 0

    from rag.ingest import _get_client
    client = _get_client()
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    files = [
        f for f in os.listdir(sop_dir)
        if f.lower().endswith((".txt", ".pdf"))
    ]

    if not files:
        logger.warning("No .txt or .pdf files found in %s", sop_dir)
        return 0

    total_chunks = 0

    for filename in sorted(files):
        filepath = os.path.join(sop_dir, filename)
        text = _extract_text(filepath)
        if not text.strip():
            logger.warning("Empty text extracted from %s — skipping.", filename)
            continue

        chunks = _chunk_text(text)
        source_name = os.path.splitext(filename)[0]  # e.g. "ppe_compliance"

        # Determine category from first non-empty "Category:" line
        category = "Safety SOP"
        for line in text.splitlines():
            if line.strip().lower().startswith("category:"):
                category = line.split(":", 1)[1].strip()
                break

        chunk_ids = []
        chunk_docs = []
        chunk_metas = []

        for chunk_idx, chunk_text in enumerate(chunks):
            # Stable ID: source filename + chunk index
            doc_id = f"sop_{source_name}_chunk{chunk_idx}"
            meta = {
                "source": filename,
                "source_name": source_name,
                "category": category,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "document_type": "safety_sop",
            }
            chunk_ids.append(doc_id)
            chunk_docs.append(chunk_text)
            chunk_metas.append(meta)

        # Batch upsert all chunks for this file
        try:
            col.upsert(
                documents=chunk_docs,
                metadatas=chunk_metas,
                ids=chunk_ids,
            )
            total_chunks += len(chunks)
            logger.info("✅ Ingested %s — %d chunks", filename, len(chunks))
        except Exception as exc:
            logger.error("❌ Failed to ingest %s: %s", filename, exc)

    logger.info("SOP ingestion complete — %d total chunks stored.", total_chunks)
    return total_chunks


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

def retrieve_sop_context(
    query: str,
    limit: int = 4,
) -> str:
    """
    Query the safety_sops ChromaDB collection and return a formatted
    string of relevant SOP passages with source attribution.

    Args:
        query:  Natural language question about safety procedures.
        limit:  Maximum number of chunks to return.

    Returns:
        A formatted string of SOP passages, or a helpful message if the
        collection is empty or the query returns no results.
    """
    if not query or not query.strip():
        return "No query provided."

    try:
        from rag.ingest import _get_client
        client = _get_client()
        col = client.get_or_create_collection(name=COLLECTION_NAME)

        count = col.count()
        if count == 0:
            return (
                "⚠️ The Safety SOP knowledge base is empty. "
                "Run `python -m rag.sop_ingest` to populate it, "
                "or check that the knowledge_base/safety_sops/ directory "
                "contains .txt or .pdf files."
            )

        n_results = min(limit, count)
        results = col.query(
            query_texts=[query],
            n_results=n_results,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return "No relevant safety procedures found for that query."

        sections = []
        for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
            source   = meta.get("source", "Unknown document")
            category = meta.get("category", "Safety SOP")
            chunk    = meta.get("chunk_index", "?")
            total    = meta.get("total_chunks", "?")

            sections.append(
                f"**[SOP Ref #{idx}]** — 📄 *{source}* (Category: {category} | "
                f"Chunk {chunk}/{total})\n\n"
                f"{doc.strip()}"
            )

        header = (
            f"Found **{len(sections)}** relevant SOP passage(s) for: "
            f"*\"{query}\"*\n\n"
            f"> ⚠️ *These passages are demonstration guidance, not official regulations.*\n\n"
        )
        return header + "\n\n---\n\n".join(sections)

    except Exception as exc:
        logger.error("SOP retrieval failed: %s", exc, exc_info=True)
        return f"⚠️ SOP retrieval error: {exc}"


# ------------------------------------------------------------------
# CLI entry point: python -m rag.sop_ingest
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    n = ingest_sop_documents()
    print(f"\n✅ SOP ingestion finished. Total chunks stored: {n}")
