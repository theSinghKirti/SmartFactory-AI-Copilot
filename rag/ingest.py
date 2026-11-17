"""
rag/ingest.py
ChromaDB ingestion for factory incidents.

Design decisions:
- Uses sqlite_event_id as the stable document ID to prevent duplicates
  across re-runs. If that field is absent, falls back to a
  timestamp+worker_id composite (legacy).
- Lazy client initialization: the PersistentClient is created on first
  call to get_collection() so unit tests can import this module without
  triggering a filesystem side-effect.
"""

import os
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database", "chroma"
)

_client = None
_collection = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("✅ ChromaDB PersistentClient ready at %s", CHROMA_PATH)
    return _client


def get_collection(name: str = "factory_incidents"):
    """Return (and lazily create) the named ChromaDB collection."""
    client = _get_client()
    return client.get_or_create_collection(name=name)


# Module-level accessor kept for backward compatibility with existing imports
# (e.g. `from rag.ingest import collection` in database_api.py and tests).
# The property forwards to the lazy initializer.
class _CollectionProxy:
    """Thin proxy so `collection.query(...)` / `collection.count()` etc. work."""
    def __getattr__(self, item):
        return getattr(get_collection("factory_incidents"), item)

    def __call__(self, *args, **kwargs):
        return get_collection("factory_incidents")(*args, **kwargs)


collection = _CollectionProxy()


# ------------------------------------------------------------------
# Public ingestion API
# ------------------------------------------------------------------

def add_incident(text: str, metadata: dict) -> None:
    """
    Upsert a single incident into the factory_incidents ChromaDB collection.

    Document ID strategy (priority order):
      1. str(sqlite_event_id)   — stable, unique, idempotent
      2. timestamp_workerId     — legacy fallback when no event_id supplied

    Required metadata keys (all produced by incident_pipeline.py):
        camera, worker_id, zone, event, severity, run_id, sqlite_event_id,
        timestamp (added by incident_pipeline.py after this fix)

    Missing keys are tolerated — the document is stored with what is available
    so a partial metadata failure never blocks incident storage.
    """
    if not text or not isinstance(text, str):
        logger.warning("add_incident: empty or non-string text — skipping.")
        return

    # Build a stable, unique document ID
    event_id = metadata.get("sqlite_event_id")
    if event_id is not None:
        doc_id = f"event_{event_id}"
    else:
        # Legacy path: requires both keys
        ts = metadata.get("timestamp", "unknown")
        wid = metadata.get("worker_id", "unknown")
        doc_id = f"{ts}_{wid}"
        logger.warning(
            "add_incident: sqlite_event_id missing — using legacy ID '%s'. "
            "This may cause duplicate documents on re-runs.",
            doc_id,
        )

    # Ensure all metadata values are ChromaDB-safe (str/int/float/bool only)
    safe_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe_meta[k] = v
        elif v is None:
            safe_meta[k] = ""
        else:
            safe_meta[k] = str(v)

    try:
        col = get_collection("factory_incidents")
        col.upsert(
            documents=[text],
            metadatas=[safe_meta],
            ids=[doc_id],
        )
        logger.info("✅ ChromaDB upsert OK | id=%s", doc_id)
    except Exception as exc:
        logger.error("❌ ChromaDB upsert failed | id=%s | error=%s", doc_id, exc)
        raise
