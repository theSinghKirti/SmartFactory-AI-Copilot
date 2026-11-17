"""
ai/semantic_search_service.py

Semantic search over the factory_incidents ChromaDB collection.

Changes from original:
- Removed all hardcoded mock_results.
- Queries the actual persistent ChromaDB collection via rag.ingest.
- Returns real incident metadata retrieved from stored vectors.
- Handles empty collections and missing metadata gracefully.
- Supports configurable top-k via `limit` parameter.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Executes semantic vector queries against the factory_incidents
    ChromaDB collection and returns structured evidence summaries.
    """

    def retrieve_concise_evidence(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Query the ChromaDB factory_incidents collection with the user's
        natural-language query and return a structured summary of the
        most semantically similar incident records.

        Args:
            query:  The user's natural language question.
            limit:  Maximum number of results to return (default 5).

        Returns:
            A formatted string of matching incidents, or a clear message
            when no results are found.
        """
        if not query or not query.strip():
            return "No query provided — please describe what you are looking for."

        try:
            from rag.ingest import get_collection

            col = get_collection("factory_incidents")

            # Guard: empty collection
            count = col.count()
            if count == 0:
                return (
                    "The incident vector database is currently empty. "
                    "Process a video through the Safety Detection pipeline first "
                    "to populate the incident history."
                )

            # Clamp n_results to collection size to avoid ChromaDB errors
            n_results = min(limit, count)

            results = col.query(
                query_texts=[query],
                n_results=n_results,
            )

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            if not docs:
                return "No matching incidents found in the semantic database for that query."

            evidence_lines = []
            for idx, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
                # Safe extraction — all fields may be absent for older records
                timestamp  = meta.get("timestamp", "N/A")
                zone       = meta.get("zone", "Unknown Zone")
                worker     = meta.get("worker_id", "N/A")
                violation  = meta.get("event", "Unknown Violation")
                severity   = str(meta.get("severity", "N/A")).upper()
                confidence = meta.get("confidence")
                image_path = meta.get("image_path", "")

                conf_str = f" | Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
                image_str = f"\n  📸 Evidence: {image_path}" if image_path else ""

                evidence_lines.append(
                    f"**[Incident #{idx}]**\n"
                    f"  🕐 Time: {timestamp}\n"
                    f"  📍 Zone: {zone} | 👷 Worker: {worker}\n"
                    f"  ⚠️  Violation: {violation} | Severity: {severity}{conf_str}\n"
                    f"  📋 Description: {doc.strip()}"
                    f"{image_str}"
                )

            header = f"Found **{len(evidence_lines)}** relevant incident(s) for: *\"{query}\"*\n\n"
            return header + "\n\n---\n\n".join(evidence_lines)

        except Exception as exc:
            logger.error("Semantic search failed: %s", exc, exc_info=True)
            return (
                f"⚠️ Semantic search encountered an error: {exc}\n"
                "If this is a ChromaDB connectivity issue, check that the database "
                "directory exists and is writable."
            )
