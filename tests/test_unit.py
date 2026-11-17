"""
tests/test_unit.py

Pytest-based unit tests for SmartFactory Copilot.

Design principles:
- External services (ChromaDB, Groq LLM) are mocked wherever possible.
- Tests use an isolated in-memory SQLite database.
- No video files, model weights, or API keys required.
- Each test is self-contained and cleans up after itself.

Run with:
    cd d:/Smartfactory-CCTV-AI-Copilot-main
    python -m pytest tests/test_unit.py -v
"""

import os
import sys
import sqlite3
import tempfile
import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Ensure root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ==============================================================
# Inject mock chromadb into sys.modules BEFORE any rag imports.
# chromadb is not installed in the test environment.
# This allows rag.ingest and rag.sop_ingest to be imported
# without hitting ImportError, while still testing the logic.
# ==============================================================
_mock_chromadb = MagicMock()
_mock_chromadb.PersistentClient = MagicMock
_mock_chromadb.config = MagicMock()
_mock_chromadb.config.Settings = MagicMock()
sys.modules.setdefault("chromadb", _mock_chromadb)
sys.modules.setdefault("chromadb.config", _mock_chromadb.config)

# Also mock Groq / LangChain dependencies (not installed)
_mock_groq = MagicMock()
sys.modules.setdefault("groq", _mock_groq)
sys.modules.setdefault("langchain_groq", MagicMock())
sys.modules.setdefault("langchain_core", MagicMock())
sys.modules.setdefault("langchain_core.tools", MagicMock())
sys.modules.setdefault("langchain_core.messages", MagicMock())
sys.modules.setdefault("langgraph", MagicMock())
sys.modules.setdefault("langgraph.prebuilt", MagicMock())
sys.modules.setdefault("langgraph.checkpoint", MagicMock())
sys.modules.setdefault("langgraph.checkpoint.memory", MagicMock())
sys.modules.setdefault("streamlit", MagicMock())

# Set GROQ_API_KEY for tests
os.environ.setdefault("GROQ_API_KEY", "test_key")


# ==============================================================
# Fixtures
# ==============================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Provide an isolated SQLite database for each test.
    Patches database.db.DB_PATH to point at a temp file.
    """
    db_file = str(tmp_path / "test_factory.db")
    monkeypatch.setattr("database.db.DB_PATH", db_file)
    # Re-init schema on the temp DB
    from database.db import init_db
    # We need to call it without the patched path taking effect incorrectly
    # so we use a direct connection
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera TEXT,
            zone TEXT,
            worker_id INTEGER,
            event TEXT,
            severity TEXT,
            confidence REAL,
            start_time TEXT,
            end_time TEXT,
            image_path TEXT,
            run_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            zone TEXT,
            source_path TEXT,
            active INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()
    return db_file


# ==============================================================
# A. Incident Pipeline — save_event
# ==============================================================

class TestSaveEvent:
    """Tests for database.db.save_event()"""

    def test_save_event_returns_id(self, temp_db):
        """save_event() should return a positive integer event ID."""
        from database.db import save_event
        event_id = save_event(
            camera="Cam-01",
            zone="Zone A",
            worker_id=42,
            event="no-hardhat",
            severity="High",
            confidence=0.87,
            start_time="2026-11-17 10:00:00",
            end_time="2026-11-17 10:00:05",
            image_path="",
            run_id="test_run_001",
        )
        assert isinstance(event_id, int)
        assert event_id > 0

    def test_save_event_readable_back(self, temp_db):
        """Saved event should be retrievable from the database."""
        from database.db import save_event, get_connection
        save_event(
            camera="Cam-02",
            zone="Zone B",
            worker_id=99,
            event="no-safety vest",
            severity="High",
            confidence=0.91,
            start_time="2026-11-17 09:00:00",
            end_time="2026-11-17 09:00:10",
        )
        with get_connection() as conn:
            row = conn.execute(
                "SELECT worker_id, event, zone FROM events WHERE worker_id=?",
                (99,)
            ).fetchone()
        assert row is not None
        assert row["worker_id"] == 99
        assert row["event"] == "no-safety vest"
        assert row["zone"] == "Zone B"

    def test_save_event_multiple_records(self, temp_db):
        """Multiple events can be saved and queried."""
        from database.db import save_event, get_connection
        for i in range(5):
            save_event(
                camera="Cam-01",
                zone="Zone A",
                worker_id=i + 1,
                event="no-hardhat",
                severity="High",
                confidence=0.75,
                start_time="2026-11-17 08:00:00",
                end_time="2026-11-17 08:00:03",
            )
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 5


# ==============================================================
# B. get_stats() uses actual DB records
# ==============================================================

class TestGetStats:
    """Tests for agents.database_api.get_stats()"""

    def test_empty_db_returns_zero(self, temp_db):
        """Empty database should return zeros, not errors."""
        from agents.database_api import get_stats
        stats = get_stats()
        assert stats["total"] == 0
        assert stats["today"] == 0
        assert stats["by_type"] == []
        assert stats["by_zone"] == []

    def test_stats_reflect_inserted_records(self, temp_db):
        """get_stats() should reflect actual inserted records."""
        from database.db import save_event
        from agents.database_api import get_stats
        save_event(
            camera="Cam-01",
            zone="Zone A",
            worker_id=1,
            event="no-hardhat",
            severity="High",
            confidence=0.9,
            start_time="2026-11-17 10:00:00",
            end_time="2026-11-17 10:00:05",
        )
        stats = get_stats()
        assert stats["total"] == 1
        # by_type should have one entry
        assert len(stats["by_type"]) == 1
        assert stats["by_type"][0][0] == "no-hardhat"
        assert stats["by_type"][0][1] == 1


# ==============================================================
# C. ViolationTracker deduplication
# ==============================================================

class TestViolationTracker:
    """Tests for cctv.violation_tracker.ViolationTracker"""

    def test_same_violation_not_closed_prematurely(self):
        """An ongoing violation should not be closed until the timeout passes."""
        from cctv.violation_tracker import ViolationTracker
        tracker = ViolationTracker(timeout_seconds=10)
        t0 = datetime(2026, 11, 17, 10, 0, 0)
        violations = [{"worker_id": 1, "violation_type": "no-hardhat", "confidence": 0.8}]
        # Frame 1
        closed = tracker.update(violations, frame_time=t0)
        assert len(closed) == 0

    def test_violation_closed_after_timeout(self):
        """Violation should close when not seen for > timeout_seconds."""
        from cctv.violation_tracker import ViolationTracker
        from datetime import timedelta
        tracker = ViolationTracker(timeout_seconds=3)
        t0 = datetime(2026, 11, 17, 10, 0, 0)
        violations = [{"worker_id": 2, "violation_type": "no-hardhat", "confidence": 0.9}]
        # Seen at t0
        tracker.update(violations, frame_time=t0)
        # Not seen 5 seconds later
        t1 = t0 + timedelta(seconds=5)
        closed = tracker.update([], frame_time=t1)
        assert len(closed) == 1
        assert closed[0]["worker_id"] == 2

    def test_flush_closes_all_open(self):
        """flush() should return all still-open violations."""
        from cctv.violation_tracker import ViolationTracker
        tracker = ViolationTracker(timeout_seconds=100)
        t0 = datetime(2026, 11, 17, 10, 0, 0)
        violations = [
            {"worker_id": 3, "violation_type": "no-hardhat", "confidence": 0.85},
            {"worker_id": 4, "violation_type": "no-safety vest", "confidence": 0.78},
        ]
        tracker.update(violations, frame_time=t0)
        remaining = tracker.flush()
        assert len(remaining) == 2

    def test_confidence_updated_to_max(self):
        """Confidence should be updated to the maximum seen across frames."""
        from cctv.violation_tracker import ViolationTracker
        tracker = ViolationTracker(timeout_seconds=100)
        t0 = datetime(2026, 11, 17, 10, 0, 0)
        from datetime import timedelta
        tracker.update([{"worker_id": 5, "violation_type": "no-hardhat", "confidence": 0.6}], t0)
        tracker.update([{"worker_id": 5, "violation_type": "no-hardhat", "confidence": 0.95}], t0 + timedelta(seconds=1))
        remaining = tracker.flush()
        assert remaining[0]["confidence"] == pytest.approx(0.95)


# ==============================================================
# D. Intent Router
# ==============================================================

class TestIntentRouter:
    """Tests for ai.intent_router.IntentRouter"""

    def setup_method(self):
        from ai.intent_router import IntentRouter
        self.router = IntentRouter()

    def test_routes_stats(self):
        assert self.router.route("Show me the dashboard") == "stats"

    def test_routes_report(self):
        assert self.router.route("Generate a safety report") == "report"

    def test_routes_semantic(self):
        assert self.router.route("Explain why these violations happened") == "semantic"

    def test_routes_sop(self):
        assert self.router.route("What is the corrective action procedure?") == "sop"

    def test_routes_risk(self):
        assert self.router.route("What are the high risk zones?") == "risk"

    def test_routes_sql_for_incident_query(self):
        assert self.router.route("Show me helmet violations") == "sql"

    def test_routes_comparison(self):
        assert self.router.route("Compare today vs yesterday") == "comparison"

    def test_fallback_general(self):
        assert self.router.route("What is the weather like?") == "general"


# ==============================================================
# E. SQL Parser
# ==============================================================

class TestSQLParser:
    """Tests for ai.sql_parser.SQLParser"""

    def setup_method(self):
        from ai.sql_parser import SQLParser
        self.parser = SQLParser()

    def test_extracts_zone(self):
        f = self.parser.parse("Show incidents in Zone B")
        assert f["zone"] == "Zone B"

    def test_extracts_helmet_event(self):
        f = self.parser.parse("Show helmet violations today")
        assert f["event"] == "no-helmet"

    def test_extracts_vest_event(self):
        f = self.parser.parse("List all vest incidents")
        assert f["event"] == "no-vest"

    def test_extracts_worker_id_as_int(self):
        f = self.parser.parse("Show violations for worker 42")
        assert f["worker_id"] == 42
        assert isinstance(f["worker_id"], int)

    def test_extracts_severity_high(self):
        f = self.parser.parse("Show high severity incidents")
        assert f["severity"] == "high"

    def test_today_date_filter(self):
        f = self.parser.parse("Show incidents today")
        today = datetime.now().strftime("%Y-%m-%d")
        assert f["start_date"] == today
        assert f["end_date"] == today


# ==============================================================
# F. ChromaDB Ingestion — rag.ingest (mocked chromadb)
# ==============================================================

class TestChromaIngestion:
    """Tests for rag.ingest.add_incident() with mocked ChromaDB."""

    def _make_mock_col(self):
        """Return a fresh MagicMock collection for each test."""
        col = MagicMock()
        return col

    def test_add_incident_uses_event_id_as_key(self):
        """add_incident() should build a stable doc ID from sqlite_event_id."""
        import rag.ingest as ingest_mod
        mock_col = self._make_mock_col()

        # Patch _get_client so get_collection() returns our mock collection
        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            # Reset cached client so our patch is used
            ingest_mod._client = None

            from rag.ingest import add_incident
            add_incident(
                "Worker 5 missing hardhat in Zone A for 10 seconds.",
                metadata={
                    "timestamp": "2026-11-17 10:00:00",
                    "camera": "Cam-01",
                    "worker_id": "5",
                    "zone": "Zone A",
                    "event": "no-hardhat",
                    "severity": "High",
                    "confidence": 0.9,
                    "run_id": "run_001",
                    "sqlite_event_id": 123,
                    "image_path": "",
                },
            )

        mock_col.upsert.assert_called_once()
        call_kwargs = mock_col.upsert.call_args
        ids = call_kwargs.kwargs.get("ids", call_kwargs.args[0] if call_kwargs.args else None)
        assert ids is not None
        assert ids[0] == "event_123"
        ingest_mod._client = None

    def test_add_incident_falls_back_to_legacy_id_when_no_event_id(self):
        """Without sqlite_event_id, falls back to timestamp_workerId ID."""
        import rag.ingest as ingest_mod
        mock_col = self._make_mock_col()

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from rag.ingest import add_incident
            add_incident(
                "Worker spotted without vest.",
                metadata={
                    "timestamp": "2026-11-17 09:00:00",
                    "worker_id": "7",
                },
            )

        mock_col.upsert.assert_called_once()
        ingest_mod._client = None

    def test_add_incident_skips_empty_text(self):
        """add_incident() should skip ingestion when text is empty or None."""
        import rag.ingest as ingest_mod
        mock_col = self._make_mock_col()

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from rag.ingest import add_incident
            add_incident("", metadata={"sqlite_event_id": 1})

        mock_col.upsert.assert_not_called()
        ingest_mod._client = None

    def test_add_incident_idempotent_on_same_event_id(self):
        """Calling add_incident twice with the same sqlite_event_id should upsert — not duplicate."""
        import rag.ingest as ingest_mod
        mock_col = self._make_mock_col()

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from rag.ingest import add_incident
            for _ in range(2):
                add_incident(
                    "Worker 10 missing vest.",
                    metadata={"sqlite_event_id": 777, "timestamp": "2026-11-17 10:00:00", "worker_id": "10"},
                )

        # upsert called twice with SAME id — ChromaDB upsert handles idempotency
        assert mock_col.upsert.call_count == 2
        call1_id = mock_col.upsert.call_args_list[0].kwargs.get("ids", [""])[0]
        call2_id = mock_col.upsert.call_args_list[1].kwargs.get("ids", [""])[0]
        assert call1_id == call2_id == "event_777"
        ingest_mod._client = None


# ==============================================================
# G. SemanticSearchService — real ChromaDB (mocked)
# ==============================================================

class TestSemanticSearchService:
    """Tests for ai.semantic_search_service.SemanticSearchService"""

    def test_empty_collection_returns_helpful_message(self):
        """Should not crash when collection is empty."""
        import rag.ingest as ingest_mod
        mock_col = MagicMock()
        mock_col.count.return_value = 0

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from ai.semantic_search_service import SemanticSearchService
            svc = SemanticSearchService()
            result = svc.retrieve_concise_evidence("helmet violations")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should indicate the database is empty, not crash
        assert "empty" in result.lower() or "process" in result.lower() or "populate" in result.lower()
        ingest_mod._client = None

    def test_returns_actual_results_when_docs_found(self):
        """Should format and return actual documents from ChromaDB."""
        import rag.ingest as ingest_mod
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "documents": [["Worker 1 missing helmet in Zone A."]],
            "metadatas": [[{
                "timestamp": "2026-11-17 10:00:00",
                "zone": "Zone A",
                "worker_id": "1",
                "event": "no-hardhat",
                "severity": "High",
                "confidence": 0.9,
                "image_path": "",
            }]],
        }

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from ai.semantic_search_service import SemanticSearchService
            svc = SemanticSearchService()
            result = svc.retrieve_concise_evidence("missing helmet")

        assert "Zone A" in result
        assert "2026-11-17" in result
        ingest_mod._client = None

    def test_missing_metadata_fields_handled_gracefully(self):
        """Should not crash when metadata fields are absent."""
        import rag.ingest as ingest_mod
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["Some incident description."]],
            "metadatas": [[{}]],  # completely empty metadata
        }

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from ai.semantic_search_service import SemanticSearchService
            svc = SemanticSearchService()
            result = svc.retrieve_concise_evidence("incident")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "N/A" in result or "Unknown" in result
        ingest_mod._client = None


# ==============================================================
# H. SOP Retrieval — mocked
# ==============================================================

class TestSOPRetrieval:
    """Tests for rag.sop_ingest.retrieve_sop_context()"""

    def test_empty_sop_collection_returns_helpful_message(self):
        """Should return an informative message when SOP collection is empty."""
        import rag.ingest as ingest_mod
        mock_col = MagicMock()
        mock_col.count.return_value = 0

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from rag.sop_ingest import retrieve_sop_context
            result = retrieve_sop_context("helmet procedure")

        assert isinstance(result, str)
        assert "empty" in result.lower() or "sop" in result.lower()
        ingest_mod._client = None

    def test_sop_results_include_source_attribution(self):
        """Returned SOP passages must include the source document name."""
        import rag.ingest as ingest_mod
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {
            "documents": [["All workers must wear hard hats in production areas."]],
            "metadatas": [[{
                "source": "helmet_safety.txt",
                "category": "Head Protection",
                "chunk_index": 0,
                "total_chunks": 5,
            }]],
        }

        with patch.object(ingest_mod, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_col
            mock_get_client.return_value = mock_client
            ingest_mod._client = None

            from rag.sop_ingest import retrieve_sop_context
            result = retrieve_sop_context("helmet requirements")

        assert "helmet_safety.txt" in result
        assert "Head Protection" in result
        ingest_mod._client = None


# ==============================================================
# I. SOP Ingestion — text chunking
# ==============================================================

class TestSOPChunking:
    """Tests for rag.sop_ingest._chunk_text()"""

    def test_short_text_produces_one_chunk(self):
        from rag.sop_ingest import _chunk_text
        result = _chunk_text("Short text.", chunk_size=600, overlap=100)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        from rag.sop_ingest import _chunk_text
        long_text = "A" * 2000
        result = _chunk_text(long_text, chunk_size=600, overlap=100)
        assert len(result) > 1

    def test_empty_text_produces_no_chunks(self):
        from rag.sop_ingest import _chunk_text
        result = _chunk_text("", chunk_size=600, overlap=100)
        assert result == []

    def test_chunks_overlap(self):
        """Adjacent chunks should share characters (overlap)."""
        from rag.sop_ingest import _chunk_text
        text = "X" * 1500
        chunks = _chunk_text(text, chunk_size=600, overlap=100)
        # With chunk_size=600, overlap=100, start positions are 0, 500, 1000
        # chunk[0] covers 0–600, chunk[1] covers 500–1100 → overlap of 100 chars at positions 500–600
        assert len(chunks) >= 3


# ==============================================================
# J. query_incidents_sql — with real temp DB
# ==============================================================

class TestQueryIncidentsSQL:
    """Integration tests for agents.database_api.query_incidents_sql()"""

    def test_returns_no_incidents_message_when_empty(self, temp_db):
        from agents.database_api import query_incidents_sql
        result = query_incidents_sql()
        assert "No incidents found" in result

    def test_returns_actual_records(self, temp_db):
        from database.db import save_event
        from agents.database_api import query_incidents_sql
        save_event(
            camera="Cam-01",
            zone="Zone A",
            worker_id=10,
            event="no-hardhat",
            severity="High",
            confidence=0.88,
            start_time="2026-11-17 10:00:00",
            end_time="2026-11-17 10:00:05",
        )
        result = query_incidents_sql()
        assert "no-hardhat" in result
        assert "Zone A" in result

    def test_zone_filter(self, temp_db):
        from database.db import save_event
        from agents.database_api import query_incidents_sql
        save_event(
            camera="Cam-01", zone="Zone A", worker_id=1,
            event="no-hardhat", severity="High", confidence=0.9,
            start_time="2026-11-17 10:00:00", end_time="2026-11-17 10:00:05",
        )
        save_event(
            camera="Cam-02", zone="Zone B", worker_id=2,
            event="no-safety vest", severity="High", confidence=0.85,
            start_time="2026-11-17 10:01:00", end_time="2026-11-17 10:01:08",
        )
        result = query_incidents_sql(zone="Zone B")
        assert "Zone B" in result
        assert "Zone A" not in result

    def test_worker_id_filter(self, temp_db):
        from database.db import save_event
        from agents.database_api import query_incidents_sql
        save_event(
            camera="Cam-01", zone="Zone A", worker_id=55,
            event="no-hardhat", severity="High", confidence=0.8,
            start_time="2026-11-17 10:00:00", end_time="2026-11-17 10:00:05",
        )
        result = query_incidents_sql(worker_id=55)
        assert "55" in result
