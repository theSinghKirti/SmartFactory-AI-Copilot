# agents/tools.py
"""
LangChain tool definitions for the SmartFactory AI Copilot.

Registered tools:
  1. get_stats_tool               — real-time factory statistics from SQLite
  2. query_incidents_sql_tool     — structured SQL queries on the events table
  3. query_incidents_semantic_tool — semantic similarity search on ChromaDB incidents
  4. query_sop_tool               — semantic search on the Safety SOP knowledge base
"""

from langchain_core.tools import tool
from agents.database_api import get_stats, query_incidents_sql
from ai.semantic_search_service import SemanticSearchService
from rag.sop_ingest import retrieve_sop_context

# Instantiate once at module level
_semantic_svc = SemanticSearchService()


@tool
def get_stats_tool() -> dict:
    """
    Fetch real-time factory safety statistics from the SQLite incident database.
    Returns total incidents, today's count, counts by violation type, and counts
    by factory zone. Use this tool for dashboard-style questions like
    'how many incidents are there?' or 'show me the safety overview'.
    """
    return get_stats()


@tool
def query_incidents_sql_tool(
    zone: str = None,
    event: str = None,
    severity: str = None,
    worker_id: str = None,
    limit: int = 20,
) -> str:
    """
    Query the incident database using structured filters.
    Use for questions like:
      - 'Show incidents in Zone B'
      - 'List all no-helmet violations'
      - 'High severity incidents today'
      - 'Show incidents for worker 42'

    Parameters:
        zone:      Filter by factory zone (e.g. 'Zone A', 'Zone B').
        event:     Filter by violation type (e.g. 'no-hardhat', 'no-safety vest').
        severity:  Filter by severity ('High', 'Medium', 'Low').
        worker_id: Filter by worker ID (integer as string, e.g. '42').
        limit:     Maximum number of records to return (default 20).
    """
    # Convert worker_id string to int if numeric for correct DB matching
    wid = None
    if worker_id is not None:
        try:
            wid = int(str(worker_id).replace("worker_", "").replace("W", ""))
        except (ValueError, TypeError):
            wid = None

    return query_incidents_sql(
        zone=zone,
        event=event,
        severity=severity,
        worker_id=wid,
        limit=limit,
    )


@tool
def query_incidents_semantic_tool(query: str) -> str:
    """
    Search historical incident records using semantic similarity.
    Use for open-ended questions like:
      - 'Find incidents similar to this situation'
      - 'Historical context around Zone B violations'
      - 'What happened with worker 104?'
      - 'Explain previous PPE violations'

    Returns actual incident records stored in the vector database.
    If the database is empty, returns an informative message.
    """
    return _semantic_svc.retrieve_concise_evidence(query=query, limit=5)


@tool
def query_sop_tool(query: str) -> str:
    """
    Search the Safety SOP (Standard Operating Procedure) knowledge base
    for relevant safety guidelines, procedures, and corrective actions.
    Use for questions like:
      - 'What are the procedures for a missing helmet violation?'
      - 'What corrective actions apply to PPE non-compliance?'
      - 'What is the helmet replacement schedule?'
      - 'What PPE is required in the loading dock?'
      - 'What should supervisors do when a vest violation is detected?'

    Returns relevant passages from safety documentation with source attribution.
    Note: Retrieved passages are demonstration guidance, not official regulations.
    """
    return retrieve_sop_context(query=query, limit=4)
