"""Industry query tool: uses KnowledgeBase (PostgreSQL) for data access."""

from backend.services.knowledge_base import KnowledgeBase


def query_industry(
    name: str | None = None,
    entry_barrier: str | None = None,
) -> list[dict]:
    """Query industries with optional filters.

    Args:
        name: Fuzzy match on industry name.
        entry_barrier: Exact match on entry_barrier (e.g. "high", "medium", "low").

    Returns:
        List of matching industries with their data.
    """
    kb = KnowledgeBase()
    all_industries = kb.all_industries
    results = []

    for ind_name, data in all_industries.items():
        if name is not None and name not in ind_name:
            continue
        if entry_barrier is not None and data.get("entry_barrier") != entry_barrier:
            continue

        results.append({"name": ind_name, **data})

    return results
