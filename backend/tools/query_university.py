"""University query tool: uses KnowledgeBase (PostgreSQL) for data access."""

from backend.services.knowledge_base import KnowledgeBase


def query_university(
    name: str | None = None,
    province: str | None = None,
    tier: str | None = None,
    min_score: int | None = None,
) -> list[dict]:
    """Query universities with optional filters.

    Args:
        name: Fuzzy match on university name.
        province: Exact match on province.
        tier: Exact match on tier (e.g. "985", "211", "双一流").
        min_score: Minimum min_score_2025 threshold.

    Returns:
        List of matching universities with their data.
    """
    kb = KnowledgeBase()
    universities = kb.all_universities
    results = []

    for uni_name, data in universities.items():
        if name is not None and name not in uni_name:
            continue
        if province is not None and data.get("province") != province:
            continue
        if tier is not None and data.get("tier") != tier:
            continue
        if min_score is not None and data.get("min_score_2025", 0) < min_score:
            continue

        results.append({"name": uni_name, **data})

    return results
