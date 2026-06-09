import json
from pathlib import Path
from typing import Optional


_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "industries.json"


def _load_industries() -> dict:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def query_industry(
    name: Optional[str] = None,
    entry_barrier: Optional[str] = None,
) -> list[dict]:
    """Query industries with optional filters.

    Args:
        name: Fuzzy match on industry name.
        entry_barrier: Exact match on entry_barrier (e.g. "high", "medium", "low").

    Returns:
        List of matching industries with their data.
    """
    industries = _load_industries()
    results = []

    for ind_name, data in industries.items():
        if name is not None and name not in ind_name:
            continue
        if entry_barrier is not None and data.get("entry_barrier") != entry_barrier:
            continue

        results.append({"name": ind_name, **data})

    return results
