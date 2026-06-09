import json
from pathlib import Path
from typing import Optional


_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "universities.json"


def _load_universities() -> dict:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def query_university(
    name: Optional[str] = None,
    province: Optional[str] = None,
    tier: Optional[str] = None,
    min_score: Optional[int] = None,
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
    universities = _load_universities()
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
