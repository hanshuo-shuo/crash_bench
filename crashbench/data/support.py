"""Source support counts: labels can overlap within a physical source."""
from typing import Iterable, Mapping


def source_label_support(labels: Iterable[bool]) -> dict[str, bool]:
    values = list(labels)
    if any(value not in (False, True, 0, 1) for value in values):
        raise ValueError("benefit labels must be binary")
    b0, b1 = any(not value for value in values), any(values)
    return {"contains_B0": b0, "contains_B1": b1,
            "contains_both": b0 and b1, "entirely_B0": b0 and not b1}


def aggregate_source_support(rows: Iterable[Mapping]) -> dict[str, int | str]:
    rows = list(rows)
    counts = {key: sum(bool(row[key]) for row in rows)
              for key in ("contains_B0", "contains_B1", "contains_both", "entirely_B0")}
    return {**counts, "benefit_zero_sources": counts["contains_B0"],
            "benefit_one_sources": counts["contains_B1"],
            "source_support_definition": "contains_label_v2_nonexclusive"}
