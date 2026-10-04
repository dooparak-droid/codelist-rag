"""validate.py

Fabrication verification logic checking generated codes against the loaded terminology store.
"""

from typing import Any
from codelist_rag.terminology import TerminologyStore


def check_fabrication(
    codes: list[dict[str, str]],
    terminology: TerminologyStore,
) -> dict[str, Any]:
    """
    Annotate each code entry with whether it exists in the loaded terminology store.

    Args:
        codes: list of dicts with 'code' and 'term'
        terminology: TerminologyStore instance

    Returns:
        dict containing:
          - 'annotated_codes': list of dicts with added 'code_is_real' boolean
          - 'total_generated': int
          - 'real_count': int
          - 'fabricated_count': int
          - 'fabrication_rate': float (0.0 to 1.0)
    """
    annotated = []
    real_count = 0
    fabricated_count = 0

    for item in codes:
        code_str = str(item["code"])
        is_real = terminology.contains_code(code_str)
        if is_real:
            real_count += 1
        else:
            fabricated_count += 1

        annotated.append({
            "code": code_str,
            "term": item["term"],
            "code_is_real": is_real,
        })

    total = len(codes)
    rate = (fabricated_count / total) if total > 0 else 0.0

    return {
        "annotated_codes": annotated,
        "total_generated": total,
        "real_count": real_count,
        "fabricated_count": fabricated_count,
        "fabrication_rate": round(rate, 4),
    }
