"""test_validate.py

Unit tests for concept verification and fabrication rate checking.
"""

from codelist_rag.terminology import TerminologyStore
from codelist_rag.validate import check_fabrication


def test_check_fabrication():
    fixture_data = {
        "99900001": "Synthetic mild airway reactivity",
        "99900002": "Fictitious acute bronchospasm",
    }
    store = TerminologyStore(fixture_data)

    generated = [
        {"code": "99900001", "term": "Synthetic mild airway reactivity"},  # Real
        {"code": "88800001", "term": "Invented code"},                     # Fabricated
    ]

    res = check_fabrication(generated, store)

    assert res["total_generated"] == 2
    assert res["real_count"] == 1
    assert res["fabricated_count"] == 1
    assert res["fabrication_rate"] == 0.5

    annotated = res["annotated_codes"]
    assert annotated[0]["code_is_real"] is True
    assert annotated[1]["code_is_real"] is False
