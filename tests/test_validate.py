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


def test_from_rf2_keeps_active_synonyms_only(tmp_path):
    header = "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n"
    rows = [
        "1\t20260201\t1\t0\t100\ten\t900000000000013009\tHeart attack\t0\n",
        "2\t20260201\t1\t0\t100\ten\t900000000000003001\tMyocardial infarction (disorder)\t0\n",  # FSN, excluded
        "3\t20260201\t0\t0\t200\ten\t900000000000013009\tRetired term\t0\n",  # inactive description, excluded
        "4\t20260201\t1\t0\t300\ten\t900000000000013009\tAsthma\t0\n",
    ]
    f = tmp_path / "desc.txt"
    f.write_text(header + "".join(rows))
    store = TerminologyStore.from_rf2(f)
    assert store.contains_code("100") and store.contains_code("300")
    assert not store.contains_code("200")
    assert store.get_term("100") == "Heart attack"
