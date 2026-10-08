"""test_pipeline_regression.py

End-to-end regression test on the synthetic fixture: build indexes, retrieve,
generate with a replaced model call, validate and score. The expected values
are worked by hand from the fixture files.
"""

import csv
from pathlib import Path

from codelist_rag import generate
from codelist_rag.generate import CodelistPipeline
from codelist_rag.index import IndexBuilder
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.score import evaluate_codelist
from codelist_rag.terminology import TerminologyStore

FIXTURES = Path(__file__).parent / "fixtures"

MODEL_OUTPUT = """[
  {"code": "99900001", "term": "Synthetic mild airway reactivity"},
  {"code": "99900004", "term": "Synthetic exercise-induced asthma"},
  {"code": "99900005", "term": "Fictitious allergic rhinitis"},
  {"code": "12345678", "term": "Invented concept"}
]"""


def test_full_pipeline_on_fixture(tmp_path, monkeypatch):
    store = TerminologyStore.from_json(FIXTURES / "synthetic_terminology.json")
    chroma, bm25 = tmp_path / "chroma", tmp_path / "bm25.pkl"
    IndexBuilder().build_indexes(store, chroma_path=chroma, bm25_output_path=bm25)
    retriever = HybridRetriever(chroma_path=chroma, bm25_path=bm25)

    monkeypatch.setattr(generate, "generate_text", lambda *a, **k: MODEL_OUTPUT)
    result = CodelistPipeline(retriever=retriever, terminology=store).generate(
        "asthma", n_results=10
    )

    assert result["success"] is True
    assert [c["code"] for c in result["codes"]] == ["99900001", "99900004", "99900005", "12345678"]
    assert [c["code_is_real"] for c in result["codes"]] == [True, True, True, False]
    assert result["fabrication_rate"] == 0.25
    assert result["n_retrieved"] == 10

    with open(FIXTURES / "synthetic_gold_standard.csv") as f:
        gold = {row["code"] for row in csv.DictReader(f)}
    score = evaluate_codelist(
        result["codes"], gold, retrieval_type="rag", retrieved_codes=set(result["retrieved_codes"])
    )
    # 2 of 4 generated codes are in the 3-code gold standard
    assert score["true_positives"] == 2
    assert score["precision"] == 0.5
    assert score["recall"] == 0.6667
    assert score["f1"] == 0.5714
