"""test_score.py

Unit tests for precision, recall, F1, F0.5, size ratio, and failure attribution calculations.
"""

from codelist_rag.score import compute_metrics, evaluate_codelist


def test_compute_metrics_perfect():
    gen = {"99900001", "99900002"}
    gold = {"99900001", "99900002"}
    res = compute_metrics(gen, gold)

    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["f05"] == 1.0
    assert res["true_positives"] == 2
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["size_ratio"] == 1.0


def test_compute_metrics_partial():
    gen = {"99900001", "88800001"}  # 1 TP, 1 FP
    gold = {"99900001", "99900002"}  # 1 TP, 1 FN
    res = compute_metrics(gen, gold)

    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1


def test_evaluate_codelist_failure_attribution():
    generated = [{"code": "99900001", "term": "Concept 1"}]
    gold = {"99900001", "99900002", "99900003"}
    retrieved = {"99900001", "99900002"}  # 99900002 was retrieved but model missed it; 99900003 was never retrieved

    eval_res = evaluate_codelist(
        generated_list=generated,
        gold_codes=gold,
        retrieval_type="rag",
        retrieved_codes=retrieved,
    )

    attr = {x["code"]: x["failure_type"] for x in eval_res["failure_attribution"]}
    assert attr["99900002"] == "generation_failure"
    assert attr["99900003"] == "retrieval_failure"
