"""score.py

Evaluation metrics calculation (Precision, Recall, F1, F0.5, Size Ratio)
and retrieval-vs-generation failure attribution against reference gold standards.
"""

from typing import Any


def _f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """Calculate F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def compute_metrics(generated_codes: set[str], gold_codes: set[str]) -> dict[str, float | int]:
    """
    Compute Precision, Recall, F1, F0.5, and Size Ratio for set of generated codes against gold standard.
    Note: Julian's rule — codes are never deduplicated before evaluation; set conversion happens here for metric math.
    """
    gold_count = len(gold_codes)
    if not generated_codes:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f05": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": gold_count,
            "generated_count": 0,
            "gold_count": gold_count,
            "size_ratio": 0.0,
        }

    true_positives = generated_codes & gold_codes
    false_positives = generated_codes - gold_codes
    false_negatives = gold_codes - generated_codes

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = _f_beta(precision, recall, 1.0)
    f05 = _f_beta(precision, recall, 0.5)
    size_ratio = len(generated_codes) / gold_count if gold_count > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f05": round(f05, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "generated_count": len(generated_codes),
        "gold_count": gold_count,
        "size_ratio": round(size_ratio, 4),
    }


def evaluate_codelist(
    generated_list: list[dict[str, str]],
    gold_codes: set[str],
    retrieval_type: str = "rag",
    retrieved_codes: set[str] | None = None,
) -> dict[str, Any]:
    """
    Full evaluation of a generated codelist against a gold standard set,
    including false-negative failure attribution.
    """
    retrieved_set = retrieved_codes or set()
    generated_set = set(c["code"] for c in generated_list)

    metrics = compute_metrics(generated_set, gold_codes)
    false_negatives = gold_codes - generated_set

    attribution = []
    for code in false_negatives:
        if retrieval_type != "rag" or code in retrieved_set:
            failure_type = "generation_failure"
        else:
            failure_type = "retrieval_failure"

        attribution.append({
            "code": code,
            "failure_type": failure_type,
        })

    return {
        **metrics,
        "failure_attribution": attribution,
    }
