"""test_retrieve.py

Unit tests for Reciprocal Rank Fusion (RRF) logic.
"""

from codelist_rag.retrieve import HybridRetriever


def test_reciprocal_rank_fusion():
    dense = [
        {"code": "A", "term": "Concept A"},
        {"code": "B", "term": "Concept B"},
    ]
    bm25 = [
        {"code": "B", "term": "Concept B"},
        {"code": "C", "term": "Concept C"},
    ]

    fused = HybridRetriever.reciprocal_rank_fusion(dense, bm25, k=60)
    codes = [x["code"] for x in fused]

    # Concept B appears in both lists, so its fused score should place it first
    assert codes[0] == "B"
    assert set(codes) == {"A", "B", "C"}
