"""retrieve.py

Hybrid retrieval engine combining dense vector similarity (ChromaDB) and
lexical scoring (BM25) via Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path
import pickle
import chromadb
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """Performs hybrid dense + BM25 retrieval over a concept collection."""

    def __init__(
        self,
        chroma_path: str | Path,
        bm25_path: str | Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "snomed_concepts",
    ):
        self.chroma_path = Path(chroma_path)
        self.bm25_path = Path(bm25_path)

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.chroma_client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load BM25 index
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_codes = data["codes"]

    def retrieve_dense(self, query: str, n_results: int = 400) -> list[dict[str, str]]:
        """Retrieve semantically similar concepts from ChromaDB."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        # Cap n_results to collection count to prevent out-of-bounds error on small fixtures
        count = self.collection.count()
        actual_n = min(n_results, count)
        if actual_n == 0:
            return []

        results = self.collection.query(query_embeddings=query_embedding, n_results=actual_n)
        return [
            {"code": code, "term": term}
            for code, term in zip(results["ids"][0], results["documents"][0])
        ]

    def retrieve_bm25(self, query: str, n_results: int = 400) -> list[dict[str, str]]:
        """Retrieve concepts using BM25 lexical matching."""
        if not self.bm25_codes:
            return []

        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)

        code_score_pairs = list(zip(self.bm25_codes, scores))
        code_score_pairs.sort(key=lambda x: x[1], reverse=True)

        actual_n = min(n_results, len(code_score_pairs))
        top_codes = [code for code, _ in code_score_pairs[:actual_n]]

        if not top_codes:
            return []

        results = self.collection.get(ids=top_codes)
        return [
            {"code": code, "term": term}
            for code, term in zip(results["ids"], results["documents"])
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[dict[str, str]],
        bm25_results: list[dict[str, str]],
        k: int = 60,
    ) -> list[dict[str, str]]:
        """Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        code_to_term: dict[str, str] = {}

        for rank, item in enumerate(dense_results):
            code = item["code"]
            scores[code] = scores.get(code, 0.0) + 1.0 / (k + rank)
            code_to_term[code] = item["term"]

        for rank, item in enumerate(bm25_results):
            code = item["code"]
            scores[code] = scores.get(code, 0.0) + 1.0 / (k + rank)
            code_to_term[code] = item["term"]

        ranked_codes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"code": code, "term": code_to_term[code]} for code, _ in ranked_codes]

    def retrieve_hybrid(
        self, query: str, n_results: int = 400, k: int = 60
    ) -> list[dict[str, str]]:
        """Execute hybrid retrieval (dense + BM25) fused with RRF."""
        dense_res = self.retrieve_dense(query, n_results=n_results)
        bm25_res = self.retrieve_bm25(query, n_results=n_results)
        fused = self.reciprocal_rank_fusion(dense_res, bm25_res, k=k)
        return fused[:n_results]
