"""index.py

Builds and persists ChromaDB vector collections and BM25 lexical indexes from a TerminologyStore.
"""

from pathlib import Path
import pickle
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from codelist_rag.terminology import TerminologyStore


class IndexBuilder:
    """Builds dense vector (ChromaDB) and lexical (BM25) indexes from concept data."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "snomed_concepts",
    ):
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name

    def build_indexes(
        self,
        terminology: TerminologyStore,
        chroma_path: str | Path,
        bm25_output_path: str | Path,
        batch_size: int = 5000,
    ) -> None:
        """Build both ChromaDB and BM25 indexes from a TerminologyStore."""
        chroma_path = Path(chroma_path)
        bm25_output_path = Path(bm25_output_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        bm25_output_path.parent.mkdir(parents=True, exist_ok=True)

        concepts = terminology.get_all_concepts()
        codes = [c[0] for c in concepts]
        terms = [c[1] for c in concepts]

        # 1. Build ChromaDB index
        client = chromadb.PersistentClient(path=str(chroma_path))
        # Remove existing collection if rebuilding
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        collection = client.create_collection(name=self.collection_name)

        model = SentenceTransformer(self.embedding_model_name)
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_terms = terms[i : i + batch_size]
            embeddings = model.encode(batch_terms, show_progress_bar=False).tolist()
            collection.add(
                ids=batch_codes,
                documents=batch_terms,
                embeddings=embeddings,
            )

        # 2. Build BM25 index
        tokenised_corpus = [term.lower().split() for term in terms]
        bm25 = BM25Okapi(tokenised_corpus)

        with open(bm25_output_path, "wb") as f:
            pickle.dump({"bm25": bm25, "codes": codes}, f)
