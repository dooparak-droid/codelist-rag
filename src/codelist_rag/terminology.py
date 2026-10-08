"""terminology.py

Handles loading, querying, and verifying SNOMED-CT concept codes and terms.
Supports loading from raw RF2 files, preprocessed CSVs, or synthetic test JSON files.
"""

from pathlib import Path
import json
import pandas as pd


class TerminologyStore:
    """In-memory or queryable container for terminology concept IDs and terms."""

    def __init__(self, concepts: dict[str, str] | None = None):
        """Initialise with a dictionary mapping concept code -> term."""
        self._concepts = concepts or {}

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, code: str) -> bool:
        return code in self._concepts

    @classmethod
    def from_json(cls, json_path: str | Path) -> "TerminologyStore":
        """Load concepts from a JSON file (list of dicts with 'code' and 'term')."""
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        concepts = {str(item["code"]): str(item["term"]) for item in data}
        return cls(concepts)

    @classmethod
    def from_csv(cls, csv_path: str | Path, code_col: str = "code", term_col: str = "term") -> "TerminologyStore":
        """Load concepts from a CSV file."""
        df = pd.read_csv(csv_path, dtype={code_col: str, term_col: str})
        concepts = dict(zip(df[code_col], df[term_col]))
        return cls(concepts)

    @classmethod
    def from_rf2(cls, description_files: str | Path | list[str | Path]) -> "TerminologyStore":
        """
        Load concepts from one or more RF2 Description Snapshot files, as in the thesis
        corpus build: keep descriptions whose own active flag is 1, keep synonyms only
        (type 900000000000013009, which excludes fully specified names), and keep the
        first synonym found for each concept. Files are read in the order given, so
        list the International file before any national extension.

        The filter is on the description's active flag, not the concept's. A concept
        retired from SNOMED-CT can therefore remain in the corpus while one of its
        descriptions is still marked active.
        """
        if isinstance(description_files, (str, Path)):
            description_files = [description_files]

        concepts: dict[str, str] = {}
        for file in description_files:
            desc_df = pd.read_csv(
                file,
                sep="\t",
                dtype={"conceptId": str, "term": str, "typeId": str, "active": int},
                usecols=["conceptId", "term", "typeId", "active"],
                quoting=3,
            )
            kept = desc_df[(desc_df["active"] == 1) & (desc_df["typeId"] == "900000000000013009")]
            for concept_id, term in zip(kept["conceptId"], kept["term"]):
                concepts.setdefault(concept_id, term)
        return cls(concepts)

    @classmethod
    def from_chroma(cls, collection, page_size: int = 50000) -> "TerminologyStore":
        """
        Load the concept IDs held in a ChromaDB collection, so that the
        fabrication check uses exactly the terminology the retriever searches.
        Only IDs are read; terms are left empty to keep memory use small.
        """
        concepts: dict[str, str] = {}
        offset = 0
        while True:
            page = collection.get(include=[], limit=page_size, offset=offset)
            ids = page["ids"]
            if not ids:
                break
            concepts.update({str(i): "" for i in ids})
            offset += len(ids)
        return cls(concepts)

    def get_term(self, code: str) -> str | None:
        """Get description term for a given code."""
        return self._concepts.get(str(code))

    def contains_code(self, code: str) -> bool:
        """Check if a code exists in the loaded terminology."""
        return str(code) in self._concepts

    def validate_codes(self, codes: list[str]) -> list[dict[str, str | bool]]:
        """
        Check a list of codes against the terminology.
        Returns a list of dicts: [{"code": code, "is_real": bool, "term": term | None}]
        """
        results = []
        for code in codes:
            code_str = str(code)
            is_real = code_str in self._concepts
            results.append({
                "code": code_str,
                "is_real": is_real,
                "term": self._concepts.get(code_str),
            })
        return results

    def get_all_concepts(self) -> list[tuple[str, str]]:
        """Return all (code, term) pairs."""
        return list(self._concepts.items())
