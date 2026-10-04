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
    def from_rf2(cls, concept_file: str | Path, description_file: str | Path) -> "TerminologyStore":
        """
        Load active concepts and active descriptions from raw SNOMED-CT RF2 files.
        Only keeps entries where active == 1.
        """
        concept_df = pd.read_csv(concept_file, sep="\t", dtype={"id": str, "active": int})
        active_concept_ids = set(concept_df[concept_df["active"] == 1]["id"])

        desc_df = pd.read_csv(
            description_file,
            sep="\t",
            dtype={"conceptId": str, "term": str, "active": int},
            usecols=["conceptId", "term", "active"],
        )
        active_desc = desc_df[(desc_df["active"] == 1) & (desc_df["conceptId"].isin(active_concept_ids))]

        # Pick one term per conceptId (e.g. first active description)
        concepts = active_desc.groupby("conceptId")["term"].first().to_dict()
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
