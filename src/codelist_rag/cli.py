"""cli.py

Command Line Interface (CLI) for index building, codelist generation, and benchmark evaluation.
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from codelist_rag.generate import CodelistPipeline
from codelist_rag.index import IndexBuilder
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.score import evaluate_codelist
from codelist_rag.terminology import TerminologyStore


def build_index_cmd(args: argparse.Namespace) -> None:
    """Build ChromaDB and BM25 indexes from a terminology dataset."""
    print("Loading terminology...")
    if args.terminology_json:
        store = TerminologyStore.from_json(args.terminology_json)
    elif args.csv:
        store = TerminologyStore.from_csv(args.csv, code_col=args.code_col, term_col=args.term_col)
    elif args.rf2_concept and args.rf2_description:
        store = TerminologyStore.from_rf2(args.rf2_concept, args.rf2_description)
    else:
        print("Error: Must specify --terminology-json, --csv, or both --rf2-concept and --rf2-description")
        sys.exit(1)

    print(f"Loaded {len(store)} concepts. Building indexes...")
    builder = IndexBuilder(embedding_model_name=args.embedding_model)
    builder.build_indexes(store, chroma_path=args.chroma_path, bm25_output_path=args.bm25_path)
    print(f"Index build complete!\n  ChromaDB: {args.chroma_path}\n  BM25: {args.bm25_path}")


def generate_cmd(args: argparse.Namespace) -> None:
    """Generate a SNOMED-CT codelist for a clinical condition."""
    print(f"Generating codelist for: '{args.condition}'...")

    retriever = None
    if not args.no_rag:
        if not Path(args.chroma_path).exists() or not Path(args.bm25_path).exists():
            print(f"Error: Index paths not found ({args.chroma_path}, {args.bm25_path}). Run build-index first.")
            sys.exit(1)
        retriever = HybridRetriever(
            chroma_path=args.chroma_path,
            bm25_path=args.bm25_path,
            embedding_model_name=args.embedding_model,
        )

    terminology = None
    if args.terminology_json and Path(args.terminology_json).exists():
        terminology = TerminologyStore.from_json(args.terminology_json)

    pipeline = CodelistPipeline(retriever=retriever, terminology=terminology)
    result = pipeline.generate(
        query=args.condition,
        strategy=args.strategy,
        use_rag=not args.no_rag,
        n_results=args.n_results,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    if not result["success"]:
        print(f"Generation failed: {result['error']}")
        sys.exit(1)

    print(f"\nSuccessfully generated {result['total_codes']} codes (Fabrication rate: {result['fabrication_rate']}):")
    print(json.dumps(result["codes"], indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full result to {args.output}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Evaluate a generated codelist against a gold standard reference CSV."""
    with open(args.generated, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    codes_list = gen_data["codes"] if isinstance(gen_data, dict) else gen_data
    retrieved_codes = set(gen_data.get("retrieved_codes", [])) if isinstance(gen_data, dict) else set()
    use_rag = gen_data.get("use_rag", True) if isinstance(gen_data, dict) else True

    gold_df = pd.read_csv(args.gold_standard, dtype={args.code_col: str})
    gold_codes = set(gold_df[args.code_col])

    results = evaluate_codelist(
        generated_list=codes_list,
        gold_codes=gold_codes,
        retrieval_type="rag" if use_rag else "non_rag",
        retrieved_codes=retrieved_codes,
    )

    print(f"\n--- Evaluation Results for {args.generated} ---")
    print(f"  Precision:       {results['precision']}")
    print(f"  Recall:          {results['recall']}")
    print(f"  F1 score:        {results['f1']}")
    print(f"  F0.5 score:      {results['f05']}")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  False Negatives: {results['false_negatives']}")
    print(f"  Size Ratio:      {results['size_ratio']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codelist-rag",
        description="Retrieval-augmented clinical codelist generation for SNOMED-CT",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build Index
    build_p = subparsers.add_parser("build-index", help="Build ChromaDB and BM25 retrieval indexes")
    build_p.add_argument("--terminology-json", type=str, help="Path to synthetic or JSON concepts file")
    build_p.add_argument("--csv", type=str, help="Path to clean descriptions CSV")
    build_p.add_argument("--rf2-concept", type=str, help="Path to RF2 Concept Snapshot file")
    build_p.add_argument("--rf2-description", type=str, help="Path to RF2 Description Snapshot file")
    build_p.add_argument("--code-col", type=str, default="code", help="Code column in CSV")
    build_p.add_argument("--term-col", type=str, default="term", help="Term column in CSV")
    build_p.add_argument("--chroma-path", type=str, default="./data/chroma_db", help="ChromaDB storage dir")
    build_p.add_argument("--bm25-path", type=str, default="./data/bm25_index.pkl", help="BM25 index path")
    build_p.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    build_p.set_defaults(func=build_index_cmd)

    # Generate
    gen_p = subparsers.add_parser("generate", help="Generate a SNOMED-CT codelist")
    gen_p.add_argument("condition", type=str, help="Clinical condition name (e.g. 'Asthma')")
    gen_p.add_argument("--strategy", type=str, choices=["zero_shot", "few_shot", "chain_of_thought"], default="zero_shot")
    gen_p.add_argument("--no-rag", action="store_true", help="Disable retrieval-augmented generation")
    gen_p.add_argument("--n-results", type=int, default=400, help="Number of retrieved candidate concepts")
    gen_p.add_argument("--provider", type=str, default="openai", choices=["openai", "google", "anthropic", "ollama"])
    gen_p.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model identifier")
    gen_p.add_argument("--base-url", type=str, default=None, help="Base URL for custom/Ollama endpoint")
    gen_p.add_argument("--chroma-path", type=str, default="./data/chroma_db", help="ChromaDB storage dir")
    gen_p.add_argument("--bm25-path", type=str, default="./data/bm25_index.pkl", help="BM25 index path")
    gen_p.add_argument("--terminology-json", type=str, default=None, help="JSON file for verification check")
    gen_p.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    gen_p.add_argument("--output", "-o", type=str, help="Path to save result JSON")
    gen_p.set_defaults(func=generate_cmd)

    # Evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a generated codelist against a gold standard CSV")
    eval_p.add_argument("--generated", type=str, required=True, help="Path to generated codelist JSON")
    eval_p.add_argument("--gold-standard", type=str, required=True, help="Path to reference gold standard CSV")
    eval_p.add_argument("--code-col", type=str, default="code", help="Code column in gold standard CSV")
    eval_p.add_argument("--output", "-o", type=str, help="Path to save evaluation result JSON")
    eval_p.set_defaults(func=evaluate_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
