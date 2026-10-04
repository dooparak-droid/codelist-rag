# Thesis Evaluation Suite (Inspect AI)

This directory contains the scientific evaluation harness used to generate the thesis benchmarks across 882 experimental runs.

## Overview

The evaluation suite uses **Inspect AI** (`inspect_pipeline.py`) to run factorial evaluations across seven clinical conditions, three prompting strategies (zero-shot, few-shot, chain-of-thought), two retrieval modes (RAG vs non-RAG), and three epochs.

### Key Components
- **Task & Solvers**: `codelist_task()` configures the evaluation task. The primary solver routes model calls through Inspect's native model engine (`USE_INSPECT_NATIVE_MODEL = True`).
- **Scorer**: `codelist_scorer()` parses output JSON, evaluates precision, recall, F1, and F0.5 against NHS Digital reference sets, and classifies missed codes into retrieval vs generation failures. Parse failures are scored as 0.0 rather than dropped.
- **Post-processing**: `derive_csvs()` generates structured metrics CSVs for per-epoch performance, pooled performance, code breakdown with real-code validation, and failure attribution.

## Running the Benchmarks

To run the evaluation suite on your own licensed SNOMED-CT release:
1. Ensure your local ChromaDB and BM25 indexes are built (`codelist-rag build-index`).
2. Place NHS Digital reference set CSV files in `data/categorised_gold_standards/`.
3. Set your provider API key (`OPENAI_API_KEY` or `GOOGLE_API_KEY`) or configure Ollama for local models.
4. Execute `inspect_pipeline.py`:
   ```bash
   python evaluation/inspect_pipeline.py
   ```
