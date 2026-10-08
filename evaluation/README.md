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
0. Install the evaluation extras: `pip install -e ".[eval]"`. Run from the repository root, since `pipeline_core.py` uses relative data paths.
1. Build the indexes at the paths `pipeline_core.py` expects: `data/snomed/chroma_db` and `data/snomed/bm25_index.pkl` (`codelist-rag build-index --chroma-path data/snomed/chroma_db --bm25-path data/snomed/bm25_index.pkl ...`).
2. Place the categorised NHS Digital reference set CSVs (file names are listed in `GOLD_STANDARD_FILES` in `pipeline_core.py`) in `data/categorised_gold_standards/`. These are not in the repository.
3. Set your provider API key (`OPENAI_API_KEY` or `GOOGLE_API_KEY`) or configure Ollama for local models.
4. Execute `inspect_pipeline.py`:
   ```bash
   python evaluation/inspect_pipeline.py
   ```
