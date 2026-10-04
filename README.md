# codelist-rag: Retrieval-Augmented SNOMED-CT Codelist Generation

`codelist-rag` is an open-source tool that turns LLM-assisted clinical codelist generation into a software pipeline that researchers can install, build, call, and evaluate.

Given a clinical condition name (such as "Type 2 Diabetes Mellitus" or "Asthma"), `codelist-rag` uses hybrid dense and lexical retrieval over SNOMED-CT, passes candidate concepts to a large language model, and validates every returned code to flag non-existent or fabricated identifiers.

---

## The Problem

Electronic health record (EHR) research relies on clinical codelists to define patient cohorts and outcomes. Today, codelists are compiled by hand. Researchers type terms into terminology browsers, navigate concept hierarchies, and manually record codes in spreadsheets. This process is slow, tedious, prone to human omission, and inconsistent across research teams.

While large language models can suggest codes, off-the-shelf LLMs frequently invent non-existent code numbers or return out-of-date concepts. `codelist-rag` solves this by pairing hybrid search over an active SNOMED-CT release with automated concept verification.

---

## How It Works

```
                     ┌──────────────────────────────┐
                     │ User Request ("Asthma")      │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  Hybrid Retrieval Engine     │
                     │  (ChromaDB + BM25 via RRF)   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼ Top 400 Concepts
                     ┌──────────────────────────────┐
                     │  LLM Generation (chatlas)    │
                     │  (Zero-shot / Few-shot / CoT)│
                     └──────────────┬───────────────┘
                                    │
                                    ▼ Candidate Codelist
                     ┌──────────────────────────────┐
                     │  Fabrication Verification    │
                     │  (Checks against SNOMED)     │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ Verified Codelist Output     │
                     └──────────────────────────────┘
```

1. **Retrieval**: Dense retrieval (`all-MiniLM-L6-v2` in ChromaDB) and lexical search (`rank-bm25`) find the top 400 candidate concepts from your local SNOMED-CT index, fused using Reciprocal Rank Fusion (RRF, k=60).
2. **Generation**: The LLM receives the candidate concepts and selects clinically relevant codes for the target condition.
3. **Verification**: Every code returned by the model is checked against the loaded terminology store and flagged as real or fabricated (`code_is_real`).

---

## Empirical Benchmark Results

The pipeline was evaluated across 882 experimental runs spanning seven frontier and open-weight language models, scored against **NHS Digital reference sets**.

### Key Scientific Findings

- **Retrieval significantly improves accuracy**: Retrieval improved F1 score across all seven conditions (median increase +0.27, p = 0.016).
- **Retrieval dominates performance**: Variance decomposition showed retrieval explained 35% of the variance in F1 score, whereas prompting strategy explained only 0.1%.
- **Fabrication is dramatically reduced**: Without retrieval, more than half of the codes produced by six of the seven models did not exist in SNOMED-CT. With retrieval, fabrication fell to 14% or less.
- **Structural failure attribution**: In Coronary Heart Disease, categories A (anatomical structures), C, and P exhibited near-total retrieval failure (0% recall across all RAG strategies across all seven models), proving that certain missed codes stem from terminology retrieval limits rather than model reasoning failures.

### Model Performance and Fabrication Summary

| Model | Best RAG Strategy F1 | Overall Mean F1 | Real Code Rate | Fabrication Rate |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-5.5** | 0.561 | 0.369 | 98.6% | 1.4% |
| **Gemini 3.1 Flash-Lite** | 0.536 | 0.257 | 83.1% | 16.9% |
| **Gemini 3.1 Pro** | 0.505 | 0.277 | 86.1% | 13.9% |
| **GPT-5.4-mini** | 0.469 | 0.257 | 89.0% | 11.0% |
| **GPT-4o (v1 baseline)** | 0.417 | 0.220 | 76.3% | 23.7% |
| **MedGemma 1.0-27B** | 0.305 | 0.120 | 63.0% | 37.0% |
| **MedGemma 1.5-4B** | 0.210 | 0.078 | 63.8% | 36.2% |

*Note: Two means are reported per model. Best RAG strategy mean reflects performance if the optimal prompting strategy is selected per condition. Overall mean reflects the average across all condition, strategy, and retrieval combinations.*

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- An API key for OpenAI (`OPENAI_API_KEY`) or Google (`GOOGLE_API_KEY`), or a local Ollama instance for keyless execution.

### Local Installation

```bash
git clone https://github.com/dooparak-droid/LLM-Assisted-Clinical-Codelist-Generation.git
cd codelist-rag
pip install -e .
```

---

## Step 1: Building the SNOMED-CT Index

Due to licensing terms, SNOMED-CT files cannot be distributed in a public code repository. UK researchers can obtain SNOMED-CT free of charge via NHS England's **TRUD (Technology Reference data Update Distribution)** portal.

To build the hybrid retrieval index from your downloaded TRUD zip or RF2 release:

```bash
codelist-rag build-index \
  --rf2-concept /path/to/sct2_Concept_Snapshot_INT.txt \
  --rf2-description /path/to/sct2_Description_Snapshot_INT.txt \
  --chroma-path ./data/chroma_db \
  --bm25-path ./data/bm25_index.pkl
```

For testing without licensed data, you can build an index from the synthetic test fixture:

```bash
codelist-rag build-index \
  --terminology-json tests/fixtures/synthetic_terminology.json \
  --chroma-path ./data/chroma_db \
  --bm25-path ./data/bm25_index.pkl
```

---

## Step 2: Generating a Codelist

### CLI Interface

Generate a codelist from the command line:

```bash
export OPENAI_API_KEY="your-api-key"

codelist-rag generate "Asthma" \
  --strategy zero_shot \
  --provider openai \
  --model gpt-4o-mini \
  --chroma-path ./data/chroma_db \
  --bm25-path ./data/bm25_index.pkl \
  --output asthma_codelist.json
```

### Example API Response

```json
{
  "condition": "Asthma",
  "success": true,
  "codes": [
    {
      "code": "195967001",
      "term": "Asthma",
      "code_is_real": true
    },
    {
      "code": "41807004",
      "term": "Childhood asthma",
      "code_is_real": true
    },
    {
      "code": "370221004",
      "term": "Exercise-induced asthma",
      "code_is_real": true
    }
  ],
  "total_codes": 3,
  "fabrication_rate": 0.0,
  "n_retrieved": 400,
  "strategy": "zero_shot",
  "use_rag": true,
  "model": "openai:gpt-4o-mini",
  "disclaimer": "Research tool only. Codelists generated by AI must be reviewed and validated by a qualified clinician before use in electronic health record research."
}
```

---

## Step 3: Running the Web API

Start the FastAPI prediction service:

```bash
uvicorn codelist_rag.api:app --host 0.0.0.0 --port 8000
```

Once running, interactive API documentation (Swagger UI) is available in your browser at `http://localhost:8000/docs`.

### Running in Docker

```bash
docker build -t codelist-rag .

docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY="your-api-key" \
  codelist-rag
```

---

## Step 4: Benchmarking Against a Gold Standard

If you have an existing reference codelist CSV (e.g. from OpenSAFELY or NHS Digital), evaluate your generated list:

```bash
codelist-rag evaluate \
  --generated asthma_codelist.json \
  --gold-standard refset_asthma.csv \
  --output evaluation_summary.json
```

---

## Limitations & Disclaimer

1. **Licensing**: SNOMED-CT is proprietary software licensed by SNOMED International and NHS Digital. Users must obtain their own licence via TRUD.
2. **Clinical Verification Required**: This software is a research productivity tool. All generated codelists must be reviewed and signed off by a qualified clinician before use in clinical trials or epidemiological cohort studies.
3. **Reference Set Scope**: Evaluation results reflect performance against NHS Digital Primary Care Domain reference sets for a fixed set of chronic conditions. Performance on other phenotypes may vary.
4. **SNOMED-CT Only**: This tool is designed strictly for SNOMED-CT concept terminology and does not generate ICD-10 or Read v2 codelists.
