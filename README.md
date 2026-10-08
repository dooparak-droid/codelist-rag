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
                     │  LLM Generation (Inspect AI) │
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
- **Fabrication falls with retrieval**: A code is fabricated if its identifier does not exist in SNOMED-CT. Among the false-positive codes (those not in the reference set), more than half were fabricated without retrieval for six of the seven models. With retrieval the share fell for every model, to 1.1% or less for four models and between 14.4% and 17.0% for GPT-4o and the two MedGemma models. See the table below.
- **Structural failure attribution**: In Coronary Heart Disease, categories A (anatomical structures), C, and P had 0% recall under every retrieval strategy in all seven models. Every missed code in categories A and C was never returned by the search (retrieval failure), so the model never had the chance to select it. This points to a limit of the retrieval configuration used (all-MiniLM-L6-v2 embeddings plus BM25), not of the models.

### Model Performance and Fabrication Summary

| Model | F1, best strategy per condition (retrieval) | F1, mean over all combinations | Fabricated share of false positives, without retrieval | Fabricated share of false positives, with retrieval |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-5.5** | 0.561 | 0.369 | 14.0% | 0.2% |
| **Gemini 3.1 Flash-Lite** | 0.536 | 0.257 | 79.8% | 0.2% |
| **Gemini 3.1 Pro** | 0.505 | 0.277 | 75.8% | 0.1% |
| **GPT-5.4-mini** | 0.469 | 0.257 | 59.9% | 1.1% |
| **GPT-4o (v1 baseline)** | 0.417 | 0.220 | 66.7% | 15.6% |
| **MedGemma 1.0-27B** | 0.305 | 0.120 | 90.9% | 14.4% |
| **MedGemma 1.5-4B** | 0.210 | 0.078 | 91.3% | 17.0% |

*Notes: Two F1 means are reported per model. The first picks the best-performing retrieval strategy for each condition, which is optimistic because it assumes that choice is known in advance. The second averages every condition, strategy and retrieval combination. Fabrication shares are computed over false-positive codes only, across the seven conditions, and pooled over strategies and repetitions. Scores are against NHS Digital reference sets. They were produced by the thesis evaluation harness in `evaluation/`, not by this service.*

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- An API key for OpenAI (`OPENAI_API_KEY`) or Google (`GOOGLE_API_KEY`), or a local Ollama instance for keyless execution. Google models also need `pip install -e ".[google]"`.

### Local Installation

```bash
git clone <URL of this repository>
cd codelist-rag
pip install -e .
```

---

## Step 1: Building the SNOMED-CT Index

Due to licensing terms, SNOMED-CT files cannot be distributed in a public code repository. UK researchers can obtain SNOMED-CT free of charge via NHS England's **TRUD (Technology Reference data Update Distribution)** portal.

To build the hybrid retrieval index from the Description Snapshot files in your downloaded TRUD release (International file first, then any UK extension file):

```bash
codelist-rag build-index \
  --rf2-description /path/to/sct2_Description_Snapshot-en_INT.txt /path/to/sct2_Description_Snapshot-en_GB.txt \
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
  --model gpt-5.5 \
  --chroma-path ./data/chroma_db \
  --bm25-path ./data/bm25_index.pkl \
  --output asthma_codelist.json
```

### Example Output

This is a real response from `POST /codelist` for "Atrial fibrillation", using the default settings (GPT-5.5, `zero_shot`, retrieval on) against an index built from the UK SNOMED-CT release of 6 May 2026. The model returned 54 codes and the first three are shown. All 54 were found in the index (`fabrication_rate` 0.0). The call took about a minute.

```json
{
  "condition": "Atrial fibrillation",
  "success": true,
  "error": null,
  "codes": [
    {
      "code": "49436004",
      "term": "Atrial fibrillation",
      "code_is_real": true
    },
    {
      "code": "155364009",
      "term": "Fibrillation - atrial",
      "code_is_real": true
    },
    {
      "code": "81216002",
      "term": "Atrial flutter-fibrillation",
      "code_is_real": true
    }
  ],
  "total_codes": 54,
  "fabrication_rate": 0.0,
  "n_retrieved": 400,
  "attempts": 1,
  "strategy": "zero_shot",
  "use_rag": true,
  "model": "openai:gpt-5.5",
  "version": "0.1.0",
  "disclaimer": "Research tool only. Codelists generated by AI must be reviewed and validated by a qualified clinician before use in electronic health record research."
}
```

The same settings for asthma (one request, scored against the NHS Digital reference set) gave precision 0.43, recall 0.68 and F1 0.52, close to the thesis results for that condition, which were F1 of 0.53, 0.52 and 0.58 over three repetitions. One request is not a replication, since output varies between calls.

### Choosing a model, strategy and retrieval

The default model is `gpt-5.5` through the OpenAI API, with the `zero_shot` prompt and retrieval switched on. Each can be changed per request (`provider`, `model`, `strategy`, `use_rag`) or per command (`--provider`, `--model`, `--strategy`, `--no-rag`). Strategies are `zero_shot`, `few_shot` and `chain_of_thought`. In the thesis, prompting strategy explained 0.1% of the variance in F1, so the choice mattered far less than whether retrieval was used. Turning retrieval off is possible but gave much lower scores and many more fabricated codes in every model tested.

### Cost

GPT-5.5 is a paid, hosted model, and it is the most expensive option here because it spends many tokens on internal reasoning before answering. In the thesis run (126 requests), GPT-5.5 used 230,788 input tokens and 783,045 output tokens, of which 551,126 were reasoning tokens. At OpenAI's published prices of $5 per million input tokens and $30 per million output tokens (checked October 2026, and subject to change), that is about $25 for the run, or roughly $0.20 per codelist. A single measured request through this service (asthma, retrieval on, `zero_shot`) used 5,263 input tokens and 7,204 output tokens, of which 3,456 were reasoning tokens. That is about $0.24 at the same prices. Cost varies with the condition, because a condition with a longer codelist produces more output. The `n_results` setting changes the prompt size, and the `chain_of_thought` strategy produces longer output.

Cheaper choices are `gpt-5.4-mini` ($0.75 and $4.50 per million input and output tokens), a Gemini model, or a local Ollama model that needs no key. In the thesis these scored lower than GPT-5.5 (see the table above), and the two MedGemma models scored lowest.

---

## Step 3: Running the Web API

Start the FastAPI prediction service:

```bash
uvicorn codelist_rag.api:app --host 0.0.0.0 --port 8000
```

Once running, interactive API documentation (Swagger UI) is available in your browser at `http://localhost:8000/docs`.

The service reads these environment variables:

| Variable | Purpose |
| :--- | :--- |
| `CHROMA_PATH`, `BM25_PATH` | Locations of the built indexes (defaults `./data/chroma_db` and `./data/bm25_index.pkl`) |
| `OPENAI_API_KEY`, `GOOGLE_API_KEY` | Key for the chosen provider. A request without the key returns HTTP 503 |
| `TERMINOLOGY_RELEASE` | Free text naming the SNOMED-CT release the index was built from, shown by `GET /health` |
| `TERMINOLOGY_JSON` | Optional concept file, used for the fabrication check only when no index is loaded |

When an index is loaded, every returned code is checked against the concept identifiers in that index. `GET /health` reports the version, default model, release label and number of concepts loaded.

If the model's reply cannot be parsed after three attempts, `POST /codelist` returns HTTP 502 with the parse error. It never returns an empty list as a success.

### Running in Docker

```bash
docker build -t codelist-rag .

docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY="your-api-key" \
  -e TERMINOLOGY_RELEASE="UK Edition 2026-05-06" \
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
4. **Not independently validated**: The evaluation figures come from the thesis harness. The service uses the same prompts, retrieval settings and model interface, but its own output has not been scored against the reference sets. Scores for the codelists you generate depend on your SNOMED-CT release and on the model.
5. **Fabrication check scope**: A code is flagged as fabricated if its identifier is not among the concepts in the index you built. As in the thesis, the index keeps concepts that have at least one active synonym, so a concept retired from SNOMED-CT can still be present. A retired concept is therefore not flagged as fabricated.
6. **SNOMED-CT Only**: This tool is designed strictly for SNOMED-CT concept terminology and does not generate ICD-10 or Read v2 codelists.
