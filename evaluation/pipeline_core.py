# pipeline_core.py

"""
pipeline_core.py

Shared logic for the LLM-assisted SNOMED-CT codelist generation pipeline —
config, retrieval stack, prompt builders, gold standard loading, and the
provider-agnostic LLM call logic (chatlas). Extracted from experiment_runner.py
so that both experiment_runner.py (the original checkpointed runner) and
inspect_pipeline.py (the Inspect AI harness) import the same code instead of
one importing from the other.

This file is a pure move — no logic, signatures, or behavior were changed
from their original experiment_runner.py versions.

Orchestration (run_single_experiment, run_full_experiment, checkpointing,
the __main__ entry point) stays in experiment_runner.py, since that logic is
specific to the original JSON-file-based runner, not shared with
inspect_pipeline.py.
"""

# Standard library
import os  # for file paths
import json  # for JSON parsing
import re  # for extracting JSON out of raw model text

# Third party
import pandas as pd  # data wrangling
from dotenv import load_dotenv  # load API keys
import chatlas  # multi-provider LLM API
import chromadb  # vector store
from sentence_transformers import SentenceTransformer  # embedding model
import pickle  # for loading the BM25 index

# ── Configuration ──────────────────────────────────────────────
# To switch models/providers, change PROVIDER / MODEL / BASE_URL only —
# no call-site code needs to change. BASE_URL is only needed for
# self-hosted endpoints (e.g. Ollama, a RunPod vLLM/Ollama server).
PROVIDER = "openai"  # one of PROVIDER_CHAT_CLASSES below
MODEL = "gpt-4o-2024-08-06"
BASE_URL = None

# TEMPERATURE=0 was the original primary setting, but GPT-5.x hard-blocks any
# non-default temperature (reasoning models collapse their internal multi-path
# generation to a single greedy path at temp=0) and Gemini 3.x's own docs warn
# that temp=0 risks looping/degraded output on their reasoning-tier models too.
# So: provider-default temperature (no temperature param sent) is now the
# PRIMARY setting for every model, applied consistently so no model gets an
# asymmetric sampling regime. TEMPERATURE=0 is reserved for a separate
# sensitivity-analysis run — flip RUN_AT_TEMPERATURE_ZERO to True for that.
TEMPERATURE = 0
RUN_AT_TEMPERATURE_ZERO = False  # False = primary (default temp); True = temperature sensitivity analysis
MAX_TOKENS = 12000  # raised from 10000 (2026-07-30) — see Gemini 3.1 Pro reasoning-token-budget
# finding in CLAUDE.md; uniform across every model. No other model has ever come close to
# the old 10,000 ceiling (max near-cap rate elsewhere: GPT-5.5 at 3.1%, 0 failures), so this
# is a no-op for all existing/other models' actual behavior — it only matters for Gemini 3.1
# Pro, combined with its per-model "reasoning_tokens" cap below.
MAX_RETRIES = 3  # retries on malformed/incomplete (e.g. truncated) JSON
N_RESULTS = 400
EPOCHS = 3
BM25_PATH = "data/snomed/bm25_index.pkl"

# Maps PROVIDER -> chatlas Chat class. Add an entry here to support a new
# provider family; RunPod/self-hosted OpenAI-compatible endpoints can reuse
# "openai" with BASE_URL set.
PROVIDER_CHAT_CLASSES = {
    "openai": chatlas.ChatOpenAI,
    "anthropic": chatlas.ChatAnthropic,
    "google": chatlas.ChatGoogle,
    "ollama": chatlas.ChatOllama,
}

# ── Models to run ────────────────────────────────────────────────
# Each entry is one (provider, model) pair run across every condition ×
# strategy × retrieval_type × epoch. Add/remove/comment-out entries here —
# no other code needs to change to add a model.
#
# UNVERIFIED — confirm before running (see chat for details):
#   - MedGemma via Ollama: the `medgemma:4b` / `medgemma:27b` tags are not
#     explicitly version-pinned in the Ollama library listing. Confirm via
#     the pulled model's card/metadata that :4b resolves to MedGemma 1.5 and
#     :27b resolves to MedGemma 1.0 (there is no MedGemma 1.5 27B) before
#     trusting results.
#
# Note: there is no stable, non-preview "gemini-3.1-pro" — 3.1 Pro is
# Preview-only as of writing, meaning it carries real deprecation risk
# during a multi-month study. gemini-2.5-pro is the stable fallback if
# 3.1 Pro Preview gets retired mid-study.
MODELS = [
    # --- Tier 1: Frontier (RQ1) ---
    {"label": "GPT-5.5", "provider": "openai", "model": "gpt-5.5", "base_url": None},
    {"label": "Gemini 3.1 Pro", "provider": "google", "model": "gemini-3.1-pro-preview", "base_url": None, "reasoning_tokens": 6000, "max_connections": 2},
    # {"label": "Qwen3.6-27B", "provider": "ollama", "model": "qwen3.6:27b", "base_url": None},
    {"label": "MedGemma 1.0-27B", "provider": "ollama", "model": "medgemma:27b", "base_url": "http://YOUR-RUNPOD-HOST:11434", "max_connections": 4},  # RunPod A40 — CORRECTED run completed 2026-08-01 (F1=0.120, 63.0% real codes over 9,535 codes — the number to trust; small validation samples earlier suggested 80-90% but that was small-sample variance). ollama/ollama:0.32.1 pinned, OLLAMA_NUM_PARALLEL=4, OLLAMA_CONTEXT_LENGTH=32768 (confirmed unsplit via `ollama ps`), OLLAMA_FLASH_ATTENTION=1. Pod terminated after; base_url reset to placeholder. Original broken run (v0.11.6, F1~0.011-0.028, 29.4% real) preserved in inspect_results/medgemma-1-0-27b-ORIGINAL-BROKEN-v0.11.6/ for appendix comparison. See CLAUDE.md. NOTE: 4B vs 27B is NOT a controlled comparison — different Ollama generation, different hardware/backend, different concurrency; report full serving config per model in methods.

    # --- Tier 2: Base/Everyday (RQ2) ---
    {"label": "GPT-5.4-mini", "provider": "openai", "model": "gpt-5.4-mini", "base_url": None},
    {"label": "Gemini 3.1 Flash-Lite", "provider": "google", "model": "gemini-3.1-flash-lite", "base_url": None},
    # {"label": "Qwen3-32B", "provider": "ollama", "model": "qwen3:32b", "base_url": None},
    {"label": "MedGemma 1.5-4B", "provider": "ollama", "model": "medgemma:4b", "base_url": None},

    # --- RQ3: Domain fine-tuning (optional, not yet decided) ---
    # {"label": "Qwen3-8B", "provider": "ollama", "model": "qwen3:8b", "base_url": None},
    # {"label": "Med42-v2-8B", "provider": "ollama", "model": "med42-v2:8b", "base_url": None},

    # --- Supplementary: within-family Qwen scaling (optional, not yet decided) ---
    # {"label": "Qwen3-8B", "provider": "ollama", "model": "qwen3:8b", "base_url": None},
    # {"label": "Qwen3-32B", "provider": "ollama", "model": "qwen3:32b", "base_url": None},
    # {"label": "Qwen3-235B-A22B", "provider": "ollama", "model": "qwen3:235b-a22b", "base_url": None},
]

CONDITIONS = [
    "Type 2 Diabetes Mellitus",
    "Asthma",
    "Atrial Fibrillation",
    "Hypothyroidism",
    "Multiple Sclerosis",
    "Polymyalgia Rheumatica",
    "Coronary Heart Disease",
]

STRATEGIES = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
]  # the three prompting strategies as strings

RETRIEVAL_TYPES = ["non_rag", "rag"]  # the two retrieval conditions as strings

# ── Gold standard datasets ──────────────────────────────────────
# NOTE: Gold standards are only used at evaluation time (evaluate.py).
# They are loaded here for reference / quick inspection but are NOT
# used during experiment generation.
GOLD_STANDARD_FILES = {
    "Type 2 Diabetes Mellitus": "dmtype2_cod_categorised.csv",
    "Asthma": "ast_cod_categorised.csv",
    "Atrial Fibrillation": "afib_cod_categorised.csv",
    "Hypothyroidism": "thy_cod_categorised.csv",
    "Multiple Sclerosis": "ms_cod_categorised.csv",
    "Polymyalgia Rheumatica": "pmr_cod_categorised.csv",
    "Coronary Heart Disease": "chd_cod_categorised.csv",
}

# ── Load gold standards ─────────────────────────────────────────
GOLD_STANDARD_DIR = "data/categorised_gold_standards/"


def load_gold_standards(gs_files, gs_dir):
    """
    Load all gold standard codelists from CSV files.
    Returns a dict mapping condition name to a set of code strings.
    """
    gold_standards = {}

    for condition, filename in gs_files.items():  # unpacks items in the dictionary
        filepath = os.path.join(gs_dir, filename)
        df = pd.read_csv(filepath, dtype={"code": str})
        gold_standards[condition] = set(df["code"])
        print(f"  {condition}: {len(gold_standards[condition])} codes loaded")

    return gold_standards


print("Loading gold standards...")
gold_standards = load_gold_standards(GOLD_STANDARD_FILES, GOLD_STANDARD_DIR)
print(f"Loaded {len(gold_standards)} conditions\n")


# ── Initialise retrieval pipeline ───────────────────────────────
def init_retrieval():
    """
    Load ChromaDB collection and embedding model.
    Returns the collection and model objects.
    """
    chroma_client = chromadb.PersistentClient(path="data/snomed/chroma_db")
    collection = chroma_client.get_collection(name="snomed_concepts")
    print(f"  ChromaDB loaded: {collection.count()} concepts")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded")

    return collection, embedding_model


def load_bm25_index(path):
    """
    Load the persisted BM25 index and corresponding codes list from disk.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["codes"]


print("Loading BM25 index...")
bm25, bm25_codes = load_bm25_index(BM25_PATH)
print(f"  BM25 index loaded: {len(bm25_codes)} codes")


def retrieve_snomed_concepts(query, collection, embedding_model, n_results=N_RESULTS):
    """
    Retrieve semantically similar SNOMED-CT concepts from ChromaDB.
    """
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(), n_results=n_results
    )

    retrieved = []
    for code, term in zip(results["ids"][0], results["documents"][0]):
        retrieved.append({"code": code, "term": term})

    return retrieved


def retrieve_bm25_concepts(query, bm25, bm25_codes, collection, n_results=N_RESULTS):
    """
    Retrieve the top n_results SNOMED concepts by BM25 lexical score.

    Args:
        query: the search string
        bm25: the loaded BM25Okapi index
        bm25_codes: list of codes, in the SAME ORDER as the documents
                    used to build the BM25 index
        collection: ChromaDB collection, used to look up term text for
                    matched codes
        n_results: how many top-scoring documents to return
    """
    tokenised_query = query.lower().split()

    scores = bm25.get_scores(tokenised_query)

    # Pair each code with its score, then sort by score descending
    code_score_pairs = list(zip(bm25_codes, scores))
    code_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Take only the top n_results
    top_codes = [code for code, score in code_score_pairs[:n_results]]

    # Look up term text for all top codes in one batched ChromaDB call
    results = collection.get(ids=top_codes)
    retrieved = [
        {"code": code, "term": term}
        for code, term in zip(results["ids"], results["documents"])
    ]

    return retrieved


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """
    Fuse dense and BM25 ranked results using Reciprocal Rank Fusion.

    Args:
        dense_results: list of dicts with 'code' and 'term', already ranked
                        (index 0 = best match)
        bm25_results: list of dicts with 'code' and 'term', already ranked
                       (index 0 = best match)
        k: RRF constant, softens the impact of rank position

    Returns:
        list of dicts with 'code' and 'term', re-ranked by fused score,
        best match first.
    """
    scores = {}  # maps code -> fused score
    code_to_term = {}  # maps code -> term, so we can rebuild full dicts at the end

    for rank, item in enumerate(dense_results):
        code = item["code"]
        scores[code] = scores.get(code, 0) + 1 / (k + rank)
        code_to_term[code] = item["term"]

    for rank, item in enumerate(bm25_results):
        code = item["code"]
        scores[code] = scores.get(code, 0) + 1 / (k + rank)
        code_to_term[code] = item["term"]

    # Sort codes by fused score, descending (highest score = best match)
    ranked_codes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = [
        {"code": code, "term": code_to_term[code]} for code, score in ranked_codes
    ]

    return fused_results


def retrieve_hybrid(
    query, collection, embedding_model, bm25, bm25_codes, n_results=N_RESULTS
):
    """
    Retrieve SNOMED concepts using hybrid dense + BM25 retrieval, combined
    via Reciprocal Rank Fusion.
    """
    dense_results = retrieve_snomed_concepts(
        query, collection, embedding_model, n_results=n_results
    )

    bm25_results = retrieve_bm25_concepts(
        query, bm25, bm25_codes, collection, n_results=n_results
    )

    fused_results = reciprocal_rank_fusion(dense_results, bm25_results)

    # Truncate to n_results — fusion may produce more unique codes than
    # either individual method since codes appearing in only one list
    # still get included
    return fused_results[:n_results]


print("Initialising retrieval pipeline...")
collection, embedding_model = init_retrieval()

# ── Load environment (.env holds provider API keys, e.g. OPENAI_API_KEY,
#    ANTHROPIC_API_KEY, GOOGLE_API_KEY — chatlas reads these automatically) ──
load_dotenv()


def build_chat_client(provider=PROVIDER, model=MODEL, base_url=BASE_URL, skip_temperature=False):
    """
    Construct a fresh chatlas Chat client for the configured provider.
    A fresh client per call keeps every LLM call fully stateless — no
    turn history leaks between epochs, conditions, or retry attempts.

    skip_temperature: some reasoning-tier models (e.g. GPT-5.x with
    reasoning enabled) reject any non-default `temperature` outright —
    call_llm sets this after detecting that specific error, so the
    parameter is omitted and only max_tokens is constrained.
    """
    chat_class = PROVIDER_CHAT_CLASSES[provider]
    kwargs = {"model": model, "system_prompt": SYSTEM_PROMPT}
    if base_url:
        kwargs["base_url"] = base_url
    chat = chat_class(**kwargs)
    if skip_temperature:
        chat.set_model_params(max_tokens=MAX_TOKENS)
    else:
        chat.set_model_params(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return chat


print(f"Using provider={PROVIDER}, model={MODEL}\n")

# ── Prompt functions ─────────────────────────────────────────────

# Few-shot examples — verified against NHSD gold standards
# Depression: all 5 codes confirmed in nhsd depr_cod refset
# Hypertension: all 5 codes confirmed in nhsd hyp_cod refset
# CKD: codes 431855005 (stage 1), 431856006 (stage 2), 433144002 (stage 3),
#       431857002 (stage 4) confirmed in nhsd ckd refsets.
#       709044004 (CKD, unspecified) is a valid parent concept but not in the
#       stage-specific refsets — included as a clinically appropriate exemplar.
FEW_SHOT_EXAMPLES = """
Example 1: Depression
[
  {"code": "35489007", "term": "Depressive disorder"},
  {"code": "310495003", "term": "Mild depression"},
  {"code": "36923009", "term": "Major depression, single episode"},
  {"code": "191610000", "term": "Recurrent major depressive episodes, mild"},
  {"code": "14183003", "term": "Chronic major depressive disorder, single episode"}
]

Example 2: Hypertension
[
  {"code": "38341003", "term": "Hypertensive disorder"},
  {"code": "59621000", "term": "Essential hypertension"},
  {"code": "46481004", "term": "Low-renin essential hypertension"},
  {"code": "73410007", "term": "Benign secondary renovascular hypertension"},
  {"code": "78975002", "term": "Malignant essential hypertension"}
]

Example 3: Chronic Kidney Disease
[
  {"code": "709044004", "term": "Chronic kidney disease"},
  {"code": "431855005", "term": "Chronic kidney disease stage 1"},
  {"code": "431856006", "term": "Chronic kidney disease stage 2"},
  {"code": "433144002", "term": "Chronic kidney disease stage 3"},
  {"code": "431857002", "term": "Chronic kidney disease stage 4"}
]
"""

SYSTEM_PROMPT = (
    "You are a clinical terminology expert supporting EHR research in UK primary care."
)


def build_zero_shot_prompt(query, retrieved_concepts=None):
    """
    Build a zero-shot prompt. If retrieved_concepts provided, use RAG context.
    Returns the user-turn content only — the system prompt is set once on
    the chatlas Chat client itself (see build_chat_client).
    """
    if retrieved_concepts:
        context_lines = "\n".join(
            [f"- {c['code']}: {c['term']}" for c in retrieved_concepts]
        )
        context_block = f"""
The following SNOMED-CT concepts have been retrieved as potentially relevant:

RETRIEVED CONCEPTS:
{context_lines}

From the retrieved concepts above, generate a comprehensive SNOMED-CT codelist for: {query}. Select ALL retrieved concepts that are clinically relevant — do not filter to a small subset. You may supplement with additional valid SNOMED-CT codes from your own clinical knowledge where clearly justified."""
    else:
        context_block = f"Using your own clinical knowledge, generate a SNOMED-CT codelist for: {query}"

    return f"""You are building a SNOMED-CT codelist for a research cohort.

{context_block}

Requirements:
- Be comprehensive — include ALL clinically relevant codes. Do not self-censor or produce a short curated list. A valid UK primary care codelist may contain dozens of codes.
- You may supplement with additional valid SNOMED-CT codes from your own knowledge where clinically justified
- Codes must be valid SNOMED-CT concept identifiers
- Return JSON only, no preamble, no markdown

Return in this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]"""


def build_few_shot_prompt(query, retrieved_concepts=None):
    """
    Build a few-shot prompt with examples. If retrieved_concepts provided, use RAG context.
    """
    if retrieved_concepts:
        context_lines = "\n".join(
            [f"- {c['code']}: {c['term']}" for c in retrieved_concepts]
        )
        context_block = f"""
The following SNOMED-CT concepts have been retrieved as potentially relevant:

RETRIEVED CONCEPTS:
{context_lines}

From the retrieved concepts above, generate a comprehensive SNOMED-CT codelist for: {query}. Select ALL retrieved concepts that are clinically relevant — do not filter to a small subset. You may supplement with additional valid SNOMED-CT codes from your own clinical knowledge where clearly justified."""
    else:
        context_block = f"Using your own clinical knowledge, generate a SNOMED-CT codelist for: {query}"

    return f"""You are building a SNOMED-CT codelist for a research cohort.

Here are examples of well-formed SNOMED-CT codelists for other conditions:

{FEW_SHOT_EXAMPLES}

{context_block}

Requirements:
- Be comprehensive — include ALL clinically relevant codes. Do not self-censor or produce a short curated list. A valid UK primary care codelist may contain dozens of codes.
- You may supplement with additional valid SNOMED-CT codes from your own knowledge where clinically justified
- Follow the same format and clinical reasoning demonstrated in the examples above
- Codes must be valid SNOMED-CT concept identifiers
- Return JSON only, no preamble, no markdown

IMPORTANT: The examples above illustrate output format only. The number of codes in the examples does NOT indicate the expected number of codes for {query}. Include all clinically relevant codes regardless of how many that is.

Return in this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]"""


def build_cot_prompt(query, retrieved_concepts=None):
    """
    Build a chain-of-thought prompt. If retrieved_concepts provided, use RAG context.
    """
    if retrieved_concepts:
        context_lines = "\n".join(
            [f"- {c['code']}: {c['term']}" for c in retrieved_concepts]
        )
        context_block = f"""
The following SNOMED-CT concepts have been retrieved as potentially relevant for: {query}

RETRIEVED CONCEPTS:
{context_lines}

Use these retrieved concepts as your primary source of evidence when reasoning through the steps below."""
    else:
        context_block = f"Using your own clinical knowledge, generate a SNOMED-CT codelist for: {query}"

    return f"""You are building a SNOMED-CT codelist for a research cohort.

{context_block}

Before generating the codelist, reason step by step through the following. Keep your reasoning concise — a few sentences per step.

1. Define the precise clinical definition and boundaries of {query}.
2. Identify the clinically significant subtypes or variants of {query}.
3. Identify the major complications and associated conditions of {query} that would warrant their own SNOMED-CT codes.
4. Identify the management or status states of {query} that are typically coded (e.g. remission, severity, treatment status).
5. Identify conditions that should explicitly be excluded from this codelist despite superficial similarity to {query}.

Now generate a codelist that includes codes for EVERY subtype, complication, management state, and associated condition you identified in your reasoning above.

Then generate the codelist following this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]

Requirements:
- Be comprehensive — include ALL clinically relevant codes. Do not self-censor or produce a short curated list. A valid UK primary care codelist may contain dozens of codes.
- You may supplement with additional valid SNOMED-CT codes from your own knowledge where clinically justified
- Codes must be valid SNOMED-CT concept identifiers
- Return your reasoning first, then the JSON codelist
- No markdown formatting"""


# ── Response parsing (manual JSON parsing replaces the OpenAI structured
#    output schema — that schema was the suspected cause of truncation on
#    long completions, e.g. Asthma RAG runs) ──────────────────────────────


def extract_json_array(raw_text):
    """
    Pull the JSON codelist array out of a raw model response, stripping
    markdown code fences and any surrounding text (e.g. chain-of-thought
    reasoning that precedes the array).
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response")

    return text[start : end + 1]


def parse_codelist_response(raw_text):
    """
    Parse and validate a raw model response into a list of {"code", "term"}
    dicts. Raises on malformed or incomplete JSON — e.g. an array truncated
    mid-object because the completion hit the token limit — so the caller
    can retry.
    """
    json_str = extract_json_array(raw_text)
    codes = json.loads(json_str)

    if not isinstance(codes, list):
        raise ValueError("Parsed JSON is not a list")

    for item in codes:
        if not isinstance(item, dict) or "code" not in item or "term" not in item:
            raise ValueError(f"Malformed code entry: {item!r}")
        if not isinstance(item["code"], str) or not isinstance(item["term"], str):
            raise ValueError(f"'code' and 'term' must be strings: {item!r}")

    return codes


# ── API call function ───────────────────────────────────────────


# Substring matched against API error messages to detect models that reject
# a non-default `temperature` outright (e.g. GPT-5.x reasoning-tier models
# with reasoning enabled — OpenAI's replacement there is `reasoning_effort`,
# which controls reasoning depth, not sampling/determinism, so there is no
# real substitute for TEMPERATURE=0 on these models; omitting it is the
# closest available option).
TEMPERATURE_UNSUPPORTED_MARKER = "'temperature' is not supported"


def call_llm(
    user_content,
    provider=PROVIDER,
    model=MODEL,
    base_url=BASE_URL,
    max_retries=MAX_RETRIES,
):
    """
    Call the configured LLM provider via chatlas and parse its response into
    a codelist. Retries up to max_retries times on malformed/incomplete
    JSON (e.g. truncation), transient API errors, or a model rejecting the
    `temperature` parameter (retried without it). Each attempt uses a fresh
    chat client, so retries never carry over prior turn history.

    Returns:
        (codes, raw_text, n_attempts, error_message, temperature_applied)
        On success: error_message is None.
        On exhausted retries: codes is None and error_message is set.
        temperature_applied is TEMPERATURE if the returning attempt sent it,
        or None if it had to be omitted (either by policy — the primary run
        uses provider-default temperature for every model, see
        RUN_AT_TEMPERATURE_ZERO — or because the model rejected it outright).
    """
    last_error = None
    raw_text = None
    skip_temperature = not RUN_AT_TEMPERATURE_ZERO

    for attempt in range(1, max_retries + 1):
        chat = build_chat_client(
            provider=provider,
            model=model,
            base_url=base_url,
            skip_temperature=skip_temperature,
        )
        try:
            response = chat.chat(user_content, echo="none", stream=False)
            raw_text = str(response)
            codes = parse_codelist_response(raw_text)
            temperature_applied = None if skip_temperature else TEMPERATURE
            return codes, raw_text, attempt, None, temperature_applied
        except Exception as e:
            if not skip_temperature and TEMPERATURE_UNSUPPORTED_MARKER in str(e):
                skip_temperature = True
            last_error = f"attempt {attempt}/{max_retries}: {e}"

    temperature_applied = None if skip_temperature else TEMPERATURE
    return None, raw_text, max_retries, last_error, temperature_applied


# ── Prompt builder lookup ────────────────────────────────────────
PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "chain_of_thought": build_cot_prompt,
}
