# inspect_pipeline.py

"""
inspect_pipeline.py

Standalone reimplementation of experiment_runner.py's runner + evaluate.py's
metrics using Inspect AI (inspect_ai) as the orchestration harness.
experiment_runner.py and evaluate.py are left untouched — this file imports
shared config/retrieval/prompt/LLM-call logic from pipeline_core.py (the
same module experiment_runner.py itself imports from), rather than
duplicating that logic or importing from experiment_runner.py directly.

Corrections made vs. the original design brief (see chat history for detail):
  - There is no `get_retrieved_concepts()` in pipeline_core.py; the actual
    hybrid-retrieval function is `retrieve_hybrid()`.
  - pc.RETRIEVAL_TYPES are lowercase ("non_rag" / "rag"), not "RAG".
  - experiment_runner.py's `if __name__ == "__main__":` guard already
    existed (it only wraps run_full_experiment()) — no edit was needed
    there.

Do NOT use Inspect's native model/generate() — every LLM call goes through
pipeline_core.call_llm() (chatlas), run off the event loop via
asyncio.to_thread(). eval() is still given a `model=` argument because
Inspect requires one, but it's "mockllm/model" — a no-op stub that's never
actually invoked, since the Solver never calls `generate()`.
"""

# Standard library
import asyncio
import json
import os
import re
from datetime import datetime

# Third party
import chromadb
import pandas as pd
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.scorer import scorer, Score, Target, mean, stderr
from inspect_ai.model import (
    ModelOutput,
    get_model,
    GenerateConfig,
    ChatMessageSystem,
    ChatMessageUser,
)

# pipeline_core's module-level setup (gold standards, ChromaDB, BM25,
# embedding model) runs on import — same shared module experiment_runner.py
# imports from, so this file never imports experiment_runner.py itself.
import pipeline_core as pc

# ── Configuration ──────────────────────────────────────────────
RERUN_EXPERIMENTS = True
LOG_DIR = "inspect_logs"
OUTPUT_DIR = "inspect_results"

# True  = inspect_solver (Inspect's native get_model()/generate(), via the
#         OpenAI Responses API for OpenAI models) — PRIMARY as of the
#         solver comparison (see chat history / CLAUDE.md Active Issue #2).
#         Captures real reasoning-trace visibility (GPT-5.5 logged 13
#         resolvable reasoning summary segments; chatlas_solver's events log
#         has zero ModelEvents, so no trace is recoverable via that path at
#         all) and a complete request/response audit trail per call.
# False = chatlas_solver (pipeline_core.call_llm(), via the Chat Completions
#         API) — kept as the non-primary fallback path for comparison runs.
# NOTE: the two paths are NOT interchangeable outputs of "the same model" —
# a side-by-side comparison (GPT-5.4-mini, GPT-5.5; T2DM/zero_shot/non_rag)
# showed only 9-14 code overlap out of ~40-60 codes per model, because the
# two paths exercise different underlying OpenAI API surfaces (Responses vs.
# Chat Completions). Switching this flag mid-study would be a real
# methodological change, not a transparent swap.
USE_INSPECT_NATIVE_MODEL = True


# ── Gold standard terms/categories ──────────────────────────────
# pipeline_core.gold_standards only holds the code SET (that's all it needs
# for its own use). evaluate.py's code_breakdown_metrics.csv schema
# additionally has `term` and `gold_category` columns, so this small loader
# mirrors evaluate.py's richer load_gold_standards() just enough to get exact
# schema parity there — not a reimplementation of pipeline_core's logic.
def _load_gold_terms_and_categories(gs_files, gs_dir):
    terms = {}
    categories = {}
    for condition, filename in gs_files.items():
        filepath = os.path.join(gs_dir, filename)
        df = pd.read_csv(filepath, dtype={"code": str})
        terms[condition] = dict(zip(df["code"], df["term"]))
        categories[condition] = dict(zip(df["code"], df["category"]))
    return terms, categories


gold_standard_terms, gold_standard_categories = _load_gold_terms_and_categories(
    pc.GOLD_STANDARD_FILES, pc.GOLD_STANDARD_DIR
)

# ── T2DM audit gold standard (dual gold-standard sensitivity analysis) ──
# Mirrors evaluate.py's ADDITIONAL_GOLD_STANDARDS / evaluate_additional_gold_standards():
# T2DM has two gold standards — the main dmtype2_cod_categorised.csv (used
# everywhere else in the pipeline, 23 codes) and a broader
# dmtype2audit_cod_categorised.csv (139 codes). This does NOT add a new
# CONDITIONS entry or re-run generation — it re-scores T2DM's *already-
# generated* codes (recovered from each sample's Score.answer) against the
# audit gold standard, producing a parallel "Type 2 Diabetes Mellitus
# (Audit)" condition in every output CSV. See derive_csvs() below.
T2DM_AUDIT_SOURCE_CONDITION = "Type 2 Diabetes Mellitus"
T2DM_AUDIT_CONDITION_LABEL = "Type 2 Diabetes Mellitus (Audit)"
T2DM_AUDIT_GOLD_STANDARD_FILE = "dmtype2audit_cod_categorised.csv"

_t2dm_audit_df = pd.read_csv(
    os.path.join(pc.GOLD_STANDARD_DIR, T2DM_AUDIT_GOLD_STANDARD_FILE), dtype={"code": str}
)
t2dm_audit_gold_codes = set(_t2dm_audit_df["code"])
t2dm_audit_terms = dict(zip(_t2dm_audit_df["code"], _t2dm_audit_df["term"]))
t2dm_audit_categories = dict(zip(_t2dm_audit_df["code"], _t2dm_audit_df["category"]))
print(
    f"  T2DM audit gold standard loaded: {len(t2dm_audit_gold_codes)} codes "
    f"(dual gold-standard sensitivity vs. {len(pc.gold_standards[T2DM_AUDIT_SOURCE_CONDITION])}-code main standard)"
)


# ── Dataset ──────────────────────────────────────────────────────
def build_dataset():
    """
    One Sample per condition × strategy × retrieval_type × epoch × model —
    7 × 3 × 2 × 3 × len(pc.MODELS) samples total.
    """
    samples = []
    for condition in pc.CONDITIONS:
        target = json.dumps(sorted(pc.gold_standards[condition]))
        for strategy in pc.STRATEGIES:
            for retrieval_type in pc.RETRIEVAL_TYPES:  # "non_rag" / "rag"
                for epoch in range(1, pc.EPOCHS + 1):
                    for model_dict in pc.MODELS:
                        samples.append(
                            Sample(
                                input=condition,
                                target=target,
                                metadata={
                                    "condition": condition,
                                    "strategy": strategy,
                                    "retrieval_type": retrieval_type,
                                    "epoch": epoch,
                                    "model": model_dict,
                                },
                            )
                        )
    return samples


# ── Solver: chatlas (pipeline_core.call_llm()) — NON-PRIMARY ──────
# Kept as the fallback/comparison path; inspect_solver (below) is primary
# as of the solver comparison — see USE_INSPECT_NATIVE_MODEL above.
@solver
def chatlas_solver():
    """
    chatlas-based Solver (NON-PRIMARY — see USE_INSPECT_NATIVE_MODEL).
    Builds the condition/strategy/retrieval_type-appropriate prompt, calls
    call_llm() (chatlas, via the Chat Completions API — NOT Inspect's
    native generate()) via asyncio.to_thread since call_llm is synchronous,
    and records the retrieval/LLM metadata the scorer needs. No visibility
    into reasoning traces via this path — its events log has zero
    ModelEvents, since the API call happens entirely outside Inspect's
    instrumentation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        condition = state.metadata["condition"]
        strategy = state.metadata["strategy"]
        retrieval_type = state.metadata["retrieval_type"]
        model_dict = state.metadata["model"]

        # Step 1: retrieval (RAG only) — pc.retrieve_hybrid(), not
        # get_retrieved_concepts() (that function doesn't exist).
        retrieved_concepts = None
        if retrieval_type == "rag":
            retrieved_concepts = pc.retrieve_hybrid(
                condition,
                pc.collection,
                pc.embedding_model,
                pc.bm25,
                pc.bm25_codes,
                n_results=pc.N_RESULTS,
            )
        state.metadata["retrieved_set"] = (
            [c["code"] for c in retrieved_concepts] if retrieved_concepts else []
        )

        # Step 2: build the prompt
        prompt_builder = pc.PROMPT_BUILDERS[strategy]
        user_content = prompt_builder(condition, retrieved_concepts)

        # Step 3: call the LLM synchronously via chatlas, off the event loop
        codes, raw_text, n_attempts, error_message, temperature_applied = (
            await asyncio.to_thread(
                pc.call_llm,
                user_content,
                provider=model_dict["provider"],
                model=model_dict["model"],
                base_url=model_dict.get("base_url"),
                max_retries=pc.MAX_RETRIES,
            )
        )

        state.metadata["raw_text"] = raw_text
        state.metadata["n_attempts"] = n_attempts
        state.metadata["temperature_applied"] = temperature_applied
        state.metadata["error_message"] = error_message

        state.output = ModelOutput.from_content(
            model=model_dict["model"], content=raw_text or ""
        )
        return state

    return solve


# ── Solver: Inspect native (get_model()/generate()) ───────────────
#
# API details below were verified against the installed inspect_ai==0.3.249
# package source directly (inspect.signature()/getsource(), plus one live
# call against the real GPT-5.5 API to see the actual error shape) — not
# assumed from docs, since a couple of things turned out to differ from the
# original design brief:
#
#   - base_url is a direct keyword arg on get_model(base_url=...), forwarded
#     straight to the provider's __init__. There is no need to set
#     OLLAMA_BASE_URL (or any env var) — that assumption in the brief was
#     wrong for this version.
#   - Ollama specifically: inspect_ai.model._providers.ollama.OllamaAPI's
#     own default is service_base_url="http://localhost:11434/v1" — i.e.
#     Inspect expects the "/v1" suffix explicitly and does NOT append it
#     for you. Our MODELS dict stores bare "http://host:port" (what chatlas
#     expects), so _to_inspect_base_url() below appends "/v1" only when
#     talking to Inspect.
#   - GenerateConfig (inspect_ai.model.GenerateConfig) has both `temperature`
#     and `max_connections` as fields — max_connections is read from the
#     config passed to each generate() call (not from get_model() itself),
#     confirmed by reading Model._connection_concurrency()'s source, so
#     setting it per-call (as done below) is sufficient — no need to set it
#     at get_model() construction time.
#   - Provider name strings are exactly "openai", "anthropic", "google",
#     "ollama" (confirmed via inspect_ai.model._providers.providers's
#     @modelapi(name=...) registrations) — matching our own MODELS dict's
#     provider strings exactly, so "{provider}/{model}" is a correct,
#     un-cached mapping requiring no provider-name translation.
#   - Errors: calling generate() with a rejected temperature raises
#     inspect_ai.model._model.ModelGenerateError — but that class lives in
#     a private module and is NOT exported from inspect_ai.model (confirmed
#     by trying to import it), so this catches generic Exception, same as
#     chatlas_solver and pipeline_core.call_llm(). Live-tested against the
#     real gpt-5.5 API: str(e) on that wrapped exception still contains the
#     exact substring "'temperature' is not supported" (nested inside the
#     wrapped OpenAI BadRequestError's repr), so
#     pc.TEMPERATURE_UNSUPPORTED_MARKER works unchanged for this path too.

INSPECT_MAX_CONNECTIONS_OLLAMA = 1
INSPECT_MAX_CONNECTIONS_DEFAULT = 5


def _to_inspect_model_string(model_dict):
    """Map our {provider, model} dict to Inspect's 'provider/model' string."""
    return f"{model_dict['provider']}/{model_dict['model']}"


def _to_inspect_base_url(model_dict):
    """
    Our MODELS dict stores Ollama base_url as bare "http://host:port" (what
    chatlas expects). Inspect's OllamaAPI needs the "/v1" suffix explicitly
    (its own default is "http://localhost:11434/v1") — see note above.
    """
    base_url = model_dict.get("base_url")
    if base_url and model_dict["provider"] == "ollama":
        return base_url.rstrip("/") + "/v1"
    return base_url


@solver
def inspect_solver():
    """
    Inspect-native Solver — PRIMARY (see USE_INSPECT_NATIVE_MODEL above).
    Same prompt-building and retry/temperature-fallback behavior as
    chatlas_solver, and writes the identical state.metadata fields
    (raw_text, n_attempts, temperature_applied, error_message) — so
    codelist_scorer() and derive_csvs() work identically regardless of
    which solver produced the sample. The LLM call itself goes through
    Inspect's get_model()/generate() (OpenAI Responses API for OpenAI
    models) instead of pipeline_core.call_llm() (chatlas, Chat Completions
    API) — this is a real methodological difference, not just a different
    code path to the same result; see the module-level note above
    USE_INSPECT_NATIVE_MODEL.

    Full request/response, including reasoning-trace segments where the
    model produces them, is captured in the Inspect eval log's ModelEvent
    for every call made through this solver — resolving Active Issue #2
    in CLAUDE.md ("reasoning/thinking traces not logged").
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        condition = state.metadata["condition"]
        strategy = state.metadata["strategy"]
        retrieval_type = state.metadata["retrieval_type"]
        model_dict = state.metadata["model"]

        # Step 1: retrieval (RAG only) — identical to chatlas_solver
        retrieved_concepts = None
        if retrieval_type == "rag":
            retrieved_concepts = pc.retrieve_hybrid(
                condition,
                pc.collection,
                pc.embedding_model,
                pc.bm25,
                pc.bm25_codes,
                n_results=pc.N_RESULTS,
            )
        state.metadata["retrieved_set"] = (
            [c["code"] for c in retrieved_concepts] if retrieved_concepts else []
        )

        # Step 2: build the prompt (same builders, same system prompt)
        prompt_builder = pc.PROMPT_BUILDERS[strategy]
        user_content = prompt_builder(condition, retrieved_concepts)
        messages = [
            ChatMessageSystem(content=pc.SYSTEM_PROMPT),
            ChatMessageUser(content=user_content),
        ]

        # Step 3: resolve provider/model -> Inspect model string + base_url
        inspect_model_string = _to_inspect_model_string(model_dict)
        inspect_base_url = _to_inspect_base_url(model_dict)
        max_connections = model_dict.get("max_connections") or (
            INSPECT_MAX_CONNECTIONS_OLLAMA
            if model_dict["provider"] == "ollama"
            else INSPECT_MAX_CONNECTIONS_DEFAULT
        )

        # Step 4: call the LLM via Inspect's native model abstraction, with
        # the same retry policy as pipeline_core.call_llm(): retry on
        # malformed/incomplete JSON, transient API errors, or a model
        # rejecting `temperature` (retried without it).
        skip_temperature = not pc.RUN_AT_TEMPERATURE_ZERO
        raw_text = None
        n_attempts = 0
        error_message = None

        for attempt in range(1, pc.MAX_RETRIES + 1):
            n_attempts = attempt
            config_kwargs = {
                "max_tokens": pc.MAX_TOKENS,
                "max_connections": max_connections,
            }
            if not skip_temperature:
                config_kwargs["temperature"] = pc.TEMPERATURE
            # Uniform reasoning-token ceiling: applied via a "reasoning_tokens"
            # key on a model's dict in pc.MODELS (currently only Gemini 3.1 Pro
            # has one set). This is a *safety ceiling*, not an attempt to force
            # equivalence with any other model — chosen empirically above what
            # GPT-5.5 naturally used (unconstrained) for the same task, so a
            # well-behaved reasoning model's own behavior is unaffected by it.
            # Models without a "reasoning_tokens" entry are untouched.
            if model_dict.get("reasoning_tokens") is not None:
                config_kwargs["reasoning_tokens"] = model_dict["reasoning_tokens"]

            try:
                model = get_model(inspect_model_string, base_url=inspect_base_url)
                response = await model.generate(
                    input=messages,
                    config=GenerateConfig(**config_kwargs),
                )
                raw_text = response.completion
                pc.parse_codelist_response(raw_text)  # validate; retry on failure
                error_message = None
                break
            except Exception as e:
                # Inspect wraps provider errors via repr() of the nested
                # exception (e.g. ModelGenerateError wrapping OpenAI's
                # BadRequestError), which double-escapes quote characters —
                # the real string contains literal `\'temperature\'`, not
                # `'temperature'`. Confirmed via a live call against gpt-5.5:
                # the same marker used by pipeline_core.call_llm() (which
                # reads chatlas's unescaped error text directly) does NOT
                # match Inspect's escaped format without normalizing first.
                error_str = str(e).replace("\\'", "'")
                if not skip_temperature and pc.TEMPERATURE_UNSUPPORTED_MARKER in error_str:
                    skip_temperature = True
                error_message = f"attempt {attempt}/{pc.MAX_RETRIES}: {e}"

        temperature_applied = None if skip_temperature else pc.TEMPERATURE

        state.metadata["raw_text"] = raw_text
        state.metadata["n_attempts"] = n_attempts
        state.metadata["temperature_applied"] = temperature_applied
        state.metadata["error_message"] = error_message

        state.output = ModelOutput.from_content(
            model=model_dict["model"], content=raw_text or ""
        )
        return state

    return solve


# ── Scorer ───────────────────────────────────────────────────────
def _compute_metrics(generated_set, gold_codes):
    """
    Same formulas as evaluate.py's compute_metrics(), fixed at beta=1.0 (F1)
    and beta=0.5 (F0.5) — both reported, matching evaluate.py's own
    precedent of tracking F0.5 alongside F1 for a precision-weighted view
    (a false positive costs more than a false negative for a clinical
    codelist feeding a phenotype algorithm). Also reports size_ratio
    (generated_count/gold_count), previously only computed by evaluate.py
    at the per-epoch level and never carried into a pooled CSV anywhere —
    added here, and to pooled_df in derive_csvs(), for the analysis report.
    """
    gold_count = len(gold_codes)
    if len(generated_set) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f05": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": gold_count,
            "generated_count": 0,
            "gold_count": gold_count,
            "size_ratio": 0.0,
        }
    true_positives = generated_set & gold_codes
    false_positives = generated_set - gold_codes
    false_negatives = gold_codes - generated_set
    precision = len(true_positives) / len(generated_set)
    recall = len(true_positives) / gold_count if gold_count > 0 else 0.0

    def _f_beta(beta):
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f_beta(1.0), 4),
        "f05": round(_f_beta(0.5), 4),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "generated_count": len(generated_set),
        "gold_count": gold_count,
        "size_ratio": round(len(generated_set) / gold_count if gold_count > 0 else 0, 4),
    }


def _score_codes_against_gold(codes, gold_codes, retrieval_type, retrieved_set, terms, categories):
    """
    Core scoring logic shared between codelist_scorer() (live scoring during
    eval(), against each condition's own gold standard) and derive_csvs()'s
    T2DM-audit re-scoring pass (post-hoc, against the audit gold standard,
    reusing already-generated codes recovered from the log — no re-run).
    Returns the same dict shape codelist_scorer() puts in Score.metadata.
    """
    # Converting to a set here is for METRIC COMPUTATION only (mirrors
    # evaluate.py's own compute_metrics()) — the raw `codes` list, with any
    # duplicates intact, is what the caller stores/preserves elsewhere.
    # Codes are never deduplicated before this point (Julian's instruction);
    # only the precision/recall/F1 math is set-based, same as evaluate.py.
    generated_set = set(c["code"] for c in codes)
    metrics = _compute_metrics(generated_set, gold_codes)

    true_positives = generated_set & gold_codes
    false_positives = generated_set - gold_codes
    false_negatives = gold_codes - generated_set

    generated_lookup = {c["code"]: c["term"] for c in codes}

    retrieval_attribution = [
        {
            "code": code,
            "failure_type": (
                "generation_failure"
                if retrieval_type != "rag" or code in retrieved_set
                else "retrieval_failure"
            ),
        }
        for code in false_negatives
    ]

    code_breakdown = (
        [
            {
                "eval_category": "true_positive",
                "code": code,
                "term": generated_lookup.get(code, terms.get(code, "")),
                "gold_category": categories.get(code, ""),
            }
            for code in true_positives
        ]
        + [
            {
                "eval_category": "false_positive",
                "code": code,
                "term": generated_lookup.get(code, ""),
                "gold_category": "",
            }
            for code in false_positives
        ]
        + [
            {
                "eval_category": "false_negative",
                "code": code,
                "term": terms.get(code, ""),
                "gold_category": categories.get(code, ""),
            }
            for code in false_negatives
        ]
    )

    return {
        **metrics,
        "generated_count": len(generated_set),
        "gold_count": len(gold_codes),
        "n_raw_codes": len(codes),  # includes duplicates, unlike generated_count
        "code_breakdown": code_breakdown,
        "retrieval_attribution": retrieval_attribution,
    }


@scorer(metrics=[mean(), stderr()])
def codelist_scorer():
    """
    Parses the raw LLM output (extract_json_array + parse_codelist_response),
    computes precision/recall/F1 against the gold standard (formulas matching
    evaluate.py's compute_metrics), and classifies false negatives as
    retrieval_failure vs. generation_failure.

    For non-RAG runs, every FN is a generation_failure — there was no
    retrieval step to have failed. This matches the *intent* of
    evaluate.py's evaluate_retrieval_attribution(), which only evaluates RAG
    rows for retrieval attribution at all (non-RAG rows are skipped
    entirely there). This scorer computes attribution uniformly for every
    sample regardless of retrieval_type, but derive_csvs() below still
    filters to RAG-only rows when writing retrieval_attribution.csv, to
    match evaluate.py's output exactly.
    """

    async def score(state: TaskState, target: Target) -> Score:
        gold_codes = set(json.loads(target.text))
        condition = state.metadata["condition"]
        retrieval_type = state.metadata["retrieval_type"]
        retrieved_set = set(state.metadata.get("retrieved_set", []))

        try:
            codes = pc.parse_codelist_response(state.output.completion)
        except Exception as e:
            return Score(
                value=0.0,
                answer=state.output.completion,
                metadata={
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "f05": 0.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": len(gold_codes),
                    "generated_count": 0,
                    "gold_count": len(gold_codes),
                    "size_ratio": 0.0,
                    "n_raw_codes": 0,
                    "parse_error": str(e),
                    "code_breakdown": [],
                    "retrieval_attribution": [],
                },
            )

        terms = gold_standard_terms.get(condition, {})
        categories = gold_standard_categories.get(condition, {})
        result_metadata = _score_codes_against_gold(
            codes, gold_codes, retrieval_type, retrieved_set, terms, categories
        )

        return Score(
            value=result_metadata["f1"],
            answer=json.dumps(codes),
            metadata=result_metadata,
        )

    return score


# ── Task ─────────────────────────────────────────────────────────
@task
def codelist_task():
    # Note: the brief's suggested `solver = inspect_solver if ... else
    # chatlas_solver` would assign the un-called @solver factory function
    # itself, not a Solver instance — Task(solver=...) needs the actual
    # callable produced by calling the factory, so both branches are called.
    solver = inspect_solver() if USE_INSPECT_NATIVE_MODEL else chatlas_solver()
    return Task(
        dataset=build_dataset(),
        solver=solver,
        scorer=codelist_scorer(),
        name="codelist_generation",
    )


# ── Post-eval CSV derivation ──────────────────────────────────────
def _slugify_model_label(label):
    """
    Filesystem-safe folder name for a model_label, e.g.
    "GPT-4o (v1 baseline model)" -> "gpt-4o-v1-baseline-model".
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", label.strip()).strip("-").lower()
    return slug or "unknown-model"


def _compact_model_prefix(label):
    """
    Compact, separator-free filename prefix for a model_label, e.g.
    "GPT-5.4-mini" -> "gpt54mini". Distinct from _slugify_model_label()
    (which is dash-separated and used for folder names) — this is for
    CSV filenames themselves, so that files uploaded together (e.g. to
    Claude Desktop) don't collide on the shared base filenames
    (per_epoch_metrics.csv etc.) once pulled out of their model folder.
    """
    prefix = re.sub(r"[^a-zA-Z0-9]+", "", label.strip()).lower()
    return prefix or "unknownmodel"


def derive_csvs(log, output_dir):
    """
    Iterates the Inspect eval log's samples and derives six CSVs matching
    evaluate.py's schema (plus subtype_metrics/subtype_metrics_pooled),
    with model_label/provider/model columns added throughout. Each CSV is
    split by model_label and written into its own subfolder under
    output_dir (e.g. output_dir/gpt-4o-v1-baseline-model/per_epoch_metrics.csv) —
    a single run's log can contain multiple models, and each model's results
    need to land in a separate, clearly-labeled place rather than overwriting
    a single shared set of files every time derive_csvs() runs.
    """
    per_epoch_rows = []
    code_breakdown_rows = []
    retrieval_attribution_rows = []

    def _append_rows_for_sample(base_info, m, retrieval_type):
        # f05/size_ratio are derived here from precision/recall/generated_count/
        # gold_count rather than read via m.get("f05")/m.get("size_ratio") —
        # those keys don't exist in Score.metadata for logs scored before this
        # field was added (the .eval file is frozen at scoring time), so
        # trusting them would silently produce 0.0 for every pre-existing log.
        # Deriving from fields every log already has works for old and new
        # logs alike.
        precision = m.get("precision", 0.0)
        recall = m.get("recall", 0.0)
        f05 = (
            (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
            if (precision + recall) > 0
            else 0.0
        )
        generated_count = m.get("generated_count", 0)
        gold_count = m.get("gold_count", 0)
        size_ratio = round(generated_count / gold_count, 4) if gold_count > 0 else 0.0

        per_epoch_rows.append(
            {
                **base_info,
                "scope": "full",
                "precision": precision,
                "recall": recall,
                "f1": m.get("f1", 0.0),
                "f05": round(f05, 4),
                "true_positives": m.get("true_positives", 0),
                "false_positives": m.get("false_positives", 0),
                "false_negatives": m.get("false_negatives", 0),
                "generated_count": generated_count,
                "gold_count": gold_count,
                "size_ratio": size_ratio,
            }
        )

        for row in m.get("code_breakdown", []):
            code_breakdown_rows.append({**base_info, **row})

        if retrieval_type == "rag":  # matches evaluate.py: RAG rows only
            for row in m.get("retrieval_attribution", []):
                retrieval_attribution_rows.append(
                    {
                        "condition": base_info["condition"],
                        "strategy": base_info["strategy"],
                        "epoch": base_info["epoch"],
                        "model_label": base_info["model_label"],
                        "provider": base_info["provider"],
                        "model": base_info["model"],
                        **row,
                    }
                )

    for sample in log.samples:
        meta = sample.metadata
        model_dict = meta["model"]
        score = next(iter(sample.scores.values()))
        m = score.metadata

        base_info = {
            "condition": meta["condition"],
            "strategy": meta["strategy"],
            "retrieval_type": meta["retrieval_type"],
            "epoch": meta["epoch"],
            "model_label": model_dict["label"],
            "provider": model_dict["provider"],
            "model": model_dict["model"],
        }

        _append_rows_for_sample(base_info, m, meta["retrieval_type"])

        # ── T2DM audit re-scoring pass (dual gold-standard sensitivity) ──
        # Re-scores this sample's already-generated codes against the
        # broader audit gold standard (139 codes vs. the main 23), with no
        # re-generation. Only fires for T2DM samples; every other condition
        # is untouched. A parse failure here just means this sample's codes
        # couldn't be recovered (mirrors codelist_scorer()'s own guard) —
        # skip the audit row rather than fabricate one.
        if meta["condition"] == T2DM_AUDIT_SOURCE_CONDITION:
            try:
                codes = json.loads(score.answer)
            except (TypeError, ValueError):
                codes = None
            if codes is not None:
                retrieved_set = set(meta.get("retrieved_set", []))
                audit_metadata = _score_codes_against_gold(
                    codes,
                    t2dm_audit_gold_codes,
                    meta["retrieval_type"],
                    retrieved_set,
                    t2dm_audit_terms,
                    t2dm_audit_categories,
                )
                audit_base_info = {**base_info, "condition": T2DM_AUDIT_CONDITION_LABEL}
                _append_rows_for_sample(audit_base_info, audit_metadata, meta["retrieval_type"])

    per_epoch_df = pd.DataFrame(per_epoch_rows)
    code_breakdown_df = pd.DataFrame(code_breakdown_rows)
    retrieval_attribution_df = pd.DataFrame(retrieval_attribution_rows)

    # ── Hallucinated-identifier check ──────────────────────────────
    # Flags every generated code (true_positive/false_positive rows) as real
    # or fabricated by checking it against the actual local SNOMED-CT
    # ChromaDB (632,230 real concept IDs) — cheap, local, no API cost. This
    # exists because "false positive" alone conflates two very different
    # failures: picking a real-but-wrong concept vs. inventing an identifier
    # that doesn't exist in the terminology at all. See CLAUDE.md
    # "hallucinated identifier rate" finding — every model tested fabricates
    # a non-trivial fraction of codes (15.5%-91.3% real across the 7 models
    # run so far), not just the smaller/quantized ones.
    if not code_breakdown_df.empty:
        chroma_client = chromadb.PersistentClient(path="data/snomed/chroma_db")
        collection = chroma_client.get_collection(name="snomed_concepts")
        unique_codes = list(code_breakdown_df["code"].dropna().astype(str).unique())
        real_codes = set()
        CHUNK = 300
        for i in range(0, len(unique_codes), CHUNK):
            chunk = unique_codes[i : i + CHUNK]
            result = collection.get(ids=chunk)
            real_codes.update(result["ids"])
        code_breakdown_df["code_is_real"] = (
            code_breakdown_df["code"].astype(str).isin(real_codes)
        )
        # false_negative rows are gold-standard codes, not generated ones —
        # always real by construction, marked True for consistency rather
        # than left ambiguous.
        code_breakdown_df.loc[
            code_breakdown_df["eval_category"] == "false_negative", "code_is_real"
        ] = True

    group_cols = [
        "condition",
        "strategy",
        "retrieval_type",
        "model_label",
        "provider",
        "model",
    ]
    pooled_df = per_epoch_df.groupby(group_cols)[
        ["precision", "recall", "f1", "f05", "size_ratio"]
    ].agg(["mean", "std"])
    pooled_df.columns = ["_".join(col) for col in pooled_df.columns]
    pooled_df = pooled_df.reset_index()
    n_epochs = per_epoch_df.groupby(group_cols).size().reset_index(name="n_epochs")
    pooled_df = pooled_df.merge(n_epochs, on=group_cols)
    counts_mean = (
        per_epoch_df.groupby(group_cols)[["generated_count", "gold_count"]]
        .mean()
        .rename(columns={"generated_count": "generated_count_mean", "gold_count": "gold_count_mean"})
        .reset_index()
    )
    pooled_df = pooled_df.merge(counts_mean, on=group_cols)

    # ── Subtype-level recall (Emmanuel's categorised gold standards) ──
    # Independent load-and-join from the categorised CSVs (pc.GOLD_STANDARD_FILES
    # / pc.GOLD_STANDARD_DIR — the same 7-condition mapping already used
    # elsewhere in the pipeline; dmtype2audit_cod_categorised.csv has no
    # corresponding entry in pc.CONDITIONS, so it's correctly excluded here
    # too, same as everywhere else). Only TP/FN rows carry a gold category —
    # FP rows aren't in the gold standard at all.
    category_lookup = {}
    for condition, filename in pc.GOLD_STANDARD_FILES.items():
        filepath = os.path.join(pc.GOLD_STANDARD_DIR, filename)
        cat_df = pd.read_csv(filepath, dtype={"code": str})
        category_lookup[condition] = dict(zip(cat_df["code"], cat_df["category"]))
    category_lookup[T2DM_AUDIT_CONDITION_LABEL] = t2dm_audit_categories

    subtype_source = code_breakdown_df[
        code_breakdown_df["eval_category"].isin(["true_positive", "false_negative"])
    ].copy()
    subtype_source["category"] = subtype_source.apply(
        lambda row: category_lookup.get(row["condition"], {}).get(row["code"], "unknown"),
        axis=1,
    )
    subtype_source["is_tp"] = subtype_source["eval_category"] == "true_positive"
    subtype_source["is_fn"] = subtype_source["eval_category"] == "false_negative"

    subtype_group_cols = [
        "model_label",
        "provider",
        "model",
        "condition",
        "strategy",
        "retrieval_type",
        "epoch",
        "category",
    ]
    subtype_df = subtype_source.groupby(subtype_group_cols).agg(
        tp_count=("is_tp", "sum"),
        fn_count=("is_fn", "sum"),
    ).reset_index()
    subtype_df["gold_count"] = subtype_df["tp_count"] + subtype_df["fn_count"]
    subtype_df["recall"] = (subtype_df["tp_count"] / subtype_df["gold_count"]).round(4)

    subtype_pooled_group_cols = [
        "model_label",
        "provider",
        "model",
        "condition",
        "strategy",
        "retrieval_type",
        "category",
    ]
    subtype_pooled_df = subtype_df.groupby(subtype_pooled_group_cols).agg(
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        gold_count_mean=("gold_count", "mean"),
        tp_count_mean=("tp_count", "mean"),
        fn_count_mean=("fn_count", "mean"),
        n_epochs=("recall", "size"),
    ).reset_index()

    # ── Write each CSV split by model_label into its own subfolder ──
    all_csvs = {
        "per_epoch_metrics.csv": per_epoch_df,
        "pooled_metrics.csv": pooled_df,
        "code_breakdown_metrics.csv": code_breakdown_df,
        "retrieval_attribution.csv": retrieval_attribution_df,
        "subtype_metrics.csv": subtype_df,
        "subtype_metrics_pooled.csv": subtype_pooled_df,
    }

    # ── Runtime metadata ──────────────────────────────────────────
    # log.stats.started_at/completed_at are wall-clock timestamps Inspect
    # itself records for the whole run — reading them straight from the log
    # is exact and doesn't depend on parsing the terminal's printed
    # "total time" summary. This is log-level, not per-model — if a single
    # log ever covered more than one model (not how any real run here has
    # been done; each run so far restricts MODELS to one entry), every
    # model in that log would get the same duration, which would be wrong.
    started_at = datetime.fromisoformat(log.stats.started_at)
    completed_at = datetime.fromisoformat(log.stats.completed_at)
    duration_seconds = (completed_at - started_at).total_seconds()
    hours, remainder = divmod(int(duration_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_hms = f"{hours}:{minutes:02d}:{seconds:02d}"

    os.makedirs(output_dir, exist_ok=True)
    model_labels = sorted(per_epoch_df["model_label"].unique())
    print(f"\nWrote CSVs to {output_dir}/, split into {len(model_labels)} model folder(s):")
    for model_label in model_labels:
        model_dir = os.path.join(output_dir, _slugify_model_label(model_label))
        os.makedirs(model_dir, exist_ok=True)
        print(f"  {model_label} -> {model_dir}/")
        prefix = _compact_model_prefix(model_label)
        for filename, df in all_csvs.items():
            # Guard against a fully-empty DataFrame (e.g. no RAG samples at
            # all, so retrieval_attribution_df has no rows and no columns)
            # — filtering by a nonexistent column would raise a KeyError.
            subset = df[df["model_label"] == model_label] if "model_label" in df.columns else df.iloc[0:0]
            prefixed_filename = f"{prefix}_{filename}"
            subset.to_csv(os.path.join(model_dir, prefixed_filename), index=False)
            print(f"    {prefixed_filename}: {len(subset)} rows")

        # Counts actual log.samples, not per_epoch_df rows — the latter
        # also includes the T2DM audit re-scoring rows, which are a re-score
        # of an already-generated sample, not a separate LLM call.
        n_samples = sum(
            1 for s in log.samples if s.metadata["model"]["label"] == model_label
        )
        run_metadata_df = pd.DataFrame([{
            "model_label": model_label,
            "n_samples": n_samples,
            "started_at": log.stats.started_at,
            "completed_at": log.stats.completed_at,
            "duration_seconds": duration_seconds,
            "duration_hms": duration_hms,
        }])
        run_metadata_filename = f"{prefix}_run_metadata.csv"
        run_metadata_df.to_csv(os.path.join(model_dir, run_metadata_filename), index=False)
        print(f"    {run_metadata_filename}: total runtime {duration_hms} for {n_samples} samples")


# ── Run it ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RERUN_EXPERIMENTS:
        raise SystemExit("RERUN_EXPERIMENTS is False — set True to run.")

    logs = eval(codelist_task(), model="mockllm/model", log_dir=LOG_DIR)
    derive_csvs(logs[0], OUTPUT_DIR)
