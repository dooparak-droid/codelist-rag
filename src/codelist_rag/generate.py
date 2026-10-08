"""generate.py

Prompt building, LLM execution through Inspect AI's model interface (the
same call path as the thesis primary runs), retry handling, and full
pipeline orchestration.

The prompt text below is copied verbatim from the thesis pipeline
(pipeline_core.py, tag thesis-v1). Do not edit it without re-running the
evaluation, because the thesis results apply only to these exact prompts.
"""

import asyncio
from typing import Any

from dotenv import load_dotenv
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from codelist_rag.parse import parse_codelist_response
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.terminology import TerminologyStore
from codelist_rag.validate import check_fabrication

load_dotenv()

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.5"
MAX_TOKENS = 12000  # as in the thesis runs
MAX_RETRIES = 3

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
    Returns the user-turn content only; the system prompt is sent separately.
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


PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "chain_of_thought": build_cot_prompt,
}


def _to_inspect_base_url(provider: str, base_url: str | None) -> str | None:
    """Inspect's Ollama provider expects the /v1 suffix to be given explicitly."""
    if base_url and provider == "ollama":
        return base_url.rstrip("/") + "/v1"
    return base_url


async def _generate_text_async(
    prompt: str, provider: str, model: str, base_url: str | None
) -> str:
    """One model call through Inspect's get_model()/generate()."""
    llm = get_model(f"{provider}/{model}", base_url=_to_inspect_base_url(provider, base_url))
    response = await llm.generate(
        input=[ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=prompt)],
        config=GenerateConfig(max_tokens=MAX_TOKENS),
    )
    return response.completion


def generate_text(prompt: str, provider: str, model: str, base_url: str | None = None) -> str:
    """Synchronous wrapper around one model call. Tests replace this function."""
    return asyncio.run(_generate_text_async(prompt, provider, model, base_url))


def call_llm(
    prompt: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[dict[str, str]], str, int, str | None]:
    """
    Call the model and parse its JSON response. Retries up to max_retries on
    malformed or truncated output and on API errors.

    Returns:
        (parsed_codes, raw_text, n_attempts, error_message)
    """
    last_error = None
    raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            raw_text = generate_text(prompt, provider, model, base_url)
            codes = parse_codelist_response(raw_text)
            return codes, raw_text, attempt, None
        except Exception as e:
            last_error = f"Attempt {attempt}/{max_retries} failed: {e}"

    return [], raw_text, max_retries, last_error


class CodelistPipeline:
    """Orchestrates retrieval, prompt construction, LLM generation, parsing, and fabrication checks."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        terminology: TerminologyStore | None = None,
    ):
        self.retriever = retriever
        self.terminology = terminology

    def generate(
        self,
        query: str,
        strategy: str = "zero_shot",
        use_rag: bool = True,
        n_results: int = 400,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Generate a validated codelist for a clinical condition."""
        if strategy not in PROMPT_BUILDERS:
            raise ValueError(f"Unknown strategy: {strategy}")
        prompt_builder = PROMPT_BUILDERS[strategy]

        retrieved_concepts = None
        if use_rag and self.retriever:
            retrieved_concepts = self.retriever.retrieve_hybrid(query, n_results=n_results)

        prompt = prompt_builder(query, retrieved_concepts)
        raw_codes, raw_text, attempts, error = call_llm(
            prompt, provider=provider, model=model, base_url=base_url
        )

        if error or not raw_codes:
            return {
                "condition": query,
                "success": False,
                "error": error or "Failed to parse valid codelist JSON",
                "codes": [],
                "attempts": attempts,
                "strategy": strategy,
                "use_rag": use_rag,
            }

        # Verification check
        if self.terminology:
            validation = check_fabrication(raw_codes, self.terminology)
            annotated_codes = validation["annotated_codes"]
            fab_rate = validation["fabrication_rate"]
        else:
            annotated_codes = [{"code": c["code"], "term": c["term"], "code_is_real": None} for c in raw_codes]
            fab_rate = None

        return {
            "condition": query,
            "success": True,
            "error": None,
            "codes": annotated_codes,
            "total_codes": len(annotated_codes),
            "fabrication_rate": fab_rate,
            "n_retrieved": len(retrieved_concepts) if retrieved_concepts else 0,
            "retrieved_codes": [c["code"] for c in (retrieved_concepts or [])],
            "attempts": attempts,
            "strategy": strategy,
            "use_rag": use_rag,
            "model": f"{provider}:{model}",
        }
