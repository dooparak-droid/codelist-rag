"""generate.py

Prompt building, LLM execution (via chatlas), retry handling, and full pipeline orchestration.
"""

from typing import Any
import os
import chatlas
from dotenv import load_dotenv

from codelist_rag.parse import parse_codelist_response
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.terminology import TerminologyStore
from codelist_rag.validate import check_fabrication

load_dotenv()

SYSTEM_PROMPT = "You are a clinical terminology expert supporting EHR research in UK primary care."

FEW_SHOT_EXAMPLES = """
Example 1: Depression
[
  {"code": "35489007", "term": "Depressive disorder"},
  {"code": "310495003", "term": "Mild depression"},
  {"code": "36923009", "term": "Major depression, single episode"}
]

Example 2: Hypertension
[
  {"code": "38341003", "term": "Hypertensive disorder"},
  {"code": "59621000", "term": "Essential hypertension"},
  {"code": "46481004", "term": "Low-renin essential hypertension"}
]
"""

PROVIDER_CHAT_CLASSES = {
    "openai": chatlas.ChatOpenAI,
    "anthropic": chatlas.ChatAnthropic,
    "google": chatlas.ChatGoogle,
    "ollama": chatlas.ChatOllama,
}


def build_zero_shot_prompt(query: str, retrieved_concepts: list[dict[str, str]] | None = None) -> str:
    """Build a zero-shot codelist generation prompt."""
    if retrieved_concepts:
        context_lines = "\n".join([f"- {c['code']}: {c['term']}" for c in retrieved_concepts])
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
- Be comprehensive — include ALL clinically relevant codes. A valid UK primary care codelist may contain dozens of codes.
- You may supplement with additional valid SNOMED-CT codes from your own knowledge where clinically justified
- Codes must be valid SNOMED-CT concept identifiers
- Return JSON only, no preamble, no markdown

Return in this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]"""


def build_few_shot_prompt(query: str, retrieved_concepts: list[dict[str, str]] | None = None) -> str:
    """Build a few-shot prompt with exemplar codelists."""
    if retrieved_concepts:
        context_lines = "\n".join([f"- {c['code']}: {c['term']}" for c in retrieved_concepts])
        context_block = f"""
The following SNOMED-CT concepts have been retrieved as potentially relevant:

RETRIEVED CONCEPTS:
{context_lines}

From the retrieved concepts above, generate a comprehensive SNOMED-CT codelist for: {query}."""
    else:
        context_block = f"Using your own clinical knowledge, generate a SNOMED-CT codelist for: {query}"

    return f"""You are building a SNOMED-CT codelist for a research cohort.

Here are examples of well-formed SNOMED-CT codelists for other conditions:

{FEW_SHOT_EXAMPLES}

{context_block}

Requirements:
- Be comprehensive — include ALL clinically relevant codes.
- Follow the same JSON format as demonstrated in the examples above.
- Return JSON only, no preamble, no markdown.

Return in this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]"""


def build_cot_prompt(query: str, retrieved_concepts: list[dict[str, str]] | None = None) -> str:
    """Build a chain-of-thought prompt requiring explicit clinical reasoning steps."""
    if retrieved_concepts:
        context_lines = "\n".join([f"- {c['code']}: {c['term']}" for c in retrieved_concepts])
        context_block = f"""
The following SNOMED-CT concepts have been retrieved as potentially relevant for: {query}

RETRIEVED CONCEPTS:
{context_lines}"""
    else:
        context_block = f"Using your own clinical knowledge, generate a SNOMED-CT codelist for: {query}"

    return f"""You are building a SNOMED-CT codelist for a research cohort.

{context_block}

Before generating the codelist, reason step by step:
1. Precise clinical definition and boundaries of {query}.
2. Subtypes and variants.
3. Complications and management states.

Now generate the codelist following this exact format:
[
  {{"code": "SNOMED_CODE", "term": "DESCRIPTION"}}
]

Requirements:
- Include ALL clinically relevant codes.
- Return your reasoning first, followed by the JSON codelist."""


PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "chain_of_thought": build_cot_prompt,
}


def call_llm(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    max_retries: int = 3,
) -> tuple[list[dict[str, str]], str, int, str | None]:
    """
    Call LLM provider via chatlas and parse JSON response.
    Retries up to max_retries on malformed output or transient errors.

    Returns:
        (parsed_codes, raw_text, n_attempts, error_message)
    """
    chat_class = PROVIDER_CHAT_CLASSES.get(provider)
    if not chat_class:
        raise ValueError(f"Unsupported provider: {provider}")

    kwargs: dict[str, Any] = {"model": model, "system_prompt": SYSTEM_PROMPT}
    if base_url:
        kwargs["base_url"] = base_url

    last_error = None
    raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            chat = chat_class(**kwargs)
            response = chat.chat(prompt, echo="none", stream=False)
            raw_text = str(response)
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
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Generate a validated codelist for a clinical condition."""
        prompt_builder = PROMPT_BUILDERS.get(strategy, build_zero_shot_prompt)

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
