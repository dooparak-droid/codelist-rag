"""test_generate.py

Unit tests for prompt building, retry handling and pipeline orchestration.
The model call is replaced, so no API key is needed.
"""

import pytest

from codelist_rag import generate
from codelist_rag.generate import CodelistPipeline, build_zero_shot_prompt, call_llm
from codelist_rag.terminology import TerminologyStore

VALID = '[{"code": "99900001", "term": "Synthetic mild airway reactivity"}]'


def fake_llm(responses):
    """Return a stand-in for generate_text that yields each response in turn."""
    calls = []

    def _fake(prompt, provider, model, base_url=None):
        calls.append((provider, model))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    _fake.calls = calls
    return _fake


def test_zero_shot_prompt_contains_retrieved_concepts():
    prompt = build_zero_shot_prompt("Asthma", [{"code": "99900001", "term": "Mild"}])
    assert "- 99900001: Mild" in prompt
    assert "Asthma" in prompt


def test_call_llm_retries_after_malformed_output(monkeypatch):
    fake = fake_llm(["not json", VALID])
    monkeypatch.setattr(generate, "generate_text", fake)
    codes, _, attempts, error = call_llm("prompt")
    assert attempts == 2
    assert error is None
    assert codes[0]["code"] == "99900001"


def test_call_llm_gives_up_after_max_retries(monkeypatch):
    fake = fake_llm(["not json"])
    monkeypatch.setattr(generate, "generate_text", fake)
    codes, _, attempts, error = call_llm("prompt", max_retries=3)
    assert codes == []
    assert attempts == 3
    assert "Attempt 3/3 failed" in error


def test_pipeline_returns_failure_not_empty_success(monkeypatch):
    monkeypatch.setattr(generate, "generate_text", fake_llm(["not json"]))
    result = CodelistPipeline().generate("Asthma", use_rag=False)
    assert result["success"] is False
    assert result["codes"] == []


def test_pipeline_flags_fabricated_code(monkeypatch):
    response = '[{"code": "99900001", "term": "A"}, {"code": "11111111", "term": "B"}]'
    monkeypatch.setattr(generate, "generate_text", fake_llm([response]))
    store = TerminologyStore({"99900001": "A"})
    result = CodelistPipeline(terminology=store).generate("Asthma", use_rag=False)
    flags = {c["code"]: c["code_is_real"] for c in result["codes"]}
    assert flags == {"99900001": True, "11111111": False}
    assert result["fabrication_rate"] == 0.5


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        CodelistPipeline().generate("Asthma", strategy="made_up", use_rag=False)
