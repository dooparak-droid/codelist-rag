"""test_parse.py

Unit tests for LLM response parsing and JSON extraction.
"""

import pytest
from codelist_rag.parse import extract_json_array, parse_codelist_response


def test_extract_json_array_valid():
    raw = '```json\n[{"code": "123", "term": "Test"}]\n```'
    extracted = extract_json_array(raw)
    assert extracted == '[{"code": "123", "term": "Test"}]'


def test_extract_json_array_with_reasoning():
    raw = """Here is my reasoning:
1. First step.
2. Second step.

[
  {"code": "123", "term": "Test"}
]"""
    extracted = extract_json_array(raw)
    assert extracted == '[\n  {"code": "123", "term": "Test"}\n]'


def test_extract_json_array_missing_brackets():
    raw = "No JSON here at all"
    with pytest.raises(ValueError, match="No JSON array found"):
        extract_json_array(raw)


def test_parse_codelist_response_valid():
    raw = '[{"code": "99900001", "term": "Synthetic condition"}]'
    result = parse_codelist_response(raw)
    assert len(result) == 1
    assert result[0] == {"code": "99900001", "term": "Synthetic condition"}


def test_parse_codelist_response_malformed_entry():
    raw = '[{"code": "99900001"}]'
    with pytest.raises(ValueError, match="Malformed code entry"):
        parse_codelist_response(raw)


def test_parse_codelist_response_empty_code():
    raw = '[{"code": "", "term": "Term"}]'
    with pytest.raises(ValueError, match="Empty code or term"):
        parse_codelist_response(raw)
