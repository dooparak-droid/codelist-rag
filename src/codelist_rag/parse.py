"""parse.py

Robust parsing and structural validation of raw LLM JSON codelist output.
"""

import json
import re


def extract_json_array(raw_text: str) -> str:
    """
    Extract the JSON array string from raw LLM output text,
    stripping markdown code fences and pre-existing thinking/reasoning blocks.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response")

    return text[start : end + 1]


def parse_codelist_response(raw_text: str) -> list[dict[str, str]]:
    """
    Parse a raw LLM text completion into a validated list of {"code": str, "term": str} dicts.
    Raises ValueError on malformed, missing, or truncated output.
    """
    json_str = extract_json_array(raw_text)
    codes = json.loads(json_str)

    if not isinstance(codes, list):
        raise ValueError("Parsed JSON is not a list")

    parsed = []
    for item in codes:
        if not isinstance(item, dict) or "code" not in item or "term" not in item:
            raise ValueError(f"Malformed code entry: {item!r}")
        code = str(item["code"]).strip()
        term = str(item["term"]).strip()
        if not code or not term:
            raise ValueError(f"Empty code or term in entry: {item!r}")
        parsed.append({"code": code, "term": term})

    return parsed
