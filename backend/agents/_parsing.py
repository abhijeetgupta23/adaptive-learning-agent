from __future__ import annotations

import json
from typing import Any


class StructuredOutputError(Exception):
    """Raised when an LLM's structured output cannot be extracted or parsed."""


def extract_text(response: Any) -> str:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    raise StructuredOutputError("Response contained no text block")


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_balanced_object(stripped)
        if candidate is None:
            raise StructuredOutputError(
                f"Response was not valid JSON: {text!r}"
            ) from None
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"Response was not valid JSON: {text!r}"
            ) from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"Response was not a JSON object: {text!r}")
    return payload


def _extract_balanced_object(text: str) -> str | None:
    """Find the first balanced `{...}` substring in `text`. Returns None if none."""
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        # unbalanced from here; try next `{`
    return None
