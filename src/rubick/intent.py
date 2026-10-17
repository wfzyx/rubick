"""
Intent detection and parsing for Rubick Gateway.
Enables auto-routing /completions requests to specialized heads (Decision, Embedding)
bypassing expensive autoregressive decodes.
"""
import re
from typing import Any

EMBED_PATTERNS = [
    re.compile(
        r"^\s*(?:please\s+)?(?:generate\s+)?(?:extract\s+)?(?:get\s+)?(?:embed|embedding|vectorize|semantic vector)(?:\s+this)?(?:\s+for)?(?:\s+of)?(?:\s+text)?\s*:\s*(.+)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"^\s*(?:embed|vectorize)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"^\s*(?:calculate|compute)\s+(?:the\s+)?(?:embedding|vector)\s+(?:for|of)\s*:\s*(.+)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"^\s*embed:\s*(.+)$", re.IGNORECASE | re.DOTALL),
]


def parse_embedding_intent(text: str, body: dict[str, Any] | None = None) -> str | None:
    """
    Detect if the user prompt or body is requesting an embedding extraction.
    Returns the target text to embed, or None.
    """
    if body:
        if body.get("intent") in ("embed", "embedding", "vectorize") or body.get("mode") == "embedding":
            return text or body.get("input") or body.get("text")
        if body.get("return_last_hidden_state") is True:
            return text

    for pattern in EMBED_PATTERNS:
        m = pattern.match(text)
        if m:
            extracted = m.group(1).strip()
            if extracted:
                return extracted
    return None


def parse_decision_options(
    text: str, body: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, str]] | None:
    """
    Detect if the user prompt or body contains categorical decision options.
    Returns (state, question, options_dict) or None.
    """
    if body and "options" in body and isinstance(body["options"], dict):
        return body.get("state", text), body.get("question", "Select the best option:"), body["options"]

    # Match A) Option text, A. Option text, A: Option text, [A] Option text, 1) Option text
    pattern_letter = re.compile(
        r"(?:^|\n)\s*(?:\[)?([A-Z0-9])(?:\]|\)|\:|\.)\s+([^\n]+)", re.MULTILINE
    )
    matches = pattern_letter.findall(text)
    if len(matches) >= 2:
        keys = [m[0].upper() for m in matches]
        # Only treat as options if keys are unique and non-empty
        if len(keys) == len(set(keys)) and len(keys) <= 26:
            options = {m[0].upper(): m[1].strip() for m in matches}
            first_match_start = text.find(matches[0][1])
            preamble = text[:first_match_start].strip()
            # Strip trailing option label prefix from preamble if left behind
            preamble = re.sub(
                r"(?:^|\n)\s*(?:\[)?" + re.escape(matches[0][0]) + r"(?:\]|\)|\:|\.)\s*$",
                "",
                preamble,
            ).strip()
            return preamble, "Select the best option:", options
    return None


def detect_intent(
    messages: list[dict[str, Any]], body: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """
    Main intent classifier for Rubick.
    Returns:
      ("embedding", {"text": ...})
      ("decision", {"state": ..., "question": ..., "options": ...})
      ("chat", {})
    """
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"].strip()
            break

    # 1. Check embedding intent
    embed_target = parse_embedding_intent(last_user_msg, body)
    if embed_target:
        return "embedding", {"text": embed_target}

    # 2. Check decision intent
    decision_tuple = parse_decision_options(last_user_msg, body)
    if decision_tuple:
        state, question, options = decision_tuple
        return "decision", {"state": state, "question": question, "options": options}

    # 3. Default: standard generative chat
    return "chat", {}
