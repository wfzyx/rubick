"""
Intent detection and parsing for Rubick Gateway.
Enables auto-routing /completions requests to specialized heads (Decision, Embedding)
bypassing expensive autoregressive decodes.

Uses LLM-based recursive extraction: queries the model directly to detect decision
intent and extract structured question and options as JSON. Zero fragile regexes.
"""
import json
from typing import Any

import httpx

DECISION_PARSER_PROMPT = """You are a decision parser. Analyze the user request.
If the user wants to decide, choose, pick, or classify between alternatives/options, extract the question and options as compact JSON.
If it is NOT a decision between alternatives (e.g. general questions, instructions, notes, todo lists, storytelling), return {{"is_decision": false}}.

User: "which should I pick? A) Red B) Blue"
JSON: {{"is_decision": true, "question": "Which should I pick?", "options": ["Red", "Blue"]}}

User: "decision: which of these should I eat\\n- pizza\\n- sushi\\n- salad"
JSON: {{"is_decision": true, "question": "Which of these should I eat?", "options": ["pizza", "sushi", "salad"]}}

User: "what should I choose between pizza, sushi, or salad?"
JSON: {{"is_decision": true, "question": "What should I choose?", "options": ["pizza", "sushi", "salad"]}}

User: "Should I pick AWS, GCP, or Azure for hosting?"
JSON: {{"is_decision": true, "question": "Which cloud provider should I pick?", "options": ["AWS", "GCP", "Azure"]}}

User: "Here are the notes from our sync:\\n- deploy v1.2\\n- review PR 30732"
JSON: {{"is_decision": false}}

User: "Tell me a short bedtime story about a friendly robot."
JSON: {{"is_decision": false}}

User: "{user_prompt}"
JSON:"""


def check_embedding_intent(text: str, body: dict[str, Any] | None = None) -> str | None:
    """
    Check if request is an embedding extraction.
    Simple string prefixes and explicit payload flags.
    """
    if body:
        if body.get("intent") in ("embed", "embedding", "vectorize") or body.get("mode") == "embedding":
            return text or body.get("input") or body.get("text")
        if body.get("return_last_hidden_state") is True:
            return text

    clean = text.strip()
    clean_lower = clean.lower()
    for prefix in ("embed:", "embedding:", "vectorize:"):
        if clean_lower.startswith(prefix):
            target = clean[len(prefix):].strip()
            if target:
                return target

    return None


async def extract_decision_via_llm(
    client: httpx.AsyncClient, backend_url: str, text: str
) -> tuple[str, dict[str, str]] | None:
    """
    Recursive LLM call: uses the model itself to parse the user's prompt into
    a clean question and normalized options dictionary {"A": ..., "B": ...}.
    Returns (question, options_dict) or None if not a decision.
    """
    prompt = DECISION_PARSER_PROMPT.format(user_prompt=text)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"enable_thinking": False},
    }

    try:
        resp = await client.post(f"{backend_url}/v1/chat/completions", json=payload, timeout=6.0)
        if resp.status_code != 200:
            return None
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        parsed = json.loads(raw[start : end + 1])
        if not parsed.get("is_decision"):
            return None

        raw_opts = parsed.get("options")
        if not raw_opts:
            return None

        options = {}
        if isinstance(raw_opts, list):
            options = {
                chr(ord("A") + i): str(opt).strip()
                for i, opt in enumerate(raw_opts)
                if str(opt).strip()
            }
        elif isinstance(raw_opts, dict):
            keys = list(raw_opts.keys())
            if all(len(k) == 1 and k.isalpha() for k in keys):
                options = {k.upper(): str(v).strip() for k, v in raw_opts.items()}
            else:
                options = {
                    chr(ord("A") + i): str(v).strip()
                    for i, v in enumerate(raw_opts.values())
                }

        if len(options) >= 2 and len(options) <= 26:
            question = parsed.get("question", text).strip()
            return question, options
    except Exception:
        pass
    return None

def parse_explicit_options(
    text: str, body: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, str]] | None:
    """Check for explicit body options {"options": {"A": ...}} or structured A) B) patterns."""
    if body and "options" in body and isinstance(body["options"], dict):
        return body.get("state", text), body.get("question", "Select the best option:"), body["options"]

    import re
    clean_text = text.strip()
    if clean_text.lower().startswith("decision:"):
        clean_text = clean_text[len("decision:"):].strip()

    pattern_letter = re.compile(
        r"(?:^|\n)\s*(?:\[)?([A-Z0-9])(?:\]|\)|\:|\.)\s+([^\n]+)", re.MULTILINE
    )
    matches = pattern_letter.findall(clean_text)
    if len(matches) >= 2:
        keys = [m[0].upper() for m in matches]
        if len(keys) == len(set(keys)) and len(keys) <= 26:
            options = {m[0].upper(): m[1].strip() for m in matches}
            first_match_start = clean_text.find(matches[0][1])
            preamble = clean_text[:first_match_start].strip()
            preamble = re.sub(
                r"(?:^|\n)\s*(?:\[)?" + re.escape(matches[0][0]) + r"(?:\]|\)|\:|\.)\s*$",
                "",
                preamble,
            ).strip()
            question = preamble or "Select the best option:"
            return clean_text, question, options

    return None


async def detect_intent(
    messages: list[dict[str, Any]],
    body: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
    backend_url: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Main intent classifier for Rubick.
    1. Embedding check: simple string prefix or explicit body flag.
    2. Explicit options in body: {"options": {"A": ...}}.
    3. Recursive LLM parsing: let the model extract question and options as JSON.
    """
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"].strip()
            break

    # 1. Embedding check
    embed_target = check_embedding_intent(last_user_msg, body)
    if embed_target:
        return "embedding", {"text": embed_target, "confidence": 1.0, "source": "explicit"}

    # 2. Explicit options in payload or obvious A) B) pattern
    explicit_tuple = parse_explicit_options(last_user_msg, body)
    if explicit_tuple:
        state, question, options = explicit_tuple
        return "decision", {
            "state": state,
            "question": question,
            "options": options,
            "confidence": 1.0,
            "source": "explicit",
        }
    # 3. Recursive LLM decision extraction
    if client and backend_url and last_user_msg:
        extracted = await extract_decision_via_llm(client, backend_url, last_user_msg)
        if extracted:
            question, options = extracted
            return "decision", {
                "state": last_user_msg,
                "question": question,
                "options": options,
                "confidence": 0.95,
                "source": "llm_recursive",
            }

    # Default: standard generative chat
    return "chat", {"confidence": 1.0, "source": "default"}
