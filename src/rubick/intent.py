"""
Intent detection and parsing for Rubick Gateway.
Enables auto-routing /completions requests to specialized heads (Decision, Embedding)
bypassing expensive autoregressive decodes.
Combines Tier-1 structural cues with a Tier-2 System-1 Decision Head probe.
"""
import math
import re
from typing import Any

import httpx

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

FEWSHOT_INTENT_ROUTER_PROMPT = """Task: Classify API request into EMBED, DECISION, or CHAT.

Request: "Extract 768-dim vector for this sentence."
Class: EMBED

Request: "Which option is correct?\nA) True\nB) False"
Class: DECISION

Request: "Explain quantum computing in simple terms."
Class: CHAT

Request: "{user_prompt}"
Class: """

ROUTER_TARGETS = ["EMBED", "DECISION", "CHAT"]


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


async def classify_intent_with_decision_head(
    client: httpx.AsyncClient, backend_url: str, text: str
) -> tuple[str, dict[str, float]]:
    """
    Use Rubick's 1-token System-1 Decision Head to classify intent semantically.
    Returns (dominant_intent, probabilities_dict).
    """
    prompt = FEWSHOT_INTENT_ROUTER_PROMPT.format(user_prompt=text)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"enable_thinking": False},
    }

    try:
        resp = await client.post(f"{backend_url}/v1/chat/completions", json=payload, timeout=5.0)
        if resp.status_code != 200:
            return "chat", {"CHAT": 1.0}
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        top_lps = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

        scores = {}
        for item in top_lps:
            raw_tok = item.get("token", "").strip().upper()
            for tgt in ROUTER_TARGETS:
                if tgt.startswith(raw_tok) or raw_tok.startswith(tgt):
                    if tgt not in scores:
                        scores[tgt] = item.get("logprob", -100.0)

        raw = {k: math.exp(scores.get(k, -100.0)) for k in ROUTER_TARGETS}
        tot = sum(raw.values())
        if not tot:
            return "chat", {"CHAT": 1.0}
        probs = {k: round(v / tot, 4) for k, v in raw.items()}
        best = max(probs, key=probs.get) if max(probs.values()) > 0.0 else "CHAT"

        mode_map = {
            "EMBED": "embedding",
            "DECISION": "decision",
            "CHAT": "chat",
        }
        return mode_map.get(best, "chat"), probs
    except Exception:
        return "chat", {"CHAT": 1.0}


async def detect_intent(
    messages: list[dict[str, Any]],
    body: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
    backend_url: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Main intent classifier for Rubick.
    Tier 1: Fast structural & explicit cues (< 0.1ms)
    Tier 2: Model-based System-1 Decision Head semantic probe (~50ms)
    Returns:
      ("embedding", {"text": ..., "confidence": ...})
      ("decision", {"state": ..., "question": ..., "options": ..., "confidence": ...})
      ("chat", {"confidence": ...})
    """
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"].strip()
            break

    # Tier 1: Check fast structural and explicit cues first
    embed_target = parse_embedding_intent(last_user_msg, body)
    if embed_target:
        return "embedding", {"text": embed_target, "confidence": 1.0, "source": "structural"}

    decision_tuple = parse_decision_options(last_user_msg, body)
    if decision_tuple:
        state, question, options = decision_tuple
        return "decision", {
            "state": state,
            "question": question,
            "options": options,
            "confidence": 1.0,
            "source": "structural",
        }

    # Tier 2: Model-based Decision Head probe
    if client and backend_url and last_user_msg:
        semantic_intent, probs = await classify_intent_with_decision_head(
            client, backend_url, last_user_msg
        )
        confidence = probs.get(semantic_intent.upper(), 0.0)
        if semantic_intent == "embedding" and confidence > 0.60:
            return "embedding", {
                "text": last_user_msg,
                "confidence": confidence,
                "source": "decision_head",
            }
        elif semantic_intent == "decision" and confidence > 0.60:
            dec_opts = parse_decision_options(last_user_msg, body)
            if dec_opts:
                state, question, options = dec_opts
                return "decision", {
                    "state": state,
                    "question": question,
                    "options": options,
                    "confidence": confidence,
                    "source": "decision_head",
                }

    # Default: standard generative chat
    return "chat", {"confidence": 1.0, "source": "default"}
