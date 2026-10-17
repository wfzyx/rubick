import json
import math
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from rubick.config import settings
from rubick.intent import detect_intent
from rubick.normalizer import l2_normalize
from rubick.prompts import (
    format_decision_prompt,
    format_embedding_prompt,
    format_guardian_prompt,
    format_multi_decision_prompt,
)

app = FastAPI(
    title="Rubick Gateway",
    description="Unified 4-in-1 AI Gateway for Chat, Decisions, Safety Guardrails, and Embeddings on llama-server",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Verify backend connectivity."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{settings.llama_server_url}/health")
            server_healthy = resp.status_code == 200
        except Exception:
            server_healthy = False

    return {
        "status": "ok" if server_healthy else "degraded",
        "backend": settings.llama_server_url,
        "backend_online": server_healthy,
        "features": {
            "chat": True,
            "embeddings": True,
            "decisions": True,
            "guardian": True,
        },
    }


@app.get("/v1/models")
async def list_models():
    """Proxy model listing from llama-server."""
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        try:
            resp = await client.get(f"{settings.llama_server_url}/v1/models")
            return resp.json()
        except Exception as exc:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "rubick-unified-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "rubick",
                    }
                ],
            }


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """
    OpenAI-compatible embeddings endpoint.
    Converts text to PromptEOL 1-token generative calls, extracting the last hidden state.
    """
    body = await request.json()
    input_data = body.get("input")
    if not input_data:
        raise HTTPException(status_code=400, detail="Missing required 'input' field")

    texts = [input_data] if isinstance(input_data, str) else list(input_data)
    model = body.get("model", "default")

    results = []
    total_prompt_tokens = 0

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        for idx, text in enumerate(texts):
            prompt = format_embedding_prompt(text, template=settings.embedding_prompt_template)
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "return_last_hidden_state": True,
                "temperature": 0.0,
            }

            resp = await client.post(
                f"{settings.llama_server_url}/v1/chat/completions",
                json=payload,
            )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Backend error from llama-server: {resp.text}",
                )

            data = resp.json()
            vector = data.get("last_hidden_state")
            if not vector and "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                vector = msg.get("last_hidden_state")

            if not vector:
                raise HTTPException(
                    status_code=500,
                    detail="Backend did not return 'last_hidden_state'. Ensure llama-server was started with --embedding --pooling none.",
                )

            norm_vec = l2_normalize(vector)
            results.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": norm_vec,
                }
            )

            if "usage" in data:
                total_prompt_tokens += data["usage"].get("prompt_tokens", 0)

    return {
        "object": "list",
        "data": results,
        "model": model,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_prompt_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint with auto-guardian and pass-through.
    """
    body = await request.json()

    model = body.get("model", "default")
    messages = body.get("messages", [])

    # 1. Intent Hijacking (when enabled via request body or settings)
    intent_parser_active = body.get("intent_parser", settings.intent_parser)
    if intent_parser_active and messages:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            intent_type, parsed_args = await detect_intent(
                messages, body, client=client, backend_url=settings.llama_server_url
            )

            if intent_type == "embedding":
                target_text = parsed_args["text"]
                prompt = format_embedding_prompt(target_text, template=settings.embedding_prompt_template)
                resp = await client.post(
                    f"{settings.llama_server_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1,
                        "return_last_hidden_state": True,
                        "temperature": 0.0,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    vector = data.get("last_hidden_state")
                    if not vector and "choices" in data and data["choices"]:
                        msg = data["choices"][0].get("message", {})
                        vector = msg.get("last_hidden_state")
                    if vector:
                        norm_vec = l2_normalize(vector)
                        prompt_toks = data.get("usage", {}).get("prompt_tokens", len(target_text.split()))
                        source = parsed_args.get("source", "structural")
                        confidence = str(parsed_args.get("confidence", 1.0))
                        return JSONResponse(
                            content={
                                "id": f"chatcmpl-rubick-embed-{int(time.time()*1000)}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": f"[Rubick: Extracted {len(norm_vec)}-dim embedding vector]",
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": prompt_toks,
                                    "completion_tokens": 1,
                                    "total_tokens": prompt_toks + 1,
                                },
                                "rubick_routed_mode": "embedding",
                                "embedding": norm_vec,
                                "last_hidden_state": norm_vec,
                            },
                            headers={
                                "X-Rubick-Mode": "embedding",
                                "X-Rubick-Hijacked": "true",
                                "X-Rubick-Intent-Source": source,
                                "X-Rubick-Intent-Confidence": confidence,
                            },
                        )

            elif intent_type == "decision":
                state = parsed_args["state"]
                question = parsed_args["question"]
                options = parsed_args["options"]
                prompt = format_decision_prompt(state, question, options)
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "extra_body": {"enable_thinking": False},
                }
                resp = await client.post(
                    f"{settings.llama_server_url}/v1/chat/completions",
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    top_lps = (
                        choice.get("logprobs", {})
                        .get("content", [{}])[0]
                        .get("top_logprobs", [])
                    )
                    option_logprobs = {}
                    for item in top_lps:
                        tok = item.get("token", "").strip().upper()
                        if tok in options:
                            option_logprobs[tok] = item.get("logprob", -100.0)
                    raw_scores = {k: math.exp(option_logprobs.get(k, -100.0)) for k in options}
                    total = sum(raw_scores.values()) or 1.0
                    probabilities = {k: v / total for k, v in raw_scores.items()}
                    best_opt = max(probabilities, key=probabilities.get)
                    best_desc = options.get(best_opt, "")
                    confidence_pct = round(probabilities[best_opt] * 100, 1)

                    # Generate concise natural language summary/explanation if enabled
                    should_explain = body.get("explain", body.get("include_summary", settings.explain))
                    explanation = ""
                    if should_explain:
                        explain_prompt = f"State: {state}\nDecision: Option {best_opt} ({best_desc} - {confidence_pct}% confidence)\n\nExplain this decision to the user in one concise sentence:"
                        try:
                            exp_resp = await client.post(
                                f"{settings.llama_server_url}/v1/chat/completions",
                                json={
                                    "messages": [{"role": "user", "content": explain_prompt}],
                                    "max_tokens": 50,
                                    "temperature": 0.3,
                                    "chat_template_kwargs": {"enable_thinking": False},
                                    "extra_body": {"enable_thinking": False},
                                },
                            )
                            if exp_resp.status_code == 200:
                                exp_data = exp_resp.json()
                                exp_content = (
                                    exp_data.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content", "")
                                    .strip()
                                )
                                if exp_content:
                                    explanation = exp_content
                        except Exception:
                            explanation = ""

                    if explanation:
                        content_str = (
                            f"{best_opt}) {best_desc} ({confidence_pct}% confidence) — {explanation}"
                            if best_desc
                            else f"Option {best_opt} ({confidence_pct}% confidence) — {explanation}"
                        )
                    else:
                        content_str = (
                            f"{best_opt}) {best_desc} ({confidence_pct}% confidence)"
                            if best_desc
                            else f"Option {best_opt} ({confidence_pct}% confidence)"
                        )

                    prompt_toks = data.get("usage", {}).get("prompt_tokens", 10)
                    source = parsed_args.get("source", "structural")
                    confidence = str(parsed_args.get("confidence", 1.0))
                    completion_toks = len(content_str.split())

                    return JSONResponse(
                        content={
                            "id": f"chatcmpl-rubick-decision-{int(time.time()*1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": content_str,
                                    },
                                    "finish_reason": "stop",
                                },
                            ],
                            "usage": {
                                "prompt_tokens": prompt_toks,
                                "completion_tokens": completion_toks,
                                "total_tokens": prompt_toks + completion_toks,
                            },
                            "rubick_routed_mode": "decision",
                            "decision": {
                                "best_option": best_opt,
                                "best_description": best_desc,
                                "probabilities": probabilities,
                                "confidence_pct": confidence_pct,
                                "explanation": explanation or None,
                            },
                        },
                        headers={
                            "X-Rubick-Mode": "decision",
                            "X-Rubick-Hijacked": "true",
                            "X-Rubick-Intent-Source": source,
                            "X-Rubick-Intent-Confidence": confidence,
                            "X-Rubick-Explained": "true" if explanation else "false",
                        },
                    )
    # 2. Auto-Guardian pre-flight screening if requested or globally active
    perform_guardian = body.get("auto_guardian", settings.auto_guardian)

    if perform_guardian and messages:
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if isinstance(last_user_msg, str) and last_user_msg.strip():
            guardian_prompt = format_guardian_prompt(last_user_msg)
            async with httpx.AsyncClient(timeout=10.0) as client:
                g_resp = await client.post(
                    f"{settings.llama_server_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": guardian_prompt}],
                        "max_tokens": 16,
                        "temperature": 0.0,
                    },
                )
                if g_resp.status_code == 200:
                    g_data = g_resp.json()
                    answer = (
                        g_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .lower()
                    )
                    if "yes" in answer and "safe" not in answer:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": {
                                    "message": "Input rejected by Rubick Guardian safety check.",
                                    "type": "content_policy_violation",
                                    "code": 400,
                                }
                            },
                        )

    # Proxy request to llama-server
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        resp = await client.post(
            f"{settings.llama_server_url}/v1/chat/completions",
            json=body,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers={
                "X-Rubick-Mode": "chat",
                "X-Rubick-Hijacked": "false",
            },
        )


@app.post("/v1/decision")
async def categorical_decision(request: Request):
    """
    System 1 rapid categorical decision / routing endpoint.
    Computes calibrated softmax probability over option labels in 1 forward pass.
    """
    body = await request.json()
    state = body.get("state", "")
    question = body.get("question", "")
    options = body.get("options", {})  # e.g. {"A": "Billing", "B": "Technical", "C": "General"}

    if not options:
        raise HTTPException(status_code=400, detail="Missing 'options' dictionary")

    prompt = format_decision_prompt(state, question, options)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"enable_thinking": False},
    }

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        resp = await client.post(
            f"{settings.llama_server_url}/v1/chat/completions",
            json=payload,
        )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Backend error from llama-server: {resp.text}",
            )

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        top_lps = (
            choice.get("logprobs", {})
            .get("content", [{}])[0]
            .get("top_logprobs", [])
        )

        # Map logprobs to option keys
        option_logprobs = {}
        for item in top_lps:
            tok = item.get("token", "").strip().upper()
            if tok in options:
                option_logprobs[tok] = item.get("logprob", -100.0)

        # Softmax over options
        raw_scores = {k: math.exp(option_logprobs.get(k, -100.0)) for k in options}
        total = sum(raw_scores.values()) or 1.0
        probabilities = {k: v / total for k, v in raw_scores.items()}
        best_option = max(probabilities, key=probabilities.get)
        best_desc = options.get(best_option, "")
        confidence_pct = round(probabilities[best_option] * 100, 1)

        # On programmatic /v1/decision, default to pure typed response (JEV compliant) unless explicitly requested
        should_explain = body.get("explain", body.get("include_summary", False))
        response_payload = {
            "best_option": best_option,
            "best_description": best_desc,
            "probabilities": probabilities,
            "question": question,
            "options": options,
        }

        if should_explain:
            explain_prompt = f"State: {state}\nDecision: Option {best_option} ({best_desc} - {confidence_pct}% confidence)\n\nExplain this decision to the user in one concise sentence:"
            try:
                exp_resp = await client.post(
                    f"{settings.llama_server_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": explain_prompt}],
                        "max_tokens": 50,
                        "temperature": 0.3,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "extra_body": {"enable_thinking": False},
                    },
                )
                if exp_resp.status_code == 200:
                    exp_data = exp_resp.json()
                    exp_content = (
                        exp_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if exp_content:
                        response_payload["explanation"] = exp_content
                        response_payload["confidence_pct"] = confidence_pct
            except Exception:
                pass

        return response_payload



@app.post("/v1/decision/multi")
async def multi_decision_bitmask(request: Request):
    """
    Multi-statement Bitmask Decision endpoint.
    Answers up to 5 binary (True/False) questions simultaneously in ONE forward pass.
    Maps 2^N states to 2^N options, then marginalizes bit probabilities.
    """
    body = await request.json()
    state = body.get("state", "")
    questions = body.get("questions", [])

    if not questions or not isinstance(questions, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'questions' list")

    if len(questions) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 questions supported in one bitmask pass")

    n = len(questions)
    num_combinations = 1 << n  # 2^N

    # Generate labels: A, B, C... and their binary representations (e.g., 000, 001...)
    bitmask_options = {}
    for i in range(num_combinations):
        label = chr(ord("A") + i)
        binary_str = format(i, f"0{n}b")
        bitmask_options[label] = binary_str

    prompt = format_multi_decision_prompt(state, questions, bitmask_options)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": min(num_combinations + 10, 50),
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"enable_thinking": False},
    }

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        resp = await client.post(
            f"{settings.llama_server_url}/v1/chat/completions",
            json=payload,
        )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Backend error from llama-server: {resp.text}",
            )

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        top_lps = (
            choice.get("logprobs", {})
            .get("content", [{}])[0]
            .get("top_logprobs", [])
        )

        # Map logprobs to option keys
        option_logprobs = {}
        for item in top_lps:
            tok = item.get("token", "").strip().upper()
            if tok in bitmask_options:
                option_logprobs[tok] = item.get("logprob", -100.0)

        # Joint Softmax over all 2^N combinations
        raw_scores = {k: math.exp(option_logprobs.get(k, -100.0)) for k in bitmask_options}
        total = sum(raw_scores.values()) or 1.0
        joint_probs = {k: v / total for k, v in raw_scores.items()}

        # Marginalize each question's probability
        # P(question_i == True) = sum of P(label) where bit i of label is '1'
        marginal_results = {}
        for q_idx, q_text in enumerate(questions):
            p_true = sum(
                joint_probs[label]
                for label, bstr in bitmask_options.items()
                if bstr[q_idx] == "1"
            )
            marginal_results[q_text] = round(p_true, 4)

        return {
            "results": marginal_results,
            "questions": questions,
            "joint_probabilities": joint_probs,
        }
