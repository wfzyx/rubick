import json
import math
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from rubick.config import settings
from rubick.normalizer import l2_normalize
from rubick.prompts import format_decision_prompt, format_embedding_prompt, format_guardian_prompt

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

    # Auto-Guardian pre-flight screening if requested or globally active
    perform_guardian = body.get("auto_guardian", settings.auto_guardian)
    messages = body.get("messages", [])

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

        return {
            "best_option": best_option,
            "best_description": options.get(best_option),
            "probabilities": probabilities,
            "question": question,
            "options": options,
        }
