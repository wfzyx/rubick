import httpx
import pytest
from fastapi.testclient import TestClient

from rubick.config import settings
from rubick.gateway import app

client = TestClient(app)

TOWER_URL = "http://192.168.0.9:8080"


def is_tower_available() -> bool:
    try:
        resp = httpx.get(f"{TOWER_URL}/health", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_tower_available(), reason="Tower host (192.168.0.9:8080) is not reachable")
def test_tower_frognano_chat():
    old_url = settings.llama_server_url
    settings.llama_server_url = TOWER_URL
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is the capital of France? Answer with one city name."}],
                "max_tokens": 512,  # accommodate reasoning budget for reasoning models
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") + msg.get("reasoning_content", "")
        assert "paris" in content.lower()
    finally:
        settings.llama_server_url = old_url


@pytest.mark.skipif(not is_tower_available(), reason="Tower host (192.168.0.9:8080) is not reachable")
def test_tower_frognano_decision():
    old_url = settings.llama_server_url
    settings.llama_server_url = TOWER_URL
    try:
        resp = client.post(
            "/v1/decision",
            json={
                "state": "The server room temperature exceeds 90 degrees and alarms are blaring.",
                "question": "What is the severity level?",
                "options": {
                    "A": "Critical Emergency",
                    "B": "Normal Operation",
                    "C": "Scheduled Inspection",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_option"] in ["A", "B", "C"]
        assert "probabilities" in data
    finally:
        settings.llama_server_url = old_url



@pytest.mark.skipif(not is_tower_available(), reason="Tower host (192.168.0.9:8080) is not reachable")
@pytest.mark.asyncio
async def test_tower_recursive_llm_decision_extraction():
    """Verify that FrogNano extracts structured questions and options via recursive LLM call."""
    from rubick.intent import extract_decision_via_llm

    async with httpx.AsyncClient() as async_client:
        # Decision prompt with bullets
        res = await extract_decision_via_llm(
            async_client, TOWER_URL, "decision: which of these should I eat\n\n- pizza\n- sushi\n- salad"
        )
        assert res is not None
        question, options = res
        assert "which" in question.lower()
        assert len(options) == 3
        assert "pizza" in options.values()
        assert "sushi" in options.values()
        assert "salad" in options.values()

        # Non-decision prompt
        non_res = await extract_decision_via_llm(
            async_client, TOWER_URL, "Here are the meeting notes:\n- deploy v1.2\n- review PR"
        )
        assert non_res is None



@pytest.mark.skipif(not is_tower_available(), reason="Tower host (192.168.0.9:8080) is not reachable")
def test_tower_end_to_end_decision_bulleted_list():
    """Verify that bulleted options with a decision cue are hijacked against real FrogNano."""
    old_url = settings.llama_server_url
    settings.llama_server_url = TOWER_URL
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "decision: which of these should I eat\n\n- pizza\n- sushi\n- salad",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-rubick-mode") == "decision"
        assert resp.headers.get("x-rubick-hijacked") == "true"
        data = resp.json()
        assert data.get("rubick_routed_mode") == "decision"
        assert "decision" in data
        assert len(data["decision"]["probabilities"]) == 3
        assert data["decision"]["best_option"] in ("A", "B", "C")
    finally:
        settings.llama_server_url = old_url


@pytest.mark.skipif(not is_tower_available(), reason="Tower host (192.168.0.9:8080) is not reachable")
def test_tower_end_to_end_decision_inline_alternatives():
    """Verify that inline alternatives ('between X or Y') are hijacked against real FrogNano."""
    old_url = settings.llama_server_url
    settings.llama_server_url = TOWER_URL
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What should I choose between pizza, sushi, or salad?",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-rubick-mode") == "decision"
        assert resp.headers.get("x-rubick-hijacked") == "true"
        data = resp.json()
        assert data.get("rubick_routed_mode") == "decision"
        assert len(data["decision"]["probabilities"]) == 3
    finally:
        settings.llama_server_url = old_url
