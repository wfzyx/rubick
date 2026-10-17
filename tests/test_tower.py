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
async def test_tower_intent_classification_decision_head():
    """Verify that FrogNano's System-1 Decision Head correctly classifies intent semantically."""
    from rubick.intent import classify_intent_with_decision_head

    async with httpx.AsyncClient() as async_client:
        # 1. Embedding intent
        mode_emb, probs_emb = await classify_intent_with_decision_head(
            async_client, TOWER_URL, "Could you generate a semantic vector representation for this documentation page?"
        )
        assert mode_emb == "embedding"
        assert probs_emb["EMBED"] > 0.70

        # 2. Decision intent
        mode_dec, probs_dec = await classify_intent_with_decision_head(
            async_client, TOWER_URL, "Which license should I choose for an open source tool? MIT, Apache, or GPL?"
        )
        assert mode_dec == "decision"
        assert probs_dec["DECISION"] > 0.70

        # 3. Conversational chat intent
        mode_chat, probs_chat = await classify_intent_with_decision_head(
            async_client, TOWER_URL, "Write a Python script to calculate Fibonacci numbers with memoization."
        )
        assert mode_chat == "chat"
        assert probs_chat["CHAT"] > 0.70
