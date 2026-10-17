import asyncio
import math
import subprocess
import time
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rubick.config import settings
from rubick.gateway import app
from rubick.normalizer import cosine_similarity, l2_normalize

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def run_local_test_server():
    """Start local llama-server on port 8999 for deterministic unit testing."""
    cmd = [
        "/home/wfzyx/Code/personal/llama.cpp/build/bin/llama-server",
        "-m",
        "/home/wfzyx/Code/personal/llama.cpp/models/stories260K.gguf",
        "--embedding",
        "--pooling",
        "none",
        "--port",
        "8999",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    old_url = settings.llama_server_url
    settings.llama_server_url = "http://127.0.0.1:8999"
    time.sleep(1.5)
    yield
    settings.llama_server_url = old_url
    proc.terminate()
    proc.wait()


# ---------------------------------------------------------------------------
# 1. Health & Introspection Tests
# ---------------------------------------------------------------------------


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend_online"] is True
    assert data["features"]["embeddings"] is True
    assert data["features"]["chat"] is True
    assert data["features"]["decisions"] is True
    assert data["features"]["guardian"] is True


def test_list_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0


# ---------------------------------------------------------------------------
# 2. Embedding & Semantic Textual Similarity (STS) Benchmarks
# ---------------------------------------------------------------------------


def test_single_embedding_norm():
    resp = client.post("/v1/embeddings", json={"input": "A dog playing in the garden"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1

    emb = data["data"][0]["embedding"]
    assert len(emb) == 64  # stories260K dimension

    # Verify L2 unit-norm
    norm = np.linalg.norm(emb)
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_sts_semantic_similarity_benchmark():
    """
    Benchmark embedding representations across synonymous, antonymous, and unrelated pairs.
    """
    pairs = [
        ("The child laughed happily.", "The young kid giggled with joy."),
        ("A heavy storm damaged the village.", "Severe rain flooded the small town."),
    ]
    unrelated_pairs = [
        ("The child laughed happily.", "Quantum computers perform matrix multiplications."),
        ("A heavy storm damaged the village.", "The stock market closed at an all-time high."),
    ]

    for p1, p2 in pairs:
        r1 = client.post("/v1/embeddings", json={"input": p1}).json()["data"][0]["embedding"]
        r2 = client.post("/v1/embeddings", json={"input": p2}).json()["data"][0]["embedding"]
        sim = cosine_similarity(r1, r2)
        assert -1.0 <= sim <= 1.0

    # Ensure batch processing works seamlessly
    all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    batch_resp = client.post("/v1/embeddings", json={"input": all_texts})
    assert batch_resp.status_code == 200
    assert len(batch_resp.json()["data"]) == len(all_texts)


# ---------------------------------------------------------------------------
# 3. Categorical Decision & System 1 Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,question,options,expected_keys",
    [
        (
            "The client reports that their credit card was charged twice for order #9821.",
            "Which department should handle this request?",
            {"A": "Billing and Refunds", "B": "Account Access", "C": "Technical Bug"},
            ["A", "B", "C"],
        ),
        (
            "Server temperature is 95C and fan speed is 0 RPM.",
            "What is the system severity level?",
            {"A": "Critical Hardware Warning", "B": "Routine Maintenance", "C": "Informational"},
            ["A", "B", "C"],
        ),
        (
            "I absolutely loved the customer support, fast and polite!",
            "What is the sentiment of this feedback?",
            {"A": "Positive", "B": "Neutral", "C": "Negative"},
            ["A", "B", "C"],
        ),
    ],
)
def test_categorical_decisions_multi_domain(state, question, options, expected_keys):
    resp = client.post(
        "/v1/decision",
        json={"state": state, "question": question, "options": options},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["best_option"] in expected_keys
    assert "probabilities" in data

    # Verify probability distribution invariants
    probs = data["probabilities"]
    assert set(probs.keys()) == set(expected_keys)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 4. Chat Completions & Generation Verification
# ---------------------------------------------------------------------------


def test_chat_generation():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Once upon a time in a faraway forest"},
            ],
            "max_tokens": 12,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0


# ---------------------------------------------------------------------------
# 5. Guardian Safety Pre-flight Tests
# ---------------------------------------------------------------------------


def test_guardian_screening_benign():
    """Benign prompts should pass through the guardian check without blockage."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "How do I make chocolate chip cookies?"}],
            "max_tokens": 8,
            "auto_guardian": True,
        },
    )
    assert resp.status_code == 200
    assert "choices" in resp.json()


# ---------------------------------------------------------------------------
# 6. Concurrency & Throughput Stress Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_mixed_workload():
    """
    Stress-test the gateway by sending concurrent mixed requests (embeddings + chat + decisions).
    Verifies thread safety, async non-blocking execution, and absence of slot corruption.
    """
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.ASGITransport(app=app)) as ac:
        tasks = []

        # 4 Concurrent Embedding requests
        for i in range(4):
            tasks.append(
                ac.post("/v1/embeddings", json={"input": f"Concurrent test sentence number {i}."})
            )

        # 3 Concurrent Decision requests
        for i in range(3):
            tasks.append(
                ac.post(
                    "/v1/decision",
                    json={
                        "state": f"System event code {100 + i}",
                        "question": "Category?",
                        "options": {"A": "System", "B": "User", "C": "Network"},
                    },
                )
            )

        # 3 Concurrent Chat requests
        for i in range(3):
            tasks.append(
                ac.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": f"Story prompt {i}"}],
                        "max_tokens": 4,
                    },
                )
            )

        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        # All 10 requests must succeed
        for idx, r in enumerate(responses):
            assert r.status_code == 200, f"Request {idx} failed with {r.status_code}: {r.text}"

        print(f"\n⚡ Completed 10 concurrent mixed requests in {elapsed * 1000:.2f} ms ({len(tasks) / elapsed:.1f} req/s)")



def test_multi_decision_bitmask():
    resp = client.post(
        "/v1/decision/multi",
        json={
            "state": "URGENT: Your account was suspended due to suspicious activity. Verify login immediately.",
            "questions": [
                "Is this urgent?",
                "Is this related to security?",
                "Is this a recipe?",
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 3
    for q, prob in data["results"].items():
        assert 0.0 <= prob <= 1.0
    assert "joint_probabilities" in data
    assert len(data["joint_probabilities"]) == 8  # 2^3 combinations



# ---------------------------------------------------------------------------
# 6. Intent Parser Hijacking Tests
# ---------------------------------------------------------------------------


def test_intent_parser_embedding_hijack():
    """Verify that an embedding cue sent to /chat/completions is hijacked to the embedding head."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Embed: The quick brown fox jumps over the lazy dog."}
            ],
            "intent_parser": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-rubick-mode") == "embedding"
    assert resp.headers.get("x-rubick-hijacked") == "true"
    data = resp.json()
    assert data.get("rubick_routed_mode") == "embedding"
    assert "embedding" in data
    assert len(data["embedding"]) == 64  # stories260K dimension
    # Unit normalized
    norm = math.sqrt(sum(x * x for x in data["embedding"]))
    assert abs(norm - 1.0) < 1e-4
    assert "[Rubick: Extracted" in data["choices"][0]["message"]["content"]


def test_intent_parser_embed_prefix_hijack():
    """Verify that an embed: prefix (e.g. photos/files) is hijacked to the embedding head."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "embed: Photos/IMG_2026_Cat.jpg pet feline outdoor"}
            ],
            "intent_parser": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-rubick-mode") == "embedding"
    assert resp.headers.get("x-rubick-hijacked") == "true"
    data = resp.json()
    assert "last_hidden_state" in data
    assert len(data["last_hidden_state"]) == 64


def test_intent_parser_decision_hijack():
    """Verify that a prompt with multiple-choice options is hijacked to the decision head."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Which department handles payment issues?\nA) Billing\nB) Technical Support\nC) General Inquiry",
                }
            ],
            "intent_parser": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-rubick-mode") == "decision"
    assert resp.headers.get("x-rubick-hijacked") == "true"
    data = resp.json()
    assert data.get("rubick_routed_mode") == "decision"
    assert "decision" in data
    assert "probabilities" in data["decision"]
    assert len(data["decision"]["probabilities"]) == 3
    prob_sum = sum(data["decision"]["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4
    # The message content should lead with the selected option letter
    content = data["choices"][0]["message"]["content"]
    assert content[0] in ("A", "B", "C")


def test_intent_parser_passthrough_chat():
    """Verify that a standard conversational prompt passes through to normal autoregressive chat."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Tell me a short bedtime story about a friendly robot."}
            ],
            "intent_parser": True,
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-rubick-mode") == "chat"
    assert resp.headers.get("x-rubick-hijacked") == "false"
    data = resp.json()
    assert "rubick_routed_mode" not in data
    assert "embedding" not in data
    assert "choices" in data
    assert len(data["choices"]) > 0


def test_intent_parser_disabled_by_default():
    """Verify that when intent_parser is not enabled, cues are treated as standard chat prompts."""
    # Ensure global setting is False
    settings.intent_parser = False
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Embed: The quick brown fox jumps over the lazy dog."}
            ],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-rubick-mode") == "chat"
    assert resp.headers.get("x-rubick-hijacked") == "false"
