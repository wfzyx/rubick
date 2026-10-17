import math
import subprocess
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rubick.gateway import app
from rubick.normalizer import cosine_similarity, l2_normalize

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def run_llama_server():
    """Start llama-server with stories260K in embedding+pooling-none mode for the test session."""
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
    time.sleep(1.5)  # allow server to boot
    yield
    proc.terminate()
    proc.wait()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend_online"] is True
    assert data["features"]["embeddings"] is True
    assert data["features"]["chat"] is True


def test_single_embedding():
    resp = client.post("/v1/embeddings", json={"input": "A dog playing in the garden"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1

    emb = data["data"][0]["embedding"]
    assert len(emb) == 64  # stories260K hidden size

    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_batch_embeddings():
    inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above the sleepy hound.",
        "Quantum mechanics describes microscopic particles.",
    ]
    resp = client.post("/v1/embeddings", json={"input": inputs})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 3

    v1 = data["data"][0]["embedding"]
    v2 = data["data"][1]["embedding"]
    v3 = data["data"][2]["embedding"]

    sim_synonyms = cosine_similarity(v1, v2)
    sim_unrelated = cosine_similarity(v1, v3)

    print(f"\nSimilarity (fox vs canine): {sim_synonyms:.4f}")
    print(f"Similarity (fox vs quantum): {sim_unrelated:.4f}")
    assert sim_synonyms != sim_unrelated


def test_chat_completions():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Once upon a time"}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0


def test_categorical_decision():
    resp = client.post(
        "/v1/decision",
        json={
            "state": "The user cannot log in and forgot their password.",
            "question": "Which department handles this?",
            "options": {
                "A": "Account and Authentication",
                "B": "Billing and Invoices",
                "C": "Hardware repair",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "best_option" in data
    assert data["best_option"] in ["A", "B", "C"]
    assert "probabilities" in data
    assert math.isclose(sum(data["probabilities"].values()), 1.0, rel_tol=1e-4)
