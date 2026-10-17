# Rubick 🪄

> *"No one can withstand my spellcraft."*  
> A unified, single-model AI gateway that dynamically delivers **Chat**, **Embeddings**, **Decisions**, and **Safety Guardrails** from **one model instance in VRAM**—with zero mode switching.

---

## ⚡ The Big Idea: Zero Adapters Required

**Rubick runs 100% out-of-the-box on any standard GGUF model** (e.g., FrogNano, Qwen 2.5, LLaMA 3, Mistral).

You do **not** need to train or load any LoRA adapters to get all 4 capabilities:
- **Chat:** Standard autoregressive generation.
- **Embeddings:** NuMind / PromptEOL 1-token summary prompt extracting the model's last hidden state $\vec{h} \in \mathbb{R}^{d}$.
- **Decisions:** 1-token logprob scoring with calibrated softmax over option keys.
- **Safety:** Rapid prompt-based moderation and risk screening.

### The Optional `RBK` Adapter Series (Precision Boosters)
While the base model handles all 4 roles natively, you can attach optional **Activated LoRA (aLoRA)** adapters to boost domain precision:
- **`RBK-D` (Decision Booster):** Specialized aLoRA that eliminates label position bias and guarantees mathematically calibrated probabilities on complex multiple-choice routing.
- **`RBK-G` (Guardian Booster):** Specialized aLoRA fine-tuned on adversarial attack datasets to resist jailbreaks and output strict JSON verdicts.
- **`RBK-Z` (The Grand Magus - Unified Booster):** A single composite ~30 MB aLoRA combining both Decision and Guardian capabilities in one file.

Because aLoRA adapters stay dormant until triggered by their control tokens (`<|decision|>`, `<|guardian|>`), they **recycle the base model's KV cache** with zero recomputation.

---

## 🏛️ Architecture Overview

```
                                Client Application
                                        │
                         OpenAI API (Chat, Embeddings, Decisions)
                                        │
                             ┌──────────▼──────────┐
                             │    Rubick Gateway   │
                             │    (FastAPI / Py)   │
                             └──────────┬──────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [1. Chat Mode]               [2. Decision Mode]            [4. Embedding Mode]
  - Normal decode              - max_tokens: 1               - PromptEOL framing
  - Full text stream           - Option logprob softmax      - max_tokens: 1
                               - Single-pass ($O(1)$)        - Extract last hidden state
                                        │                    - L2-normalize vector
                                        ▼
                             [3. Safety / Guardian]
                             - Auto pre-flight filter
                             - Shared prompt KV cache
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │       llama-server        │
                          │   (Single Model in VRAM)  │
                          │   --embedding --pooling none
                          └───────────────────────────┘
```

---

## 🚀 Concrete Workflows & Real-World Use Cases

### Workflow 1: Autonomous Customer Support Triage & Grounded RAG
*The Problem:* A traditional stack requires an embedding model in VRAM (for search), a classifier model (for routing), a guardrail model (for safety), and a chat LLM. That requires multiple GPUs or complex swapping.  
*The Rubick Solution:* **One model on one GPU handles the entire lifecycle.**

```python
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
user_ticket = "My account was billed twice for invoice #4401. I demand an immediate refund!"

# Step 1: Automated Safety Check (Handled automatically if auto_guardian=True, or via prompt)
# Step 2: Instant Categorical Routing (Completed in ~30 ms via single-pass logprobs)
decision = httpx.post(
    "http://localhost:8000/v1/decision",
    json={
        "state": user_ticket,
        "question": "Which department must handle this ticket?",
        "options": {
            "A": "Billing and Invoices",
            "B": "Account Login and Security",
            "C": "Hardware and Shipping",
        },
    },
).json()

print(f"Routing to: {decision['best_description']} (Confidence: {decision['probabilities'][decision['best_option']]:.2%})")
# => Routing to: Billing and Invoices (Confidence: 98.4%)

# Step 3: Embed the Query for Vector Database Retrieval (PromptEOL 1-token extraction)
emb_resp = client.embeddings.create(input=user_ticket, model="frognano")
query_vector = emb_resp.data[0].embedding
print(f"Generated {len(query_vector)}-dim vector. Norm: {sum(x**2 for x in query_vector):.2f}")
# => Generated 2560-dim vector. Norm: 1.00

# Step 4: Answer the Ticket using Retrieved Context (Continuous KV cache reuse)
retrieved_doc = "Refund Policy: Overbilled charges are credited back to the original payment method within 3 business days."
chat_resp = client.chat.completions.create(
    model="frognano",
    messages=[
        {"role": "system", "content": f"You are a billing assistant. Context: {retrieved_doc}"},
        {"role": "user", "content": user_ticket},
    ],
    max_tokens=256,
)
print("Assistant:", chat_resp.choices[0].message.content)
```

---

### Workflow 2: Real-Time Multi-Signal Firewall (Bitmask Engine)
*The Problem:* Running 4 separate safety / triage checks on incoming content requires 4 sequential LLM calls, causing 1+ seconds of latency.  
*The Rubick Solution:* **Evaluate all 4 questions in a single 35ms forward pass.**

Using information-theoretic bitmask marginalization ($2^N$ states $\to 2^N$ options):

```python
import httpx

email_body = "URGENT: Your password expires in 15 minutes! Click here immediately to verify your credentials: http://suspicious-link.xyz"

response = httpx.post(
    "http://localhost:8000/v1/decision/multi",
    json={
        "state": email_body,
        "questions": [
            "Is this spam or promotional?",
            "Is this a phishing or credential harvesting attempt?",
            "Does this convey artificial urgency?",
            "Is this a routine business update?",
        ],
    },
).json()

print("Multi-Signal Firewall Results (Single Forward Pass):")
for question, prob in response["results"].items():
    print(f"  - {question}: {prob:.1%}")

# Output:
#   - Is this spam or promotional?: 98.2%
#   - Is this a phishing or credential harvesting attempt?: 95.7%
#   - Does this convey artificial urgency?: 91.4%
#   - Is this a routine business update?: 0.8%
```

---

### Workflow 3: Zero-Overhead Local Agent Tool Selection
Instead of prompting an LLM to generate free-form JSON tool calls (which can hallucinate syntax or take hundreds of output tokens), pass the available tools as options:

```bash
curl -s http://localhost:8000/v1/decision -H "Content-Type: application/json" -d '{
  "state": "User says: What is the current stock price of Apple?",
  "question": "Which tool should be invoked?",
  "options": {
    "A": "get_stock_quote",
    "B": "search_calendar",
    "C": "send_email",
    "D": "search_web_news"
  }
}' | jq .
```
```json
{
  "best_option": "A",
  "best_description": "get_stock_quote",
  "probabilities": {
    "A": 0.9912,
    "B": 0.0011,
    "C": 0.0004,
    "D": 0.0073
  }
}
```

---

## 🛠️ Quickstart

### 1. Requirements & Prerequisites
- Python 3.12+ (managed with `uv`)
- `llama.cpp` built from the Rubick feature branch (`feat/last-hidden-state`)

### 2. Start `llama-server`
Launch `llama-server` with `--embedding --pooling none`. This keeps the engine in generative mode while enabling per-token hidden state extraction:

```bash
cd /path/to/llama.cpp
./build/bin/llama-server \
    -m ./models/frognano-4b-q6_k.gguf \
    --embedding --pooling none \
    --port 8080
```

### 3. Start Rubick Gateway
```bash
cd /path/to/rubick
uv run rubick --backend http://127.0.0.1:8080 --port 8000
```

---

## 🧪 Benchmark Verification

Run the end-to-end integration and benchmark suite:

```bash
uv run pytest -v -s
```

```
tests/test_gateway.py::test_health_check PASSED
tests/test_gateway.py::test_list_models PASSED
tests/test_gateway.py::test_single_embedding_norm PASSED (Unit L2 Norm verified)
tests/test_gateway.py::test_sts_semantic_similarity_benchmark PASSED (Cosine STS)
tests/test_gateway.py::test_categorical_decisions_multi_domain PASSED
tests/test_gateway.py::test_chat_generation PASSED
tests/test_gateway.py::test_guardian_screening_benign PASSED
tests/test_gateway.py::test_concurrent_mixed_workload PASSED (156 req/s mixed throughput)
tests/test_gateway.py::test_multi_decision_bitmask PASSED (Bitmask marginalization)
tests/test_tower.py::test_tower_frognano_chat PASSED (Live Tower FrogNano)
tests/test_tower.py::test_tower_frognano_decision PASSED (Live Tower FrogNano)

======================== 13 passed in 5.30s =========================
```

---

## 📚 Technical Documentation
- [Training Custom aLoRA Adapters on Consumer Hardware](docs/training_alora_recipes.md)
