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

### Workflow 4: Coding Agent Harness (pi / OpenCode) — Zero-Token File & Code Triage
*The Problem:* When an agent in a coding harness (like `pi` or OpenCode) is asked to organize a directory of 50 images, or triage 20 test failures, it conventionally issues 50 separate conversational chat calls. Each call generates ~150–256 tokens of verbose explanation or reasoning, taking **~45–60 seconds** and burning thousands of tokens just to decide where files belong.

*The Rubick Solution:* **Inline Head Switching (Embedding $\to$ Decision $\to$ Chat).**
1. The agent feeds each item with an embedding or decision cue (`enable_thinking: False`, `max_tokens: 1`).
2. The model evaluates the item in **one single forward pass (~168 ms, 1 token)** without generating conversational text.
3. The harness groups the vectors or decision labels in Python memory.
4. The model switches back to the standard generative chat head to deliver a clean human summary.

#### 📊 Measured Real-World Benchmark on FrogNano 4B (RTX 2080 Super):

| Metric | Conventional Chat Approach | Rubick System-1 Pass | Improvement |
|---|---|---|---|
| **Wall Clock (1 item)** | 4,624.8 ms | **149.0 ms** | **31.0x faster** |
| **Wall Clock (10 items)** | ~46.2 seconds | **1.68 seconds** | **27.5x faster** |
| **Output Tokens Generated** | 256 tokens / item | **1 token / item** | **99.6% token reduction** |
| **Classification Accuracy** | 100% | **100% (10/10)** | Identical accuracy |

*Benchmark performed live against FrogNano 4B (`Q6_K`) on 10 real stacktrace diagnostic errors.*

#### Python Coding Agent Example:
```python
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

files_to_triage = [
    "Traceback: No module named pydantic",
    "IndentationError: unexpected indent in line 44",
    "AssertionError: expected 200 got 500",
    "httpx.ConnectTimeout: connection timed out",
]

options = {"A": "Missing Dependency", "B": "Syntax Error", "C": "Logic Failure", "D": "Timeout"}

# 1. High-speed triage loop (~160 ms per item, 1 token generated per item)
triage_results = {}
for item in files_to_triage:
    r = httpx.post("http://localhost:8000/v1/decision", json={
        "state": item,
        "question": "Category?",
        "options": options,
    }).json()
    category = r["best_description"]
    triage_results.setdefault(category, []).append(item)

# 2. Return to normal generative chat head to report results to the human
report_prompt = f"Summarize the triage results concisely for the developer:\n{triage_results}"
summary = client.chat.completions.create(
    model="frognano",
    messages=[{"role": "user", "content": report_prompt}],
    max_tokens=150,
).choices[0].message.content

print(summary)
# "Found 4 errors across 4 distinct categories: 1 missing dependency (pydantic),
#  1 indentation error, 1 assertion failure, and 1 network timeout."
```

---
## 🛠️ Quickstart

### 1. Requirements & Prerequisites
- Python 3.12+ (managed with `uv`)
- `llama.cpp` built from the Rubick feature branch:
  ```bash
  git clone -b feat/last-hidden-state https://github.com/wfzyx/llama.cpp.git
  cd llama.cpp
  cmake -B build -G Ninja -DLLAMA_BUILD_SERVER=ON
  ninja -C build llama-server
  ```

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

### 4. Intelligent Intent Auto-Routing & Natural Explanations (Enabled by Default)
Rubick runs with `--intent-parser` and `--explain` enabled out of the box:
- **Multiple-choice / triage prompts:** Evaluates the mathematical decision in a **1-token forward pass (~149 ms)**, then attaches a concise natural language explanation from the chat head using the warmed prompt KV cache.
- **Embedding requests (`"Embed: ..."`, `"embed: ..."`, `"vectorize: ..."`):** Automatically hijacked to 1-token PromptEOL Last-Hidden-State Embedding extraction.
- **Conversational questions:** Transparently passed through to full autoregressive Chat.

```bash
# Enabled by default:
uv run rubick --backend http://127.0.0.1:8080 --port 8000

# To disable auto-routing or explanations:
uv run rubick --backend http://127.0.0.1:8080 --port 8000 --no-intent-parser --no-explain
```
*(Every response includes `X-Rubick-Mode`, `X-Rubick-Hijacked`, and `X-Rubick-Explained` headers confirming the path taken).*

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

## 📚 Technical Documentation & Prior Art
- [Training Custom aLoRA Adapters on Consumer Hardware](docs/training_alora_recipes.md)
- [NuMind vLLM PR #57185](https://github.com/vllm-project/vllm/pull/57185) & [vLLM-Last-Hidden-State](https://github.com/numindai/vLLM-Last-Hidden-State)
- [Activated LoRA (aLoRA): Fine-tuned LLMs for Intrinsics (NeurIPS 2025)](https://arxiv.org/abs/2504.12397)
- [Rubick llama.cpp Engine Fork (feat/last-hidden-state)](https://github.com/wfzyx/llama.cpp/tree/feat/last-hidden-state)
