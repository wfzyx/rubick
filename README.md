# Rubick 🪄

> *"No one can withstand my spellcraft."*  
> A unified, single-model AI gateway that dynamically switches between **Chat**, **Decisions**, **Safety Guardrails**, and **Embeddings** from one model instance in VRAM.

---

## 1. Overview & Vision

Running modern agentic workflows typically demands multiple models:
1. A **Chat / Assistant LLM** for natural language responses.
2. A **Decision / Routing model** (System 1 fast classification, routing tickets, picking tools).
3. A **Safety Guardrail model** (moderating inputs/outputs, detecting jailbreaks and toxicity).
4. An **Embedding model** for vector search, RAG retrieval, and document clustering.

Traditionally, serving all four requires multiple dedicated servers or GPUs, or running swapping managers like `llama-swap` that incur severe VRAM overhead or high latency. In `llama-server` (`llama.cpp`), serving chat and embeddings simultaneously from a single model instance is fundamentally mutually exclusive: launching with `--embedding` locks the engine into pooling mode, breaking `/v1/chat/completions`.

**Rubick** solves this by unifying all four use cases under a **single autoregressive generative forward pass** on a single model instance in memory, backed by prefix-cache reuse and lightweight routing.

```
                              Client Request
                                    │
                     POST /v1/chat/completions or /v1/embeddings
                                    │
                         ┌──────────▼──────────┐
                         │   Rubick Gateway    │
                         │   (Proxy Router)    │
                         └──────────┬──────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
[1. Chat Mode]             [2. Decision Mode]         [4. Embedding Mode]
- Standard decode           - max_tokens: 1            - PromptEOL framing
- Unconstrained text        - Option label logprobs    - max_tokens: 1
                            - Single-pass softmax      - Last hidden state vector
                                    │
                                    ▼
                         [3. Safety / Guardian]
                         - Auto pre-flight or explicit
                         - 16 tokens JSON verdict
                         - Shared prompt KV-cache
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │       llama-server        │
                      │  (Single Model in VRAM)   │
                      │  - Shared Prefix Cache    │
                      │  - Exposed Last-State     │
                      └───────────────────────────┘
```

---

## 2. The 4-in-1 Unified Generative Paradigm

Instead of maintaining different model architectures or server modes, Rubick frames every capability as a generative prompt evaluation:

| Mode | Trigger / Intent | Prompt Construction | Decode Budget | Extracted Output |
|---|---|---|---|---|
| **Chat** | Standard conversational prompt | Natural chat template | $N$ tokens | Generated text stream |
| **Decision** | Categorical routing, tool selection, `/v1/systemone` | `State: ... ### DECISION \n Question: ... \n Options: A, B, C` | `max_tokens: 1` | Softmax distribution over option label logits ($O(1)$) |
| **Safety** | Guardrail pre-flight or explicit moderation | `<guardian> Criteria: ... \n Text: ...` | `max_tokens: 16` | Constrained JSON verdict (`{"score": "yes"}`) |
| **Embedding** | `/v1/embeddings` or similarity query | `Text: ... \n Summary in one word:` (PromptEOL / E5-V) | `max_tokens: 1` | **Last hidden state vector** $\vec{h} \in \mathbb{R}^{d_{\text{model}}}$ |

### The NuMind Breakthrough for Embeddings
By adopting the PromptEOL / E5-V formulation (validated by NuMind's `vLLM-Last-Hidden-State`):
- Appending a single-word summary prompt causes the causal attention layers to concentrate semantic information about the preceding context into the final prompt position ($T-1$).
- The final decoder hidden state $\vec{h}_{T-1}$ (before LM head projection) serves as a dense semantic embedding.
- **Zero mode switching:** Embedding generation uses the exact same generative graph as chat, enabling `/v1/embeddings` and `/v1/chat/completions` to run concurrently against one model in `llama-server`.
- **Zero compute duplication:** If a document is embedded for retrieval, its tokens are already in `llama-server`'s prefix KV-cache, making subsequent conversation about that document instantaneous.

---

## 3. Architecture Components

### A. The Rubick Gateway (Headroom Proxy)
A lightweight HTTP reverse proxy exposing standard OpenAI endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`):
1. **Route Resolution:** Routes `/v1/embeddings` calls to the PromptEOL 1-token pipeline.
2. **Auto-Guardian Pre-flight:** Optionally screens incoming chat prompts with a rapid 1-turn safety check before releasing generation.
3. **Decision Logprob Aggregator:** Detects structured options and reads option logprobs to compute calibrated probabilities.
4. **Vector Formatter:** Normalizes (L2) the extracted hidden state vector and outputs standard OpenAI embedding JSON.

### B. `llama-server` Engine Extension
A minimal, surgical extension to `llama.cpp`'s `llama-server`:
- Adds an optional request parameter to `/v1/chat/completions`: `"return_last_hidden_state": true`.
- When set (with `max_tokens: 1`), `llama-server` retrieves the final hidden state of the prompt via `llama_get_embeddings_ith(ctx, -1)` and returns it in `choices[0].hidden_state`.
- Operates strictly in standard generative mode without forcing `--embedding`.

---

## 4. Next Steps & Roadmap

- [ ] **Phase 1: Study & Fork**
  - Read NuMind's vLLM PR (#57185) and `numindai/vLLM-Last-Hidden-State` implementation contract.
  - Review `llama.cpp` contribution guidelines, C API (`llama.h`), and `tools/server/` architecture.
  - Fork `llama.cpp` locally.
- [ ] **Phase 2: Implement Hidden State Exposure in `llama-server`**
  - Add `return_last_hidden_state` handling in `tools/server/server.cpp`.
  - Validate against a test GGUF (e.g. Qwen 2.5 or LLaMA 3) comparing extracted vectors with raw PyTorch outputs.
- [ ] **Phase 3: Build the Rubick Gateway**
  - Implement FastAPI / async HTTP proxy with OpenAI-compatible endpoints.
  - Implement PromptEOL embedding framing and vector L2 normalization.
  - Implement Decision option logprob extraction.
  - Implement Guardian pre-flight screening.
- [ ] **Phase 4: Benchmarking & Integration**
  - Evaluate MTEB retrieval performance of PromptEOL embeddings on target base models.
  - Measure end-to-end latency and KV-cache reuse efficiency across mixed workloads.
