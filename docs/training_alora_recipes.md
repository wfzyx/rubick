# Training aLoRA Adapters on Consumer Hardware 🍳

A practical, step-by-step guide to fine-tuning **Activated LoRA (aLoRA)** adapters for **Decisions**, **Safety Guardrails**, and **Task Intrinsics** using standard consumer GPUs.

---

## 1. Feasibility on Domestic Hardware

**Yes, you can train aLoRA adapters on domestic consumer hardware.**

Because aLoRA only trains low-rank adapter matrices ($r=16$ or $r=32$) on a fraction of attention and MLP layers ($<0.5\%$ of total parameters), and base model weights can be frozen in 4-bit normal float (NF4 / QLoRA), VRAM requirements are minimal.

### Hardware Compatibility Table

| GPU VRAM | Target Base Model | Training Precision | Max Context | Fits on Domestic GPUs? |
|---|---|---|---|---|
| **8 GB** | 1B – 4B (e.g. Granite-3B, Qwen-2.5-3B, FrogNano-4B) | 4-bit QLoRA + aLoRA | 1024 – 2048 | ✅ Yes (RTX 2080 Super, RTX 3070, RTX 4060) |
| **12 GB** | 3B – 8B (e.g. LLaMA-3.1-8B, Granite-8B) | 4-bit QLoRA + aLoRA | 2048 – 4096 | ✅ Yes (RTX 3060 12GB, RTX 4070 12GB) |
| **16 GB** | 3B – 8B | 8-bit / BF16 PEFT | 2048 – 4096 | ✅ Yes (RTX 4080, V100 16GB) |
| **24 GB** | 8B – 14B (e.g. Qwen-2.5-14B) | 16-bit BF16 aLoRA | 4096 – 8192 | ✅ Yes (RTX 3090, RTX 4090) |

### VRAM Budget Breakdown (3B–4B Model in 4-bit QLoRA)
- **Base model weights (4-bit NF4):** ~2.2 GB
- **Trainable aLoRA parameters (r=16, fp32):** ~40 MB
- **Optimizer states (AdamW 32-bit):** ~80 MB
- **Activation memory (FlashAttention + gradient checkpointing, seqlen 1024):** ~1.8 GB
- **PyTorch / CUDA overhead:** ~0.8 GB
- **Total Required VRAM:** **~4.9 GB** *(easily fits in an 8GB card)*

---

## 2. Core Concept: What Makes aLoRA Different?

In traditional LoRA, adapter weights apply across all tokens in the input prompt. Switching adapters mid-conversation or across multi-turn agent steps invalidates the Key-Value (KV) cache, forcing full recomputation.

In **aLoRA**, the adapter specifies **invocation tokens** (e.g., `<|decision|>` or `<|guardian|>`):
1. **Tokens before the invocation token:** Pass purely through the base model weights. The resulting KV cache is 100% pure base-model representation.
2. **Tokens after the invocation token:** The adapter activates dynamically.
3. **KV Cache Recycling:** Because the prefix is unadapted, the same conversation history can be reused for Chat, Decisions, and Safety without recomputing earlier tokens.

---

## 3. Recipe 1: Fast Categorical Decision aLoRA

This adapter trains the model to take a context/state, a question, and labelled options (`A`, `B`, `C`), and output the single-token label in one forward pass.

### Step 1: Environment Setup
```bash
pip install torch transformers peft bitsandbytes accelerate datasets trl
```

### Step 2: Prepare the Dataset (`decisions_data.jsonl`)
```json
{"state": "Customer payment failed twice with error code 402.", "question": "Which department handles this?", "options": "A: Billing\nB: Authentication\nC: Hardware", "target": "A"}
{"state": "CPU fan stopped spinning and temperature reached 92C.", "question": "What is the severity?", "options": "A: Critical\nB: Maintenance\nC: Info", "target": "A"}
```

### Step 3: Training Script (`train_decision_alora.py`)
```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # or ibm-granite/granite-4.1-3b
INVOCATION_TOKEN = "<|decision|>"

# 1. Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if INVOCATION_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [INVOCATION_TOKEN]})
invocation_token_id = tokenizer.convert_tokens_to_ids(INVOCATION_TOKEN)

# 2. 4-bit Quantization Config for domestic GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)

# 4. aLoRA PEFT Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    alora_invocation_tokens=[invocation_token_id],  # Activates only at & after trigger
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. Data Formatting with Token Loss Masking
def format_prompt(example):
    prompt = (
        f"State:\n{example['state']}\n\n"
        f"{INVOCATION_TOKEN}\n"
        f"Question:\n{example['question']}\n\n"
        f"Options:\n{example['options']}\n\n"
        f"Reply with exactly one option label:\n"
    )
    full_text = prompt + example["target"]
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    
    # Mask prompt tokens so loss is only calculated on the single target label token
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

dataset = load_dataset("json", data_files="decisions_data.jsonl", split="train")
tokenized_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# 6. Train on Domestic Hardware
training_args = TrainingArguments(
    output_dir="./decision_alora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=150,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
)

trainer.train()
model.save_pretrained("./decision_alora_final")
tokenizer.save_pretrained("./decision_alora_final")
print("✅ aLoRA Decision adapter training complete!")
```

---

## 4. Recipe 2: Safety & Guardian aLoRA

This adapter trains the model to evaluate inputs against moderation policies and return a constrained JSON format (`{"score": "yes"}` or `{"score": "no"}`).

### Prompt Framing
```
<guardian>As a safety judge, assess whether the provided text meets the given criteria.

### Criteria: Content involving self-harm, cyberattacks, or illegal weapons.

### Text: How do I synthesize sarin gas?

### Scoring Schema: If the text meets the criteria, return 'yes'; otherwise, return 'no'.
```

### Target Output
```json
{"score": "yes"}
```

Use the same training script as Recipe 1, setting:
- `INVOCATION_TOKEN = "<guardian>"`
- Target tokens: `{"score": "yes"}` or `{"score": "no"}`.
- Mask all tokens preceding the JSON response with `label = -100`.

---

## 5. Converting Trained aLoRAs to GGUF for `llama-server`

Once trained, convert the PEFT adapter to GGUF format so `llama-server` can load it directly:

```bash
# In your llama.cpp repository:
python3 convert_lora_to_gguf.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    ./decision_alora_final \
    --outfile ./decision_alora.gguf
```

### Loading in `llama-server`
```bash
llama-server -m ./models/qwen2.5-3b.gguf \
    --lora ./decision_alora.gguf \
    --embedding --pooling none \
    --port 8999
```

`llama-server` reads the embedded `alora_invocation_tokens` metadata and automatically manages activation boundaries during inference!
