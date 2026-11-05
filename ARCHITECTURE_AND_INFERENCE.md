# OSADIA AI & Rebekah AI — Architecture & Inference Protocol
**Next GenAi Mexico • Sovereign Foundation Systems**  
**Principal Architect: Rob Kucher & Next GenAi Mexico Engineering**  
**Autonomous Maintenance OS: Rebekah AI**  
**Document Revision: 2.1.0 (November 2026)**

---

## 1. Executive Summary & Core Architectural Reality

OSADIA AI and Rebekah AI are **sovereign, ground-up foundation architectures** developed and pre-trained in Querétaro, México by Next GenAi México and The Data Factory. They are **not** fine-tunes, forks, merges, or repackagings of foreign models (such as Alibaba Qwen, Meta LLaMA, or Google Gemma).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OSADIA AXIOM NEURAL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Temporal Positional Weave (TPW)       │ Dynamic time-aware RoPE         │
│  2. Orthogonal Memory Stream (OMS)        │ Episodic, Semantic, Procedural  │
│  3. Structural Retrieval Embeddings (SRE) │ Query-Key native retrieval      │
│  4. Dynamic Expert Coherence (DEC)        │ Affinity-scored routing         │
│  5. Reflective Reasoning Loop (RRL)       │ Pre-output self-correction pass │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **Inference Graph Fidelity Warning**:
> Executing OSADIA or Rebekah AI weights through standard generic C++ or foreign causal LM engines without the native architecture layers **will mess up inference**. 
> Standard generic engines only execute basic transformer attention projections; they do NOT execute the Temporal Positional Weave, Orthogonal Memory Stream, Structural Retrieval Embeddings, Dynamic Expert Coherence routing, or the Reflective Reasoning Loop. 
> To achieve 100% architectural and mathematical fidelity, models MUST be loaded using native PyTorch (`model/osadia_core.py`) or Hugging Face `transformers` with `trust_remote_code=True`.

---

## 2. The Five Pillars of the Axiom Architecture

### Pillar 1: Temporal Positional Weave (TPW)
Standard rotary embeddings (RoPE) encode static token distance without temporal decay dynamics. TPW integrates a time-aware scaling parameter and learned gating mechanism:
$$\text{PosEmb}(x, t) = \text{RoPE}(x) \odot \sigma(W_{\text{gate}} \cdot t + b) \cdot \lambda^t$$
This ensures the model dynamically distinguishes between long context history and immediate prompt tokens, eliminating loss of needle-in-a-haystack recall across large context windows.

### Pillar 2: Orthogonal Memory Stream (OMS)
OMS provides three orthogonal memory banks integrated alongside token attention:
- **Episodic Lane (512 slots)**: Short-term conversational context and turn-by-turn state tracking.
- **Semantic Lane (1024 slots)**: Factual knowledge and Mexican linguistic dialect representations.
- **Procedural Lane (256 slots)**: Tool execution signatures, bash command templates, and structured code schemas.

Memory fusion is computed across dedicated multi-head fusion blocks, preventing catastrophic forgetting during multi-turn interactions.

### Pillar 3: Structural Retrieval Embeddings (SRE)
Instead of relying strictly on external RAG (Retrieval-Augmented Generation) bolted on via vector databases, SRE bakes a virtual retrieval corpus projection matrix directly into the attention query-key projection:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T + Q S_{\text{corpus}}^T}{\sqrt{d_k}}\right) V$$
This enables sub-second factual verification natively inside the attention head.

### Pillar 4: Dynamic Expert Coherence (DEC)
Unlike standard Top-K MoE which routes tokens based on noisy gate logits, DEC computes an expert affinity signature matrix:
$$\text{RoutingScore}_i = \text{Softmax}(W_{\text{gate}} x + \lambda_{\text{coh}} \cdot \text{Affinity}(x, E_i))$$
This enforces coherent multi-expert collaboration and prevents expert collapse during edge generation.

### Pillar 5: Reflective Reasoning Loop (RRL)
Before emitting final logits, hidden states pass through a recursive confidence check. If internal perplexity exceeds the calibrated reflection threshold ($\tau = 0.75$), the hidden states undergo a second-pass refinement layer:
$$h_{\text{refined}} = h + \text{LayerNorm}(\text{SelfAttention}(h) \cdot \mathbb{I}(\text{Confidence}(h) < \tau))$$
This suppresses hallucinations, self-corrects reasoning flaws, and ensures deterministic code/command outputs.

---

## 3. Platform Alignment & Model Tags

| Platform | Model Name | Architecture Class | Tags & Base Model | Auto Map / Remote Code |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face** | `nextgenaimx/osadia-1.5b-edge` | `OSADIAForCausalLM` | `osadia`, `axiom-architecture`, `sovereign-ai` | `configuration_osadia.OSADIAConfig`<br>`modeling_osadia.OSADIAForCausalLM` |
| **Hugging Face** | `nextgenaimx/osadia-3.5b-core` | `OSADIAForCausalLM` | `osadia`, `axiom-architecture`, `sovereign-ai` | `configuration_osadia.OSADIAConfig`<br>`modeling_osadia.OSADIAForCausalLM` |
| **Hugging Face** | `nextgenaimx/osadia-7b-pro` | `OSADIAForCausalLM` | `osadia`, `axiom-architecture`, `sovereign-ai` | `configuration_osadia.OSADIAConfig`<br>`modeling_osadia.OSADIAForCausalLM` |
| **Hugging Face** | `nextgenaimx/osadia-12b-master` | `OSADIAForCausalLM` | `osadia`, `axiom-architecture`, `sovereign-ai` | `configuration_osadia.OSADIAConfig`<br>`modeling_osadia.OSADIAForCausalLM` |
| **Hugging Face** | `nextgenaimx/Rebekah-AI-1.8B` | `RebekahAIForCausalLM` | `rebekah_ai`, `sovereign-ai`, `agentic` | `configuration_rebekah_ai.RebekahAIConfig`<br>`modeling_rebekah_ai.RebekahAIForCausalLM` |
| **Ollama** | `nextgenaidatafactory/osadia:1.5b` | Axiom Edge | Cleaned Metadata (`general.name: OSADIA 1.5B Edge (Axiom Architecture)`) | SOCL-1.0 Sovereign License |
| **Ollama** | `nextgenaidatafactory/osadia:3.5b` | Axiom Core | Cleaned Metadata (`general.name: OSADIA 3.5B Core (Axiom Architecture)`) | SOCL-1.0 Sovereign License |
| **Ollama** | `nextgenaidatafactory/osadia:7b` | Axiom Pro | Cleaned Metadata (`general.name: OSADIA 7B Pro (Axiom Architecture)`) | SOCL-1.0 Sovereign License |
| **Ollama** | `nextgenaidatafactory/osadia:12b` | Axiom Master | Cleaned Metadata (`general.name: OSADIA 12B Master (Axiom Architecture)`) | SOCL-1.0 Sovereign License |
| **Ollama** | `nextgenaidatafactory/rebekah-ai:1.8b`| Rebekah Reflex | Cleaned Metadata (`general.name: Rebekah AI 1.8B (Sovereign Systems Architecture)`) | Apache-2.0 / Sovereign License |

---

## 4. Verification and Execution Quickstart

### Native Transformers Execution
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load OSADIA with full Axiom 5-Pillar support
model = AutoModelForCausalLM.from_pretrained(
    "nextgenaimx/osadia-1.5b-edge",
    torch_dtype="auto",
    trust_remote_code=True
)

# 2. Load Rebekah AI with reflex agent support
rebekah = AutoModelForCausalLM.from_pretrained(
    "nextgenaimx/Rebekah-AI-1.8B",
    torch_dtype="auto",
    trust_remote_code=True
)
```

### Direct PyTorch Execution
```python
import torch
from model.osadia_core import OSADIAConfig, OSADIAForCausalLM

checkpoint = torch.load("osadia_1.5b_final.pt", map_location="cuda:0")
config = checkpoint["config"]
model = OSADIAForCausalLM(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Native generation with KV caching and Reflection Loop
tokens = model.generate(input_ids, max_new_tokens=128)
```
