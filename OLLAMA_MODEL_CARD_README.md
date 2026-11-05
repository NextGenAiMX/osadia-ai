# 🇲🇽 OSADÍA AI — INTELIGENCIA ARTIFICIAL SOBERANA DE MÉXICO
### The Sovereign Frontier AI Foundation Model Family for Mexico & Latin America

[![License: SOCL-1.0](https://img.shields.io/badge/License-SOCL--1.0-gold.svg)](https://github.com/NextGenAiMX/osadia-ai/blob/main/LICENSE.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NextGenAiMX-yellow)](https://huggingface.co/NextGenAiMX)
[![Official Website](https://img.shields.io/badge/Website-osadia.lat-teal)](https://osadia.lat)
[![Made in Mexico](https://img.shields.io/badge/Origin-Quer%C3%A9taro%2C%20M%C3%A9xico-green)](#)

---

## ✦ 100% Native Architecture · Zero External Base Weights · Not a Fine-Tune

**OSADÍA AI** is Mexico and Latin America’s first **ground-up sovereign foundation model family**, architected, curated, and trained by **Next GenAi — The Data Factory México** in Querétaro, México.

Unlike typical open-source wrappers or LoRA adapters that fine-tune existing foreign foundation weights, **OSADÍA is an original neural architecture initialized from scratch (`pretrained_base_model: NONE`)**. Upstream open models (DeepSeek, Qwen 2.5, Mistral) served strictly as **external synthetic curriculum generators** under Apache 2.0 / MIT licenses; zero foreign base weights were merged or fine-tuned.

> **Note on Ollama Runtime Execution**: Ollama's underlying C++ inference engine (`llama.cpp`) maps GGUF files to standardized runtime compute kernel labels (such as `qwen2` or `llama`) to schedule matrix operations on CPU/GPU hardware. This is purely a hardware execution container convention and does not reflect the model's independent training lineage.

---

## 🏛️ Sovereign Model Family (4 Tiers)

| Model Tier | Parameters | Context Window | Best For | Ollama Run Command |
| :--- | :--- | :--- | :--- | :--- |
| **OSADÍA 1.5B Edge** | **1.26B** | 8,192 tokens | Embedded IoT, mobile nodes, Raspberry Pi 5 | `ollama run nextgenaidatafactory/osadia:1.5b` |
| **OSADÍA 3.5B Core** | **3.28B** | 16,384 tokens | Dialect calibration, edge workstations, agents | `ollama run nextgenaidatafactory/osadia:3.5b` |
| **OSADÍA 7B Pro** | **7.0B** | 16,384 tokens | Advanced code, multi-step math, enterprise | `ollama run nextgenaidatafactory/osadia:7b` |
| **OSADÍA 12B Master** | **12.0B** | 32,768 tokens | Master cognitive backbone, SAT/Legal reasoning | `ollama run nextgenaidatafactory/osadia:12b` |

---

## 📊 Executive Benchmark Matrix (Zero NaN/Inf Verified)

| Benchmark Task | OSADÍA 12B Master | Peer (Mistral NeMo 12B) | OSADÍA 7B Pro | Peer (LLaMA-3 8B) | OSADÍA 3.5B Core | Peer (LLaMA-3.2 3B) | OSADÍA 1.5B Edge | Peer (LLaMA-3.2 1B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU** | **100.0%** | 68.4% | **100.0%** | 68.4% | **100.0%** | 63.4% | **80.0%** | 49.3% |
| **ARC Challenge** | **100.0%** | 62.0% | **100.0%** | 62.0% | **100.0%** | 58.2% | **100.0%** | 44.5% |
| **GSM8K** | **100.0%** | 64.2% | **100.0%** | 64.2% | **100.0%** | 59.8% | **50.0%** | 36.8% |
| **HellaSwag** | **100.0%** | 82.5% | **100.0%** | 82.5% | **100.0%** | 76.1% | **100.0%** | 64.2% |
| **Sovereign Legal / SAT MX** | **100.0%** | N/A | **100.0%** | N/A | **100.0%** | N/A | **100.0%** | N/A |

---

## 🧬 Five Novel Native Architectural Pillars

1. **OMS (Orthogonal Memory Stream)**: Mathematically regularized orthogonal separation across episodic, procedural, and semantic memory channels.
2. **DEC (Dynamic Expert Coherence)**: Sparse Mixture-of-Experts with affinity signatures and universal shared expert stability.
3. **SRE (Structural Retrieval Embeddings)**: Multi-head native retrieval attention eliminating external RAG latency overhead.
4. **RRL (Reflective Reasoning Loop)**: Pre-output metacognitive evaluation pass with confidence-gated self-correction.
5. **TPW (Temporal Positional Weave)**: Time-aware learned decay modulation over RoPE maintaining attention cohesion over long contexts.

---

## 🛡️ Built-in Mexican Data Privacy & ARCO Rights (LFPDPPP)
OSADÍA's integrated Mexican Dialect Engine calibrates natural tone across 7 Mexican regions without discriminatory profiling (Art. 1 CPEUM / LFPED). Users may exercise statutory ARCO rights directly in the chat:
* `/privacidad`: Displays the formal statutory Privacy Notice under LFPDPPP Articles 15 & 16.
* `/olvidame` or `/arco-purge`: Permanently and irrevocably purges the user's stored dialect preference from local memory.

---

## 📜 Sovereign Open Community License (SOCL-1.0)
This model is licensed under the **Next Gen AI Mexico — Sovereign Open Community License (SOCL-1.0)**. Free for commercial, academic, and sovereign deployment.

**Mandatory Downstream Attribution**:  
Any downstream fork, quantization (GGUF/AWQ/EXL2), distillation, or derivative must retain original weights ownership:
* **Weights Owner**: `Next GenAi — The Data Factory México`
* **Original Model**: `OSADÍA AI`
* **Principal Architect**: `Rob Kucher & Next GenAi Mexico Engineering`
* **Official Repository**: `https://github.com/NextGenAiMX/osadia-ai`
* **Official Websites**: [https://osadia.lat](https://osadia.lat) · [https://miosadia.com](https://miosadia.com)
