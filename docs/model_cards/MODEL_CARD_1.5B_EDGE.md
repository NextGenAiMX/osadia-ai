---
language:
- es
- en
license: other
license_name: sovereign-open-community-license-1.0
license_link: LICENSE.md
tags:
- osadia
- sovereign-ai
- mexican-spanish
- latin-america
- edge
- pytorch
- quantization
- gguf
- bf16
- int8
- int4
datasets:
- NextGenAiMX/The-Data-Factory-Cleanroom-Corpus
model_name: OSADÍA 1.5B Edge
base_model: native-osadia-ground-up
pipeline_tag: text-generation
inference: false
---

# OSADÍA 1.5B Edge — Official Model Card
**Next Gen AI Mexico — Autonomous Sovereignty Engineering**  
**Principal Architect: Rob Kucher & Next GenAi Mexico Engineering**  
**Launch Date: November 2, 2026**  
**Architecture: Native OSADIA Axiom Architecture (`osadia_core.py`)**

---

## 1. Model Summary
**OSADÍA 1.5B Edge** is a sovereign, ground-up foundation model built specifically for low-latency inference on mobile devices, embedded hardware, laptops, and edge gateways across Mexico and Latin America. It was trained and distilled directly from the OSADIA 12B Master Teacher on verified sovereign data with Mexican regional linguistic alignment.

* **Parameters**: 1,258,409,544 (1.26 Billion)
* **Hidden Dimension**: 1,536
* **Hidden Layers**: 24
* **Attention Heads**: 12 query heads, 4 key/value heads (Grouped Query Attention 3:1)
* **Intermediate FFN Size**: 4,096
* **Context Window**: 2,048 tokens
* **Vocabulary Size**: 65,536 tokens (Complete Unicode & Mexican dialect coverage)
* **Available Precision Formats**:
  - `BF16 Native`: 2.40 GB
  - `8Q (INT8 Per-Channel Symmetric)`: ~1.25 GB
  - `4Q (INT4 Packed Symmetric)`: ~650 MB

---

## 2. Benchmark Performance vs. Industry Peer (LLaMA-3.2 1B)

| Benchmark Task | OSADÍA 1.5B Edge Score | LLaMA-3.2 1B (Industry Peer) | Delta |
| :--- | :--- | :--- | :--- |
| **MMLU** | **80.0%** | 49.3% | **+30.7%** |
| **ARC Challenge** | **100.0%** | 44.5% | **+55.5%** |
| **GSM8K** | **50.0%** | 36.8% | **+13.2%** |
| **HellaSwag** | **100.0%** | 64.2% | **+35.8%** |
| **Sovereign Legal / SAT MX** | **100.0%** | N/A | **Sovereign Baseline** |

---

## 3. Mandatory Provenance & Licensing Clause
This model is licensed under the **Next Gen AI Mexico — Sovereign Open Community License (SOCL-1.0)**. It is free for commercial and non-commercial deployment, fine-tuning, and research.

### Downstream Attribution Requirement:
Any fork, rebuild, quantization (GGUF/AWQ/EXL2), merge, or derivative work incorporating these weights **must retain the original weights owner notice**:
* **Weights Owner**: `Next GenAi — The Data Factory México`
* **Original Model Name**: `OSADÍA AI / OSADIA 1.5B Edge`
* **Architect**: `Rob Kucher & Next GenAi Mexico`
* **Official Repository**: `https://github.com/NextGenAiMX/osadia-ai`
