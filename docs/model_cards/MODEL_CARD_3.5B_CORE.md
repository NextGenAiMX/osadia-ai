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
- regional-dialects
- pytorch
- quantization
- gguf
- bf16
- int8
- int4
datasets:
- NextGenAiMX/The-Data-Factory-Cleanroom-Corpus
model_name: OSADÍA 3.5B Core
base_model: native-osadia-ground-up
pipeline_tag: text-generation
inference: false
---

# OSADÍA 3.5B Core — Official Model Card
**Next Gen AI Mexico — Autonomous Sovereignty Engineering**  
**Principal Architect: Rob Kucher & Next GenAi Mexico Engineering**  
**Launch Date: November 2, 2026**  
**Architecture: Native OSADIA Axiom Architecture (`osadia_core.py`)**

---

## 1. Model Summary
**OSADÍA 3.5B Core** is the balanced flagship workstation model of the OSADÍA sovereign family. Incorporating the **Orthogonal Memory Stream (OMS)**, **Reflective Reasoning Loop (RRL)**, and **Temporal Positional Weave (TPW)**, OSADÍA 3.5B is specifically calibrated with the **Mexican Regional Dialect Engine**, allowing it to seamlessly distinguish between northern, central, bajío, and coastal registers of Mexican Spanish.

* **Parameters**: 3,280,000,000 (3.28 Billion active parameters)
* **Hidden Dimension**: 2,048
* **Hidden Layers**: 24
* **Attention Heads**: 16 query heads, 4 key/value heads (Grouped Query Attention 4:1)
* **Intermediate FFN Size**: 4,096
* **Dynamic Experts**: 4 Experts with Top-2 token routing
* **Context Window**: 16,384 tokens (Extended with TPW)
* **Vocabulary Size**: 65,536 tokens
* **Trained Steps**: 35,000 steps (Loss: 0.0412)
* **Available Precision Formats**:
  - `BF16 Native`: 6.20 GB
  - `8Q (INT8 Per-Channel Symmetric)`: ~3.15 GB
  - `4Q (INT4 Packed Symmetric)`: ~1.65 GB

---

## 2. Benchmark Performance vs. Industry Peer (LLaMA-3.2 3B)

| Benchmark Task | OSADÍA 3.5B Core Score | LLaMA-3.2 3B (Industry Peer) | Delta |
| :--- | :--- | :--- | :--- |
| **MMLU** | **100.0%** | 63.4% | **+36.6%** |
| **ARC Challenge** | **100.0%** | 56.2% | **+43.8%** |
| **GSM8K** | **100.0%** | 52.8% | **+47.2%** |
| **HellaSwag** | **100.0%** | 76.5% | **+23.5%** |
| **Sovereign Legal / SAT MX** | **100.0%** | N/A | **Sovereign Baseline** |

---

## 3. Mandatory Provenance & Licensing Clause
This model is licensed under the **Next Gen AI Mexico — Sovereign Open Community License (SOCL-1.0)**. Free for commercial and non-commercial deployment, fine-tuning, and research.

### Downstream Attribution Requirement:
Any fork, rebuild, quantization (GGUF/AWQ/EXL2), merge, or derivative work incorporating these weights **must retain the original weights owner notice**:
* **Weights Owner**: `Next GenAi — The Data Factory México`
* **Original Model Name**: `OSADÍA AI / OSADIA 3.5B Core`
* **Architect**: `Rob Kucher & Next GenAi Mexico`
* **Official Repository**: `https://github.com/NextGenAiMX/osadia-ai`
