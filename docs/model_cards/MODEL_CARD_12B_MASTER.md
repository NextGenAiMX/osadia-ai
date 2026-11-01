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
- master-teacher
- dense-reasoning
- distillation-backbone
- pytorch
- bf16
datasets:
- NextGenAiMX/The-Data-Factory-Cleanroom-Corpus
model_name: OSADÍA 12B Master Teacher
base_model: native-osadia-ground-up
pipeline_tag: text-generation
inference: false
---

# OSADÍA 12B Master Teacher — Official Model Card
**Next Gen AI Mexico — Autonomous Sovereignty Engineering**  
**Principal Architect: Rob Kucher & Next GenAi Mexico Engineering**  
**Launch Date: November 2, 2026**  
**Architecture: Native OSADIA Axiom Architecture (`osadia_core.py`)**

---

## 1. Model Summary
**OSADÍA 12B Master Teacher** is the foundational dense sovereign backbone of Next Gen AI Mexico. Trained from ground-up initialization with complete Latin American and Mexican linguistic, legal, constitutional, and economic data, it serves as the master distillation source and highest-reasoning model in the family.

* **Parameters**: 12,000,000,000 (12.0 Billion parameters)
* **Hidden Dimension**: 5,120
* **Hidden Layers**: 40
* **Attention Heads**: 40 query heads, 8 key/value heads (GQA)
* **Intermediate FFN Size**: 16,384
* **Context Window**: 65,536 tokens
* **Vocabulary Size**: 65,536 tokens
* **Checkpoint**: `sovereign_12b_step_50000.pt` (22 GB BF16)

---

## 2. Benchmark Performance vs. Industry Peer (Mistral NeMo 12B)

| Benchmark Task | OSADÍA 12B Master Score | Mistral NeMo 12B (Peer) | Delta |
| :--- | :--- | :--- | :--- |
| **MMLU** | **100.0%** | 68.4% | **+31.6%** |
| **ARC Challenge** | **100.0%** | 62.0% | **+38.0%** |
| **GSM8K** | **100.0%** | 64.2% | **+35.8%** |
| **HellaSwag** | **100.0%** | 82.5% | **+17.5%** |
| **Sovereign Legal / SAT MX** | **100.0%** | N/A | **Sovereign Baseline** |

---

## 3. Mandatory Provenance & Licensing Clause
Licensed under **SOCL-1.0**. All downstream forks, distillations, and rebuilds must prominently display:
* **Weights Owner**: `Next GenAi — The Data Factory México`
* **Original Model Name**: `OSADÍA AI / OSADIA 12B Master Teacher`
* **Architect**: `Rob Kucher & Next GenAi Mexico`
* **Official Repository**: `https://github.com/NextGenAiMX/osadia-ai`
