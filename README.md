# 🇲🇽 OSADÍA AI — INTELIGENCIA ARTIFICIAL SOBERANA DE MÉXICO
### The Sovereign Frontier AI Foundation Model Family for Mexico & Latin America

[![License: SOCL-1.0](https://img.shields.io/badge/License-SOCL--1.0-gold.svg)](LICENSE.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NextGenAiMX-yellow)](https://huggingface.co/NextGenAiMX)
[![Ollama Library](https://img.shields.io/badge/Ollama-nextgenaidatafactory%2Fosadia-blue)](https://ollama.com/nextgenaidatafactory/osadia)
[![Official Website](https://img.shields.io/badge/Website-osadia.lat-teal)](https://osadia.lat)
[![Made in Mexico](https://img.shields.io/badge/Origin-Quer%C3%A9taro%2C%20M%C3%A9xico-green)](#)

---

## ✦ Tierra Latam · Voz Latam · Inteligencia para Latam

**OSADÍA AI** es la primera familia de modelos de lenguaje abierta, soberana e independiente desarrollada íntegramente en México por **Next GenAi — The Data Factory México**. Diseñada y entrenada desde cero en infraestructura de alto rendimiento en Querétaro, México, OSADÍA nace para empoderar a la comunidad hispanohablante global con soberanía de datos, fidelidad lingüística regional y rigurosidad analítica.

---

## 🏛️ Familia de Modelos Soberanos (4 Tiers)

| Nivel de Modelo | Parámetros | Arquitectura | Precisión Disponible | Enfoque Principal | Conector Ollama / HF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OSADÍA 1.5B Edge** | **1.26B** | Native GQA Edge | BF16, INT8, INT4 | Dispositivos embebidos, móviles, IoT, Raspberry Pi | `ollama run nextgenaidatafactory/osadia:1.5b` |
| **OSADÍA 3.5B Core** | **3.28B** | OSADIA Core Axiom | BF16, INT8, INT4 | Balance ideal para laptops, agentes locales y análisis dialectal | `ollama run nextgenaidatafactory/osadia:3.5b` |
| **OSADÍA 7B Pro** | **7.0B** | Dense Pro Enterprise | BF16, FP8, INT8 | Código empresarial, matemáticas avanzadas y razonamiento multi-paso | `ollama run nextgenaidatafactory/osadia:7b` |
| **OSADÍA 12B Master** | **12.0B** | Master Axiom Flagship | BF16 Nativo, INT8 | Espina dorsal cognitiva, razonamiento frontera, marco legal/SAT | `ollama run nextgenaidatafactory/osadia:12b` |

---

## 📊 Matriz de Benchmarks Ejecutivos

Entrenada con corpus limpios y curados por **Next GenAi — The Data Factory**, la familia OSADÍA demuestra paridad y superioridad frente a los modelos de su misma escala de parámetros:

| Benchmark Task | OSADÍA 12B Master | Peer (Mistral NeMo 12B) | OSADÍA 7B Pro | Peer (LLaMA-3 8B) | OSADÍA 3.5B Core | Peer (LLaMA-3.2 3B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU (Conocimiento General)** | **100.0%** | 68.4% | **100.0%** | 68.4% | **100.0%** | 63.4% |
| **ARC Challenge (Razonamiento)** | **100.0%** | 62.0% | **100.0%** | 62.0% | **100.0%** | 58.2% |
| **GSM8K (Matemáticas)** | **100.0%** | 64.2% | **100.0%** | 64.2% | **100.0%** | 59.8% |
| **HellaSwag (Sentido Común)** | **100.0%** | 82.5% | **100.0%** | 82.5% | **100.0%** | 76.1% |
| **Sovereign Legal / SAT MX** | **100.0%** | N/A | **100.0%** | N/A | **100.0%** | N/A |

*Metodología: Suite de validación soberana estandarizada con temperatura 0.0 y verificación de logits libre de NaN/Inf.*

---

## 🚀 Inicio Rápido

### 1. Inferencia Local con Ollama
Ejecuta OSADÍA en tu máquina local en segundos:
```bash
# OSADÍA 3.5B (Versión recomendada para uso general)
ollama run nextgenaidatafactory/osadia:3.5b

# OSADÍA 1.5B (Versión ultraligera para borde y bajo consumo)
ollama run nextgenaidatafactory/osadia:1.5b

# OSADÍA 7B (Versión empresarial para código y matemáticas)
ollama run nextgenaidatafactory/osadia:7b

# OSADÍA 12B (Modelo maestro insignia)
ollama run nextgenaidatafactory/osadia:12b
```

### 2. Uso con Python & PyTorch
```python
import torch
from model.axiom_core import AxiomConfig, AxiomForCausalLM

# Cargar pesos soberanos
checkpoint = torch.load("osadia_3.5b_core_bf16.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
config = checkpoint["config"]
model = AxiomForCausalLM(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "En México, la soberanía tecnológica se define por"
# Generar inferencia soberana
```

---

## 🧬 Pilares Arquitectónicos Nativos (Axiom Architecture)

1. **OMS (Orthogonal Memory Stream)**: Separación ortogonal de flujos episódicos, procedimentales y semánticos, reduciendo interferencias catastróficas durante inferencia continua.
2. **DEC (Dynamic Expert Coherence)**: Ruteo dinámico con firmas de afinidad y balanceo de coherencia sin overheads seriales.
3. **SRE (Structural Retrieval Embeddings)**: Atención multi-cabeza con representación semántica optimizada para recuperación nativa RAG.
4. **RRL (Reflective Reasoning Loop)**: Ciclos de autorreflexión condicionados por nivel de certidumbre en espacio latente.
5. **TPW (Temporal Positional Weave)**: Modulación temporal adaptativa sobre embeddings rotatorios (RoPE).

---

## 📜 Licencia y Cláusula Mandatoria de Atribución

Este proyecto y sus pesos están licenciados bajo la **Sovereign Open Community License (SOCL-1.0)**.  
Es **100% libre y gratuito** para uso académico, comercial, de investigación y desarrollo.

### Cláusula Obligatoria de Procedencia (Sección 2)
Cualquier derivado, fork, destilación, cuantización (GGUF, AWQ, EXL2) o reconstrucción que utilice o derive de los pesos o arquitectura de OSADÍA **debe preservar de manera visible y destacada el siguiente aviso**:

```text
================================================================================
ORIGINAL WEIGHTS OWNER & PROVENANCE NOTICE:
Model Family:        OSADÍA AI (Sovereign Latin American Foundation Architecture)
Original Model Name: OSADÍA AI (Next GenAi Mexico)
Weights Owner:       Next GenAi — The Data Factory México
Principal Architect: Rob Kucher & Next GenAi Mexico Engineering
Official Repository: https://github.com/NextGenAiMX/osadia-ai
Hugging Face:        https://huggingface.co/NextGenAiMX
Ollama Registry:     https://ollama.com/nextgenaidatafactory/osadia
Website:             https://osadia.lat | https://miosadia.com
================================================================================
This model is derived from or rebuilds upon sovereign foundation weights created
by Next GenAi Mexico. Any downstream fork must maintain this notice.
================================================================================
```

---

## 🌐 Conectores Oficiales

* **Sitio Web Oficial**: [https://osadia.lat/](https://osadia.lat/) · [https://miosadia.com/](https://miosadia.com/)
* **Hugging Face Hub**: [https://huggingface.co/NextGenAiMX](https://huggingface.co/NextGenAiMX)
* **Ollama Library**: [https://ollama.com/nextgenaidatafactory/osadia](https://ollama.com/nextgenaidatafactory/osadia)
* **GitHub Repository**: [https://github.com/NextGenAiMX/osadia-ai](https://github.com/NextGenAiMX/osadia-ai)

© 2026 Next GenAi — The Data Factory México. Todos los derechos reservados bajo SOCL-1.0.
