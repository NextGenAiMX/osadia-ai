"""
AXIOM Data Pipeline
===================
Preprocessing, tokenization, and data utilities for AXIOM training.

Supports:
- Web-scale text data (Common Crawl style)
- Instruction-following datasets (Alpaca, ShareGPT format)
- Code datasets
- Multilingual data (English + Spanish priority for The Data Factory)
- Document packing for efficient training (no wasted padding)

Copyright (c) 2026 Next Gen AI / Rob Kucher
"""

import os
import json
import random
from typing import List, Dict, Optional, Iterator, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset


# ─────────────────────────────────────────────────────────────────────────────
# PACKED SEQUENCE DATASET — Efficient training without wasted tokens
# ─────────────────────────────────────────────────────────────────────────────

class PackedSequenceDataset(Dataset):
    """
    Packs multiple documents into fixed-length sequences for training.
    This avoids padding waste and significantly improves training efficiency.
    
    Documents are separated by EOS tokens and packed greedily until
    max_seq_length is reached.
    """
    
    def __init__(
        self, 
        tokenized_docs: List[List[int]],
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        self.packed = self._pack(tokenized_docs)
        
    def _pack(self, docs: List[List[int]]) -> List[List[int]]:
        """Greedy document packing into fixed-length sequences."""
        packed_sequences = []
        current_pack = []
        
        for doc in docs:
            doc_with_eos = doc + [self.eos_token_id]
            
            if len(current_pack) + len(doc_with_eos) <= self.max_seq_length:
                current_pack.extend(doc_with_eos)
            else:
                # Flush current pack with padding
                if current_pack:
                    padded = current_pack + [self.pad_token_id] * (
                        self.max_seq_length - len(current_pack)
                    )
                    packed_sequences.append(padded[:self.max_seq_length])
                
                # Start new pack
                current_pack = doc_with_eos[:self.max_seq_length]
        
        # Flush remaining
        if current_pack:
            padded = current_pack + [self.pad_token_id] * (
                self.max_seq_length - len(current_pack)
            )
            packed_sequences.append(padded[:self.max_seq_length])
        
        return packed_sequences
    
    def __len__(self) -> int:
        return len(self.packed)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(self.packed[idx], dtype=torch.long)
        mask = (ids != self.pad_token_id).long()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTION DATASET — Fine-tuning on conversations
# ─────────────────────────────────────────────────────────────────────────────

class AXIOMInstructionDataset(Dataset):
    """
    Formats instruction-response pairs for AXIOM's special tokens.
    
    Supports formats:
    - Alpaca: {"instruction": ..., "input": ..., "output": ...}
    - ShareGPT: {"conversations": [{"from": "human", "value": ...}, ...]}
    - Custom AXIOM: {"system": ..., "messages": [{"role": ..., "content": ...}]}
    """
    
    AXIOM_SYSTEM_PROMPT = (
        "You are AXIOM, an advanced AI assistant built by Next Gen AI. "
        "You are thoughtful, precise, and capable of complex reasoning. "
        "When uncertain, you reflect before responding."
    )
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_seq_length: int = 2048,
        system_prompt: Optional[str] = None,
        train_on_inputs: bool = False  # If False, only train on assistant responses
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt or self.AXIOM_SYSTEM_PROMPT
        self.train_on_inputs = train_on_inputs
        
        self.examples = [self._format(item) for item in data]
        self.examples = [e for e in self.examples if e is not None]
    
    def _format_alpaca(self, item: Dict) -> str:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        if inp:
            instruction = f"{instruction}\n\nContext: {inp}"
        return (
            f"<|system|>{self.system_prompt}"
            f"<|user|>{instruction}"
            f"<|assistant|>{output}<|end|>"
        )
    
    def _format_sharegpt(self, item: Dict) -> Optional[str]:
        conversations = item.get("conversations", [])
        if not conversations:
            return None
        
        text = f"<|system|>{self.system_prompt}"
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role in ["human", "user"]:
                text += f"<|user|>{value}"
            elif role in ["gpt", "assistant"]:
                text += f"<|assistant|>{value}<|end|>"
        
        return text
    
    def _format(self, item: Dict) -> Optional[str]:
        if "conversations" in item:
            return self._format_sharegpt(item)
        elif "instruction" in item:
            return self._format_alpaca(item)
        elif "messages" in item:
            return self._format_messages(item)
        elif "text" in item:
            return item["text"]
        return None
    
    def _format_messages(self, item: Dict) -> str:
        system = item.get("system", self.system_prompt)
        text = f"<|system|>{system}"
        for msg in item.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                text += f"<|user|>{content}"
            elif role == "assistant":
                text += f"<|assistant|>{content}<|end|>"
        return text
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATA MIXING — Multi-source training data blend
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    name: str
    path: str
    weight: float = 1.0         # Sampling weight
    format: str = "jsonl"        # "jsonl", "text", "hf"
    language: str = "en"         # "en", "es", "multilingual"


class MixedDataset(Dataset):
    """
    Combines multiple data sources with configurable sampling weights.
    Useful for curriculum training: e.g., 70% web text, 20% code, 10% instruct.
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        
        if weights is None:
            weights = [1.0] * len(datasets)
        
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Build index mapping for weighted sampling
        self._build_index()
    
    def _build_index(self):
        """Build weighted index for sampling across datasets."""
        total = sum(len(ds) for ds in self.datasets)
        
        self.index = []
        for ds_idx, (ds, weight) in enumerate(zip(self.datasets, self.weights)):
            n_samples = int(total * weight)
            indices = list(range(len(ds))) * (n_samples // len(ds) + 1)
            random.shuffle(indices)
            self.index.extend([(ds_idx, i) for i in indices[:n_samples]])
        
        random.shuffle(self.index)
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, item_idx = self.index[idx]
        return self.datasets[ds_idx][item_idx % len(self.datasets[ds_idx])]


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOADER — Pre-configured public dataset sources
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDED_DATASETS = {
    "pretraining": [
        # High quality web text
        {"hf_name": "HuggingFaceFW/fineweb", "split": "train", "weight": 0.4},
        # Code (for reasoning and technical capability)
        {"hf_name": "bigcode/the-stack-dedup", "split": "train", "weight": 0.2},
        # Academic papers
        {"hf_name": "togethercomputer/RedPajama-Data-1T", "split": "train", "weight": 0.2},
        # Multilingual (Spanish priority for TDF)  
        {"hf_name": "uonlp/CulturaX", "split": "train", "weight": 0.1},
        # Books
        {"hf_name": "storytracer/US-PD-Books", "split": "train", "weight": 0.1},
    ],
    "finetuning": [
        # Instruction following
        {"hf_name": "tatsu-lab/alpaca", "split": "train", "weight": 0.3},
        {"hf_name": "WizardLM/WizardLM_evol_instruct_V2_196k", "split": "train", "weight": 0.3},
        # Reasoning
        {"hf_name": "open-thoughts/OpenThoughts-114k", "split": "train", "weight": 0.2},
        # Safety and helpfulness
        {"hf_name": "Anthropic/hh-rlhf", "split": "train", "weight": 0.2},
    ]
}


def print_data_plan():
    """Print the recommended data training plan."""
    print("\n" + "="*60)
    print("  AXIOM RECOMMENDED TRAINING DATA PLAN")
    print("="*60)
    
    print("\n📊 STAGE 1 — Pre-training")
    print("  ~1T tokens from:")
    for ds in RECOMMENDED_DATASETS["pretraining"]:
        print(f"    • {ds['hf_name']} ({int(ds['weight']*100)}%)")
    
    print("\n📝 STAGE 2 — Supervised Fine-Tuning (SFT)")
    print("  ~2M examples from:")
    for ds in RECOMMENDED_DATASETS["finetuning"]:
        print(f"    • {ds['hf_name']} ({int(ds['weight']*100)}%)")
    
    print("\n🔄 STAGE 3 — RLHF / DPO Alignment")
    print("  • Collect human preference data")
    print("  • Apply Direct Preference Optimization (DPO)")
    print("  • AXIOM-specific: optimize for OMS memory utilization")
    print("  • AXIOM-specific: train RRL confidence calibration")
    
    print("\n🌐 STAGE 4 — Multilingual Enhancement (Spanish)")
    print("  • Fine-tune on bilingual EN/ES data")
    print("  • Cultural context data for Mexico/LATAM")
    
    print("\n💡 Compute Estimate (7B parameter model):")
    print("  Stage 1: ~500 A100-GPU-hours")
    print("  Stage 2: ~50 A100-GPU-hours")
    print("  Stage 3: ~100 A100-GPU-hours")
    print("  Recommended: Google Cloud Vertex AI (A3 instances)")
    print("="*60 + "\n")
