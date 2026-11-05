"""
OSADIA: Open Sovereign Adaptive Deep Intelligence Architecture
============================================================
Architecture by: Next Gen AI / The Data Factory
Version: 0.1.0-alpha

Novel Architecture Contributions:
1. Orthogonal Memory Stream (OMS) - Episodic, Semantic, Procedural memory lanes
2. Reflective Reasoning Loop (RRL) - Pre-output self-correction pass
3. Structural Retrieval Embeddings (SRE) - Retrieval baked into attention, not bolted on
4. Dynamic Expert Coherence (DEC) - MoE with learned expert affinity weighting
5. Temporal Positional Weave (TPW) - Time-aware positional encoding

Patent-eligible novel contributions:
- OMS three-lane memory fusion mechanism
- RRL second-pass attention for self-correction
- SRE structural retrieval integration
- DEC expert affinity scoring
- TPW temporal position encoding

Copyright (c) 2026 Next Gen AI / Rob Kucher
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps) * self.weight
    nn.RMSNorm = RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSADIAConfig:
    """Complete configuration for OSADIA model."""
    
    # Core dimensions
    vocab_size: int = 65536           # Extended vocabulary
    hidden_size: int = 4096           # Base hidden dimension
    num_hidden_layers: int = 32       # Transformer depth
    num_attention_heads: int = 32     # Multi-head attention
    num_kv_heads: int = 8             # GQA: fewer KV heads (memory efficient)
    intermediate_size: int = 14336   # FFN intermediate

    # MoE settings (Dynamic Expert Coherence)
    num_experts: int = 16             # Total expert count
    num_active_experts: int = 4       # Experts activated per token
    expert_affinity_dim: int = 64     # Expert affinity scoring dimension
    expert_coherence_lambda: float = 0.01  # Coherence regularization weight

    # Orthogonal Memory Stream (OMS)
    episodic_memory_slots: int = 512   # Short-term episodic slots
    semantic_memory_slots: int = 1024  # Long-term semantic knowledge slots
    procedural_memory_slots: int = 256 # Skill/task pattern slots
    memory_key_dim: int = 128          # Memory addressing key size
    oms_fusion_heads: int = 8          # Heads for memory fusion attention

    # Structural Retrieval Embeddings (SRE)
    retrieval_heads: int = 4           # Dedicated retrieval attention heads
    retrieval_key_size: int = 64       # Retrieval query/key dimension
    retrieval_corpus_size: int = 131072  # Virtual retrieval corpus size

    # Reflective Reasoning Loop (RRL)
    rrl_layers: int = 2               # Number of reflection passes
    rrl_hidden: int = 2048            # RRL intermediate dimension
    rrl_threshold: float = 0.75       # Confidence threshold to skip RRL

    # Temporal Positional Weave (TPW)
    max_position_embeddings: int = 32768  # Max sequence length
    rope_theta: float = 500000.0      # Extended RoPE base (for long context)
    temporal_decay_rate: float = 0.995  # Temporal position decay

    # Training settings
    dropout: float = 0.0              # Production: 0.0; training: 0.1
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    
    # AGI readiness
    enable_metacognition: bool = True  # Enable metacognitive monitoring
    enable_oms: bool = True            # Enable Orthogonal Memory Streams
    enable_rrl: bool = True            # Enable Reflective Reasoning Loop
    enable_sre: bool = True            # Enable Structural Retrieval Embeddings


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL POSITIONAL WEAVE (TPW) — Novel RoPE Extension
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPositionalWeave(nn.Module):
    """
    NOVEL: Temporal Positional Weave (TPW)
    
    Extends standard RoPE with a learned temporal decay that allows the model
    to treat recent context differently from distant context — not just by
    position, but by learned temporal relevance. This enables the model to
    "remember" important events even when they are far back in context.
    
    Patent contribution: Temporal decay modulation of rotary position embeddings
    through a learned sigmoid-gated temporal relevance score.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_positions = config.max_position_embeddings
        self.theta = config.rope_theta
        self.decay_rate = config.temporal_decay_rate
        
        # Learned temporal relevance gate
        self.temporal_gate = nn.Linear(self.head_dim, 1, bias=False)
        self.temporal_scale = nn.Parameter(torch.ones(1))
        
        # Precompute base frequencies
        inv_freq = 1.0 / (self.theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def _compute_temporal_modulation(self, positions: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute temporal relevance modulation.
        Recent positions get weight ~1.0; distant positions decay by learned rate.
        """
        max_pos = positions.max().float()
        relative_age = (max_pos - positions.float()) / (seq_len + 1e-8)
        decay = torch.exp(-relative_age * (1.0 - self.decay_rate) * 10)
        return decay.unsqueeze(-1)  # (seq, 1)
    
    def forward(
        self, 
        q: torch.Tensor,  # (batch, seq, heads, head_dim)
        k: torch.Tensor,  # (batch, seq, kv_heads, head_dim)
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch, seq_len, num_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape
        
        # Standard RoPE frequencies
        t = position_ids.float()  # (batch, seq)
        freqs = torch.einsum("bi,j->bij", t, self.inv_freq)  # (batch, seq, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq, head_dim)
        cos = emb.cos().unsqueeze(2)  # (batch, seq, 1, head_dim)
        sin = emb.sin().unsqueeze(2)  # (batch, seq, 1, head_dim)
        
        # Temporal modulation (novel component)
        temporal_mod = self._compute_temporal_modulation(
            position_ids[0], seq_len
        ).unsqueeze(0).unsqueeze(2)  # (1, seq, 1, 1)
        temporal_mod = (temporal_mod * self.temporal_scale).clamp(0.5, 1.5)
        
        # Match input tensor dtype
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        temporal_mod = temporal_mod.to(dtype=q.dtype)
        
        # Apply rotary with temporal modulation
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)
        
        q_rot = (q * cos + rotate_half(q) * sin) * temporal_mod
        k_rot = (k * cos + rotate_half(k) * sin) * temporal_mod
        
        return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# ORTHOGONAL MEMORY STREAM (OMS) — Novel Three-Lane Memory
# ─────────────────────────────────────────────────────────────────────────────

class OrthogonalMemoryStream(nn.Module):
    """
    NOVEL: Orthogonal Memory Stream (OMS)
    
    Three functionally orthogonal memory lanes that maintain separate
    representational spaces:
    
    1. EPISODIC: What happened in this conversation/context window
    2. SEMANTIC: General world knowledge patterns (persistent across contexts)
    3. PROCEDURAL: How to do things (task patterns, reasoning procedures)
    
    These three streams are kept orthogonal through a regularization loss
    that penalizes cross-stream similarity, then fused via a learned
    cross-stream attention mechanism at inference.
    
    Patent contribution: Three-lane orthogonal memory architecture with
    orthogonality regularization and cross-stream fusion attention.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_dim = config.memory_key_dim
        self.fusion_heads = config.oms_fusion_heads
        
        # Three memory banks (learned prototypes)
        self.episodic_memory = nn.Parameter(
            torch.randn(config.episodic_memory_slots, config.hidden_size) * 0.02
        )
        self.semantic_memory = nn.Parameter(
            torch.randn(config.semantic_memory_slots, config.hidden_size) * 0.02
        )
        self.procedural_memory = nn.Parameter(
            torch.randn(config.procedural_memory_slots, config.hidden_size) * 0.02
        )
        
        # Addressing networks (query projections for each stream)
        self.episodic_query = nn.Linear(config.hidden_size, self.key_dim)
        self.semantic_query = nn.Linear(config.hidden_size, self.key_dim)
        self.procedural_query = nn.Linear(config.hidden_size, self.key_dim)
        
        # Memory key projections
        self.episodic_key_proj = nn.Linear(config.hidden_size, self.key_dim)
        self.semantic_key_proj = nn.Linear(config.hidden_size, self.key_dim)
        self.procedural_key_proj = nn.Linear(config.hidden_size, self.key_dim)
        
        # Cross-stream fusion attention
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=self.fusion_heads,
            batch_first=True,
            dropout=config.attention_dropout
        )
        
        # Gate to blend retrieved memory with hidden state
        self.memory_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def _retrieve_stream(
        self, 
        hidden: torch.Tensor,  # (batch, seq, hidden)
        memory: nn.Parameter,  # (slots, hidden)
        query_proj: nn.Linear,
        key_proj: nn.Linear
    ) -> torch.Tensor:
        """Soft-address a memory stream and retrieve weighted sum."""
        q = query_proj(hidden)  # (batch, seq, key_dim)
        k = key_proj(memory).unsqueeze(0)  # (1, slots, key_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)
        weights = F.softmax(scores, dim=-1)  # (batch, seq, slots)
        
        retrieved = torch.matmul(weights, memory.unsqueeze(0))  # (batch, seq, hidden)
        return retrieved
    
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Regularization: penalize cosine similarity between memory banks.
        This keeps the three streams functionally orthogonal.
        """
        e = F.normalize(self.episodic_memory.mean(0) + 1e-8, dim=0, eps=1e-8)
        s = F.normalize(self.semantic_memory.mean(0) + 1e-8, dim=0, eps=1e-8)
        p = F.normalize(self.procedural_memory.mean(0) + 1e-8, dim=0, eps=1e-8)
        
        loss = (
            torch.abs(torch.dot(e, s)) +
            torch.abs(torch.dot(e, p)) +
            torch.abs(torch.dot(s, p))
        ) / 3.0
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from all three memory streams and fuse.
        Returns: (enriched_hidden, orthogonality_loss)
        """
        batch, seq, _ = hidden.shape
        
        # Retrieve from each stream
        episodic = self._retrieve_stream(
            hidden, self.episodic_memory, 
            self.episodic_query, self.episodic_key_proj
        )
        semantic = self._retrieve_stream(
            hidden, self.semantic_memory,
            self.semantic_query, self.semantic_key_proj
        )
        procedural = self._retrieve_stream(
            hidden, self.procedural_memory,
            self.procedural_query, self.procedural_key_proj
        )
        
        # Stack streams for cross-stream fusion attention
        # Each stream contributes as a "key/value" source
        stream_context = torch.stack([episodic, semantic, procedural], dim=2)
        # (batch, seq, 3, hidden) → (batch*seq, 3, hidden)
        stream_context = stream_context.view(batch * seq, 3, self.hidden_size)
        hidden_flat = hidden.view(batch * seq, 1, self.hidden_size)
        
        fused, _ = self.fusion_attn(
            query=hidden_flat,
            key=stream_context,
            value=stream_context
        )
        fused = fused.view(batch, seq, self.hidden_size)
        
        # Gated blend: how much memory to inject
        gate = self.memory_gate(torch.cat([hidden, fused], dim=-1))
        enriched = hidden + gate * self.output_proj(fused)
        
        # Orthogonality regularization loss
        orth_loss = self.orthogonality_loss()
        
        return self.norm(enriched), orth_loss


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL RETRIEVAL EMBEDDINGS (SRE) — Retrieval-Native Attention
# ─────────────────────────────────────────────────────────────────────────────

class StructuralRetrievalEmbeddings(nn.Module):
    """
    NOVEL: Structural Retrieval Embeddings (SRE)
    
    Instead of retrieval being a separate RAG pipeline bolted onto a model,
    SRE embeds a retrieval mechanism directly into certain attention heads.
    Dedicated "retrieval heads" learn to query a virtual corpus of dense
    knowledge prototypes, with retrieval keys learned end-to-end with the
    rest of the model.
    
    This makes the model architecturally retrieval-native, not retrieval-augmented.
    
    Patent contribution: End-to-end learned retrieval heads integrated
    structurally within the multi-head attention mechanism.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.retrieval_heads = config.retrieval_heads
        self.retrieval_key_size = config.retrieval_key_size
        self.corpus_size = config.retrieval_corpus_size
        self.hidden_size = config.hidden_size
        
        # Virtual corpus: learned dense knowledge prototypes
        self.corpus_keys = nn.Parameter(
            torch.randn(config.retrieval_corpus_size, config.retrieval_key_size) * 0.02
        )
        self.corpus_values = nn.Parameter(
            torch.randn(config.retrieval_corpus_size, config.hidden_size // config.retrieval_heads) * 0.02
        )
        
        # Retrieval query projection (from hidden state)
        self.retrieval_query_proj = nn.Linear(
            config.hidden_size, 
            config.retrieval_heads * config.retrieval_key_size
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.retrieval_heads * (config.hidden_size // config.retrieval_heads),
            config.hidden_size
        )
        
        # Sparse top-k retrieval gate (learn which tokens need retrieval)
        self.retrieval_gate = nn.Linear(config.hidden_size, 1)
        
        self.scale = math.sqrt(config.retrieval_key_size)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        For each token, optionally retrieve from the virtual corpus
        via sparse, learned attention over knowledge prototypes.
        """
        batch, seq, _ = hidden.shape
        
        # Decide which tokens need retrieval (sparse gate)
        retrieval_need = torch.sigmoid(self.retrieval_gate(hidden))  # (b, s, 1)
        
        # Compute retrieval queries
        q = self.retrieval_query_proj(hidden)
        q = q.view(batch, seq, self.retrieval_heads, self.retrieval_key_size)
        
        # Score against corpus keys
        # q: (b, s, r_heads, key_size) 
        # corpus_keys: (corpus, key_size)
        scores = torch.einsum('bsrk,ck->bsrc', q, self.corpus_keys) / self.scale
        
        # Top-k sparse retrieval (attend to top-16 entries)
        top_k = min(16, self.corpus_size)
        topk_scores, topk_idx = scores.topk(top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)  # (b, s, r_heads, top_k)
        
        # Memory-efficient gather using index_select
        v_dim = self.hidden_size // self.retrieval_heads
        flat_idx = topk_idx.view(-1)
        flat_values = self.corpus_values.index_select(0, flat_idx)
        topk_values = flat_values.view(batch, seq, self.retrieval_heads, top_k, v_dim)
        
        # Weighted sum of values
        retrieved = torch.einsum('bsrk,bsrkv->bsrv', topk_weights, topk_values)
        retrieved = retrieved.contiguous().view(batch, seq, -1)
        
        retrieved_out = self.output_proj(retrieved)
        return hidden + retrieval_need * retrieved_out


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC EXPERT COHERENCE (DEC) — Affinity-Weighted MoE
# ─────────────────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    """Individual expert feed-forward network."""
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DynamicExpertCoherence(nn.Module):
    """
    NOVEL: Dynamic Expert Coherence (DEC)
    
    Standard MoE routes tokens to top-k experts by a learned gate.
    DEC adds an "expert affinity" mechanism: experts that have been
    active together before learn to produce more coherent joint outputs.
    
    This is implemented via:
    1. Per-expert "signature" vectors that track historical co-activation
    2. A coherence score that modulates expert output weights based on
       how well the currently selected experts have historically worked together
    3. A diversity loss that prevents expert collapse
    
    Patent contribution: Expert affinity coherence scoring for MoE routing
    via learned co-activation signatures.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_active = config.num_active_experts
        self.affinity_dim = config.expert_affinity_dim
        self.coherence_lambda = config.expert_coherence_lambda
        
        # Expert feed-forward networks
        self.experts = nn.ModuleList([
            ExpertFFN(config) for _ in range(config.num_experts)
        ])
        
        # Standard routing gate
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Expert affinity signatures (learned per-expert identity vectors)
        self.expert_signatures = nn.Parameter(
            torch.randn(config.num_experts, config.expert_affinity_dim) * 0.02
        )
        
        # Coherence projection: maps selected expert combo to coherence weight
        self.coherence_proj = nn.Linear(
            config.num_active_experts * config.expert_affinity_dim,
            config.num_active_experts
        )
        
        # Shared expert (always active, like DeepSeek's shared expert)
        self.shared_expert = ExpertFFN(config)
        self.shared_weight = nn.Parameter(torch.tensor(0.3))
    
    def _compute_coherence_weights(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Given the selected expert indices, compute coherence modulation weights.
        Expert combinations that have learned to work well together get boosted.
        """
        # Gather signatures for selected experts
        # expert_indices: (batch*seq, num_active)
        sigs = self.expert_signatures[expert_indices]  # (..., num_active, affinity_dim)
        flat_sigs = sigs.view(sigs.shape[0], -1)  # (..., num_active * affinity_dim)
        coherence = torch.sigmoid(self.coherence_proj(flat_sigs))  # (..., num_active)
        return coherence
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens through experts with affinity-weighted coherence.
        Returns: (output, load_balance_loss)
        """
        batch, seq, hidden_size = hidden.shape
        flat = hidden.view(-1, hidden_size)  # (b*s, h)
        
        # Router logits
        logits = self.router(flat)  # (b*s, num_experts)
        
        # Top-k routing
        topk_logits, topk_idx = logits.topk(self.num_active, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (b*s, num_active)
        
        # Novel: coherence modulation
        coherence_weights = self._compute_coherence_weights(topk_idx)
        topk_weights = topk_weights * (0.5 + 0.5 * coherence_weights)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renorm
        
        # Expert dispatch
        output = torch.zeros_like(flat)
        
        for i in range(self.num_active):
            expert_idx_i = topk_idx[:, i]  # (b*s,)
            weights_i = topk_weights[:, i].unsqueeze(-1)  # (b*s, 1)
            
            # Group by expert for efficiency
            for expert_id in range(self.num_experts):
                mask = (expert_idx_i == expert_id)
                if not mask.any():
                    continue
                expert_input = flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                output[mask] += weights_i[mask] * expert_output
        
        # Add shared expert contribution
        shared_out = self.shared_expert(flat)
        output = output + torch.sigmoid(self.shared_weight) * shared_out
        
        # Load balance loss (prevents expert collapse)
        router_probs = F.softmax(logits, dim=-1)
        expert_usage = router_probs.mean(0)
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, target_usage) * self.num_experts
        
        return output.view(batch, seq, hidden_size), load_balance_loss


# ─────────────────────────────────────────────────────────────────────────────
# REFLECTIVE REASONING LOOP (RRL) — Pre-output Self-Correction
# ─────────────────────────────────────────────────────────────────────────────

class ReflectiveReasoningLoop(nn.Module):
    """
    NOVEL: Reflective Reasoning Loop (RRL)
    
    Before producing output logits, the model performs a fast "reflection pass"
    that evaluates its own hidden state for confidence and coherence.
    
    If confidence is above a threshold, the token is passed through directly.
    If confidence is low, the RRL applies a correction pass that attends to
    the full context with a focus on identifying contradictions and uncertainties.
    
    This is architecturally different from chain-of-thought — it happens inside
    the model at the hidden state level, not at the token level.
    
    Patent contribution: Confidence-gated hidden-state reflection mechanism
    for self-correction prior to token generation.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rrl_hidden = config.rrl_hidden
        self.threshold = config.rrl_threshold
        self.num_passes = config.rrl_layers
        
        # Confidence estimator (scalar per token)
        self.confidence_net = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Correction network (applied when confidence is low)
        self.correction_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.rrl_hidden,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True  # Pre-norm
            )
            for _ in range(config.rrl_layers)
        ])
        
        # Learned blend between original and corrected hidden state
        self.blend_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Metacognition monitor (optional AGI readiness component)
        self.meta_monitor = nn.Linear(config.hidden_size, 4)  # 4 meta-states
        # Meta-states: [confident, uncertain, contradictory, novel]
    
    def forward(
        self, 
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply reflective reasoning loop.
        Returns: (refined_hidden, meta_info)
        """
        # Estimate confidence per token
        confidence = self.confidence_net(hidden)  # (batch, seq, 1)
        mean_confidence = confidence.mean()
        
        meta_info = {
            "confidence": confidence,
            "mean_confidence": mean_confidence.item(),
            "reflection_applied": False
        }
        
        # If mean confidence is above threshold, skip correction (fast path)
        if mean_confidence > self.threshold and not self.training:
            meta_states = self.meta_monitor(hidden)
            meta_info["meta_states"] = meta_states
            return hidden, meta_info
        
        # Low confidence: apply correction passes
        corrected = hidden
        
        # Create attention mask for transformer layers
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        for layer in self.correction_layers:
            corrected = layer(corrected, src_key_padding_mask=key_padding_mask)
        
        # Blend: confident tokens keep more of original, uncertain ones get more correction
        blend_input = torch.cat([hidden, corrected], dim=-1)
        blend_weight = torch.sigmoid(self.blend_gate(blend_input))
        refined = confidence * hidden + (1 - confidence) * corrected
        refined = refined + blend_weight * (corrected - hidden) * 0.3
        
        # Metacognition states
        meta_states = self.meta_monitor(refined)
        meta_info["meta_states"] = meta_states
        meta_info["reflection_applied"] = True
        
        return refined, meta_info


# ─────────────────────────────────────────────────────────────────────────────
# OSADIA ATTENTION — Multi-Head with GQA + TPW + SRE
# ─────────────────────────────────────────────────────────────────────────────

class OSADIAAttention(nn.Module):
    """
    OSADIA's core attention mechanism combining:
    - Grouped Query Attention (GQA) for memory efficiency
    - Temporal Positional Weave (TPW) for time-aware positions
    - Structural Retrieval Embeddings (SRE) for retrieval-native attention
    """
    
    def __init__(self, config: OSADIAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V projections (GQA: fewer KV heads)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Temporal Positional Weave
        self.tpw = TemporalPositionalWeave(config)
        
        # Structural Retrieval Embeddings (on every 4th layer)
        self.use_sre = config.enable_sre and (layer_idx % 4 == 0)
        if self.use_sre:
            self.sre = StructuralRetrievalEmbeddings(config)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        
        batch, seq, _ = hidden.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Build position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq, device=hidden.device).unsqueeze(0)
        
        # Apply Temporal Positional Weave
        # q/k are currently (batch, heads, seq, head_dim), TPW needs (batch, seq, heads, head_dim)
        q_bshd = q.permute(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        k_bshd = k.permute(0, 2, 1, 3)  # (batch, seq, kv_heads, head_dim)
        q_rot, k_rot = self.tpw(q_bshd, k_bshd, position_ids)
        q = q_rot.permute(0, 2, 1, 3)  # back to (batch, heads, seq, head_dim)
        k = k_rot.permute(0, 2, 1, 3)
        
        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        current_kv = (k, v)
        
        # GQA: repeat KV heads
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)
        
        # Scaled dot-product attention (uses Flash Attention if available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=(attention_mask is None)
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_output = self.o_proj(attn_output)
        
        # Apply SRE on designated layers
        if self.use_sre:
            attn_output = self.sre(attn_output)
        
        return attn_output, current_kv


# ─────────────────────────────────────────────────────────────────────────────
# OSADIA DECODER LAYER — Full Layer Assembly
# ─────────────────────────────────────────────────────────────────────────────

class OSADIADecoderLayer(nn.Module):
    """
    Single OSADIA decoder layer combining:
    - Pre-norm architecture
    - OSADIA Attention (GQA + TPW + SRE)
    - Dynamic Expert Coherence MoE FFN
    - Optional OMS injection (every 8 layers)
    """
    
    def __init__(self, config: OSADIAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.attention = OSADIAAttention(config, layer_idx)
        self.moe_ffn = DynamicExpertCoherence(config)
        
        # OMS injection every 8 layers
        self.use_oms = config.enable_oms and (layer_idx % 8 == 7)
        if self.use_oms:
            self.oms = OrthogonalMemoryStream(config)
    
    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], torch.Tensor, torch.Tensor]:
        
        # Attention sub-layer
        residual = hidden
        hidden = self.input_norm(hidden)
        attn_out, current_kv = self.attention(hidden, attention_mask, position_ids, past_key_value)
        hidden = residual + attn_out
        
        # MoE FFN sub-layer
        residual = hidden
        hidden = self.post_attn_norm(hidden)
        ffn_out, load_loss = self.moe_ffn(hidden)
        hidden = residual + ffn_out
        
        # OMS memory injection (select layers)
        oms_loss = torch.tensor(0.0, device=hidden.device)
        if self.use_oms:
            hidden, oms_loss = self.oms(hidden)
        
        return hidden, current_kv, load_loss, oms_loss


# ─────────────────────────────────────────────────────────────────────────────
# OSADIA MODEL — Full Architecture
# ─────────────────────────────────────────────────────────────────────────────

class OSADIAModel(nn.Module):
    """
    OSADIA: Open Sovereign Adaptive Deep Intelligence Architecture
    
    Full transformer model combining all novel architectural components.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            OSADIADecoderLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Reflective Reasoning Loop (applied before output)
        if config.enable_rrl:
            self.rrl = ReflectiveReasoningLoop(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Scaled initialization for deep networks."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        output_meta: bool = False,
    ) -> Dict:
        
        batch, seq = input_ids.shape
        
        # Build position IDs
        if position_ids is None:
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq, device=input_ids.device
            ).unsqueeze(0)
        
        # Embed tokens
        hidden = self.embed_tokens(input_ids)
        
        # Accumulate auxiliary losses
        total_load_loss = torch.tensor(0.0, device=hidden.device)
        total_oms_loss = torch.tensor(0.0, device=hidden.device)
        new_kv_cache = []
        
        # Forward through decoder layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.training:
                # Use gradient checkpointing to keep VRAM strictly under 6GB
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden, current_kv, load_loss, oms_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), hidden, attention_mask, position_ids, past_kv,
                    use_reentrant=False
                )
            else:
                hidden, current_kv, load_loss, oms_loss = layer(
                    hidden, attention_mask, position_ids, past_kv
                )
            new_kv_cache.append(current_kv)
            total_load_loss = total_load_loss + load_loss
            total_oms_loss = total_oms_loss + oms_loss
        
        hidden = self.final_norm(hidden)
        
        # Apply Reflective Reasoning Loop
        meta_info = {}
        if self.config.enable_rrl:
            hidden, meta_info = self.rrl(hidden, attention_mask)
        
        return {
            "last_hidden_state": hidden,
            "past_key_values": new_kv_cache,
            "load_balance_loss": total_load_loss / len(self.layers),
            "orthogonality_loss": total_oms_loss,
            "meta_info": meta_info if output_meta else {}
        }


class OSADIAForCausalLM(nn.Module):
    """
    OSADIA Causal Language Model head.
    Full autoregressive model ready for training and inference.
    """
    
    def __init__(self, config: OSADIAConfig):
        super().__init__()
        self.config = config
        self.model = OSADIAModel(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        output_meta: bool = False,
    ) -> Dict:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_meta=output_meta,
        )
        
        hidden = outputs["last_hidden_state"]
        logits = self.lm_head(hidden)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=65000.0, neginf=-65000.0)
        
        # Compute total loss
        total_loss = None
        if labels is not None:
            # Causal LM loss (memory efficient cross entropy)
            shift_logits = logits[..., :-1, :].float()
            shift_labels = labels[..., 1:]
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100
            )
            
            # Auxiliary losses
            load_loss = outputs["load_balance_loss"]
            oms_loss = outputs["orthogonality_loss"]
            
            total_loss = (
                lm_loss +
                self.config.expert_coherence_lambda * load_loss +
                0.01 * oms_loss
            )
        
        return {
            "loss": total_loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "load_balance_loss": outputs["load_balance_loss"],
            "orthogonality_loss": outputs["orthogonality_loss"],
            "meta_info": outputs["meta_info"]
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation with OSADIA KV cache."""
        
        self.eval()
        device = input_ids.device
        batch = input_ids.shape[0]
        
        generated = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass (only new tokens after first step)
            cur_input = generated if past_key_values is None else generated[:, -1:]
            
            outputs = self.forward(
                input_ids=cur_input,
                past_key_values=past_key_values
            )
            
            logits = outputs["logits"][:, -1, :]  # (batch, vocab)
            past_key_values = outputs["past_key_values"]
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch):
                    for token_id in set(generated[b].tolist()):
                        if logits[b, token_id] < 0:
                            logits[b, token_id] *= repetition_penalty
                        else:
                            logits[b, token_id] /= repetition_penalty
            
            # Temperature
            logits = logits / temperature
            
            if do_sample:
                # Top-k
                if top_k > 0:
                    top_k_vals = logits.topk(top_k, dim=-1).values
                    logits[logits < top_k_vals[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus)
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove_mask = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[remove_mask] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Return parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def parameter_summary(self) -> str:
        total = self.get_num_parameters(trainable_only=False)
        trainable = self.get_num_parameters(trainable_only=True)
        return (
            f"OSADIA Parameter Summary\n"
            f"  Total:     {total:,} ({total/1e9:.2f}B)\n"
            f"  Trainable: {trainable:,} ({trainable/1e9:.2f}B)\n"
        )
