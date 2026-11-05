# coding=utf-8
"""
OSADIA Model Configuration
Sovereign Latin American Foundation Architecture with Axiom 5-Pillar Neural Core
Next GenAi Mexico & The Data Factory
"""

from transformers.configuration_utils import PretrainedConfig

class OSADIAConfig(PretrainedConfig):
    model_type = "osadia"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 1536,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_kv_heads: int = 4,
        intermediate_size: int = 4096,
        num_experts: int = 1,
        num_active_experts: int = 1,
        expert_affinity_dim: int = 64,
        expert_coherence_lambda: float = 0.01,
        episodic_memory_slots: int = 512,
        semantic_memory_slots: int = 1024,
        procedural_memory_slots: int = 256,
        memory_key_dim: int = 128,
        oms_fusion_heads: int = 8,
        retrieval_heads: int = 4,
        retrieval_key_size: int = 64,
        retrieval_corpus_size: int = 131072,
        rrl_layers: int = 2,
        rrl_hidden: int = 2048,
        rrl_threshold: float = 0.75,
        max_position_embeddings: int = 2048,
        rope_theta: float = 500000.0,
        temporal_decay_rate: float = 0.995,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        enable_metacognition: bool = False,
        enable_oms: bool = False,
        enable_rrl: bool = False,
        enable_sre: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.expert_affinity_dim = expert_affinity_dim
        self.expert_coherence_lambda = expert_coherence_lambda
        self.episodic_memory_slots = episodic_memory_slots
        self.semantic_memory_slots = semantic_memory_slots
        self.procedural_memory_slots = procedural_memory_slots
        self.memory_key_dim = memory_key_dim
        self.oms_fusion_heads = oms_fusion_heads
        self.retrieval_heads = retrieval_heads
        self.retrieval_key_size = retrieval_key_size
        self.retrieval_corpus_size = retrieval_corpus_size
        self.rrl_layers = rrl_layers
        self.rrl_hidden = rrl_hidden
        self.rrl_threshold = rrl_threshold
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.temporal_decay_rate = temporal_decay_rate
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.enable_metacognition = enable_metacognition
        self.enable_oms = enable_oms
        self.enable_rrl = enable_rrl
        self.enable_sre = enable_sre
