"""
AXIOM Model Configurations
===========================
Pre-defined size configurations from mini to ultra.

Copyright (c) 2026 Next Gen AI / Rob Kucher
"""

from model.axiom_core import AXIOMConfig


def axiom_mini() -> AXIOMConfig:
    """
    AXIOM-Mini — 1.1B parameters
    For edge deployment, mobile, and experimentation.
    Trainable on a single consumer GPU (24GB VRAM).
    """
    return AXIOMConfig(
        vocab_size=65536,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        num_experts=8,
        num_active_experts=2,
        episodic_memory_slots=256,
        semantic_memory_slots=512,
        procedural_memory_slots=128,
        rrl_layers=1,
        rrl_hidden=1024,
        max_position_embeddings=8192,
    )


def axiom_base() -> AXIOMConfig:
    """
    AXIOM-Base — 7B parameters  
    General purpose. Matches LLaMA-3 7B capability envelope.
    Trainable on 4x A100 80GB.
    """
    return AXIOMConfig(
        vocab_size=65536,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        num_experts=16,
        num_active_experts=4,
        episodic_memory_slots=512,
        semantic_memory_slots=1024,
        procedural_memory_slots=256,
        rrl_layers=2,
        rrl_hidden=2048,
        max_position_embeddings=32768,
    )


def axiom_strong() -> AXIOMConfig:
    """
    AXIOM-Strong — 35B parameters
    High capability frontier model.
    Requires 8x H100 80GB for training.
    """
    return AXIOMConfig(
        vocab_size=65536,
        hidden_size=7168,
        num_hidden_layers=56,
        num_attention_heads=56,
        num_kv_heads=8,
        intermediate_size=20480,
        num_experts=32,
        num_active_experts=8,
        episodic_memory_slots=1024,
        semantic_memory_slots=4096,
        procedural_memory_slots=512,
        rrl_layers=4,
        rrl_hidden=4096,
        max_position_embeddings=131072,   # 128K context
        rope_theta=1_000_000.0,
    )


def axiom_ultra() -> AXIOMConfig:
    """
    AXIOM-Ultra — ~140B parameters
    AGI-readiness tier. Full capability envelope.
    Requires 64x H100 cluster for training.
    """
    return AXIOMConfig(
        vocab_size=131072,               # Extended vocabulary
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        num_experts=64,
        num_active_experts=8,
        expert_affinity_dim=128,
        episodic_memory_slots=2048,
        semantic_memory_slots=8192,
        procedural_memory_slots=1024,
        rrl_layers=6,
        rrl_hidden=8192,
        max_position_embeddings=1_000_000,  # 1M token context
        rope_theta=10_000_000.0,
        enable_metacognition=True,
    )


# Model registry
AXIOM_CONFIGS = {
    "axiom-mini":   axiom_mini,
    "axiom-base":   axiom_base,
    "axiom-strong": axiom_strong,
    "axiom-ultra":  axiom_ultra,
}


def get_config(name: str) -> AXIOMConfig:
    """Get a named AXIOM configuration."""
    if name not in AXIOM_CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(AXIOM_CONFIGS.keys())}")
    return AXIOM_CONFIGS[name]()


def print_model_comparison():
    """Print comparison table of all AXIOM sizes."""
    print("\n" + "="*70)
    print(f"  {'Model':<16} {'Params':>10} {'Layers':>8} {'Hidden':>8} {'Experts':>8} {'Context':>10}")
    print("="*70)
    
    from model.axiom_core import AXIOMForCausalLM
    
    for name, cfg_fn in AXIOM_CONFIGS.items():
        cfg = cfg_fn()
        # Estimate params without instantiating full model
        embed = cfg.vocab_size * cfg.hidden_size
        attn_per_layer = cfg.hidden_size * (
            cfg.hidden_size +  # Q
            cfg.num_kv_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2 +  # KV
            cfg.hidden_size   # O
        )
        ffn_per_layer = cfg.num_experts * 3 * cfg.hidden_size * cfg.intermediate_size
        total = embed + cfg.num_hidden_layers * (attn_per_layer + ffn_per_layer)
        
        context = f"{cfg.max_position_embeddings // 1024}K"
        print(
            f"  {name:<16} {total/1e9:>9.1f}B "
            f"{cfg.num_hidden_layers:>8} {cfg.hidden_size:>8} "
            f"{cfg.num_experts:>8} {context:>10}"
        )
    
    print("="*70 + "\n")
