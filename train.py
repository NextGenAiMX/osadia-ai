"""
AXIOM Training Entry Point
===========================
Start a training run with:

    python train.py --model axiom-mini --data ./data/train.jsonl

Or import for programmatic use:

    from train import run_training
    run_training(model_name="axiom-mini", data_path="./data/train.jsonl")

Copyright (c) 2026 Next Gen AI / Rob Kucher
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from model.axiom_core import AXIOMForCausalLM
from training.trainer import AXIOMTrainer, AXIOMTrainingConfig
from config.model_configs import get_config, print_model_comparison, AXIOM_CONFIGS
from data.pipeline import print_data_plan


def run_training(
    model_name: str = "axiom-mini",
    data_path: str = "./data/train.jsonl",
    val_data_path: str = None,
    output_dir: str = "./axiom_checkpoints",
    tokenizer: str = "mistralai/Mistral-7B-v0.1",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 3e-4,
    max_seq_length: int = 2048,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    use_wandb: bool = False,
    resume_from: str = None,
):
    """
    Launch AXIOM training.
    
    Args:
        model_name: One of axiom-mini, axiom-base, axiom-strong, axiom-ultra
        data_path: Path to training data (.jsonl or .txt)
        output_dir: Where to save checkpoints
        ...
    """
    print("\n" + "█"*60)
    print("  AXIOM — Adaptive eXpanded Intelligence with Orthogonal Memory")
    print("  Next Gen AI / The Data Factory")
    print("  v0.1.0-alpha")
    print("█"*60)
    
    # Load model config
    model_config = get_config(model_name)
    print(f"\n  Model: {model_name.upper()}")
    print(f"  Layers: {model_config.num_hidden_layers}")
    print(f"  Hidden: {model_config.hidden_size}")
    print(f"  Experts: {model_config.num_experts} (active: {model_config.num_active_experts})")
    print(f"  OMS slots: E={model_config.episodic_memory_slots} S={model_config.semantic_memory_slots} P={model_config.procedural_memory_slots}")
    print(f"  Context: {model_config.max_position_embeddings:,} tokens\n")
    
    # Build training config
    train_config = AXIOMTrainingConfig(
        output_dir=output_dir,
        data_path=data_path,
        val_data_path=val_data_path,
        tokenizer_name=tokenizer,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=lr,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        use_wandb=use_wandb,
        resume_from_checkpoint=resume_from,
    )
    
    # Build and train
    trainer = AXIOMTrainer(model_config, train_config)
    metrics = trainer.train()
    
    print(f"\n✓ AXIOM training complete.")
    print(f"  Final model: {output_dir}/best_model/")
    
    return trainer.model


def run_inference_test(model_name: str = "axiom-mini", checkpoint_path: str = None):
    """Quick inference test to verify model works."""
    import torch
    
    print(f"\nRunning inference test for {model_name}...")
    
    model_config = get_config(model_name)
    model = AXIOMForCausalLM(model_config)
    
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    
    model.eval()
    print(model.parameter_summary())
    
    # Forward pass test
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, output_meta=True)
    
    print(f"  Loss:           {outputs['loss'].item():.4f}")
    print(f"  Logits shape:   {outputs['logits'].shape}")
    print(f"  Load balance ℒ: {outputs['load_balance_loss'].item():.6f}")
    print(f"  OMS ortho ℒ:    {outputs['orthogonality_loss'].item():.6f}")
    
    if outputs["meta_info"]:
        meta = outputs["meta_info"]
        print(f"  RRL confidence: {meta.get('mean_confidence', 0):.3f}")
        print(f"  RRL applied:    {meta.get('reflection_applied', False)}")
    
    # Generation test
    test_input = torch.randint(0, model_config.vocab_size, (1, 10))
    generated = model.generate(test_input, max_new_tokens=20, do_sample=False)
    print(f"  Generation:     input {test_input.shape} → output {generated.shape}")
    
    print("\n✓ Inference test passed.")
    return model


def main():
    parser = argparse.ArgumentParser(description="AXIOM Training CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train AXIOM")
    train_parser.add_argument("--model", choices=list(AXIOM_CONFIGS.keys()), default="axiom-mini")
    train_parser.add_argument("--data", type=str, default="./data/train.jsonl")
    train_parser.add_argument("--val-data", type=str, default=None)
    train_parser.add_argument("--output", type=str, default="./axiom_checkpoints")
    train_parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--grad-accum", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--max-seq-length", type=int, default=2048)
    train_parser.add_argument("--fp32", action="store_true", help="Use fp32 (default bf16)")
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--resume", type=str, default=None)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test model inference")
    test_parser.add_argument("--model", choices=list(AXIOM_CONFIGS.keys()), default="axiom-mini")
    test_parser.add_argument("--checkpoint", type=str, default=None)
    
    # Info commands
    subparsers.add_parser("models", help="Show model size comparison")
    subparsers.add_parser("data-plan", help="Show recommended training data plan")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_training(
            model_name=args.model,
            data_path=args.data,
            val_data_path=args.val_data,
            output_dir=args.output,
            tokenizer=args.tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_seq_length=args.max_seq_length,
            bf16=not args.fp32,
            use_wandb=args.wandb,
            resume_from=args.resume,
        )
    
    elif args.command == "test":
        run_inference_test(args.model, args.checkpoint)
    
    elif args.command == "models":
        print_model_comparison()
    
    elif args.command == "data-plan":
        print_data_plan()
    
    else:
        # No command: run inference test as demo
        print_model_comparison()
        run_inference_test("axiom-mini")


if __name__ == "__main__":
    main()
