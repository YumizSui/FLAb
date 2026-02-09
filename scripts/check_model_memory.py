#!/usr/bin/env python3
"""Check GPU memory usage for each Chai-1 model component."""

import torch
import gc
from chai_lab.chai1 import load_exported

def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def main():
    device = torch.device("cuda:0")

    print("=" * 80)
    print("GPU Memory Usage per Model Component")
    print("=" * 80)

    # Baseline
    torch.cuda.reset_peak_memory_stats()
    baseline = get_gpu_memory_mb()
    print(f"\nBaseline (no models loaded): {baseline:.2f} MB")

    models = {}
    model_names = [
        "bond_loss_input_proj.pt",
        "feature_embedding.pt",
        "token_embedder.pt",
        "confidence_head.pt",
        "diffusion_module.pt",
        "trunk.pt",
    ]

    cumulative = baseline

    for model_name in model_names:
        print(f"\nLoading {model_name}...")

        # Load model
        models[model_name] = load_exported(model_name, device)

        # Measure memory
        after_load = get_gpu_memory_mb()
        delta = after_load - cumulative
        cumulative = after_load

        print(f"  Individual:  {delta:>8.2f} MB")
        print(f"  Cumulative:  {cumulative:>8.2f} MB")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total GPU memory for all 6 models: {cumulative - baseline:.2f} MB")
    print(f"                                  = {(cumulative - baseline) / 1024:.2f} GB")
    print(f"\nPeak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"                      = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("=" * 80)

    # Cleanup
    del models
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nAfter cleanup: {get_gpu_memory_mb():.2f} MB")

if __name__ == "__main__":
    main()
