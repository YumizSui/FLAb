#!/usr/bin/env python3
"""Test dual GPU approach: GPU0 for most processing, GPU1 for diffusion only."""

import torch
import gc
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from predict_structures_chai import create_fasta

import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

def get_gpu_memory_gb(device_id=0):
    """Get current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / 1024**3
    return 0

def print_memory(stage, device_id=0):
    """Print current memory usage."""
    allocated = get_gpu_memory_gb(device_id)
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    print(f"[GPU{device_id}] {stage:50s} | Allocated: {allocated:6.2f} GB | Reserved: {reserved:6.2f} GB")

def main():
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        print(f"ERROR: Need 2 GPUs, but only {torch.cuda.device_count()} available")
        sys.exit(1)

    print(f"Found {torch.cuda.device_count()} GPUs")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")

    # Load test data
    csv_path = Path("data/thermostability/garbinski2023_tm1.csv")
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    heavy_seq = row["heavy"]
    light_seq = row["light"]

    output_dir = Path("test_dual_gpu")
    output_dir.mkdir(exist_ok=True)
    fasta_path = output_dir / "test.fasta"
    create_fasta(heavy_seq, light_seq, fasta_path)

    num_samples = 1
    gpu0 = torch.device("cuda:0")
    gpu1 = torch.device("cuda:1")

    print("\n" + "=" * 100)
    print("DUAL GPU TEST: GPU0 for preprocessing/postprocessing, GPU1 for diffusion")
    print("=" * 100)

    # Import chai_lab modules
    from chai_lab.chai1 import (
        load_exported,
        make_all_atom_feature_context,
        feature_factory,
        raise_if_too_many_tokens,
        raise_if_too_many_templates,
        raise_if_msa_too_deep,
        DiffusionConfig,
        _bin_centers,
    )
    from chai_lab.data.collate.collate import Collate
    from chai_lab.data.features.generators.token_bond import TokenBondRestraint
    from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
    from chai_lab.model.utils import center_random_augmentation
    from chai_lab.utils.tensor_utils import move_data_to_device, und_self
    from einops import rearrange, repeat

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)

    print("\n### PHASE 1: GPU0 - Load models and preprocessing ###\n")

    # Load models on GPU0 (except diffusion_module)
    print("Loading models on GPU0...")
    models_gpu0 = {
        "feature_embedding": load_exported("feature_embedding.pt", gpu0),
        "bond_loss_input_proj": load_exported("bond_loss_input_proj.pt", gpu0),
        "token_input_embedder": load_exported("token_embedder.pt", gpu0),
        "trunk": load_exported("trunk.pt", gpu0),
        "confidence_head": load_exported("confidence_head.pt", gpu0),
    }
    print_memory("After loading 5 models (no diffusion)", device_id=0)

    # Feature context (includes ESM on GPU0)
    print("\nCreating feature context on GPU0...")
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=output_dir,
        use_esm_embeddings=True,
        use_msa_server=False,
        esm_device=gpu0,
    )
    print_memory("After feature context + ESM", device_id=0)

    # Validate
    raise_if_too_many_tokens(feature_context.structure_context.num_tokens)
    raise_if_too_many_templates(feature_context.template_context.num_templates)
    raise_if_msa_too_deep(feature_context.msa_context.depth)

    # Collate batch
    collator = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
    feature_contexts = [feature_context]
    batch_size = 1
    batch = collator(feature_contexts)

    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]

    # Feature embedding on GPU0
    print("\nRunning feature embedding on GPU0...")
    _, _, model_size = inputs["msa_mask"].shape
    embedded_features = models_gpu0["feature_embedding"].forward(
        crop_size=model_size,
        move_to_device=gpu0,
        return_on_cpu=False,  # Keep on GPU0
        **features,
    )
    print_memory("After feature embedding", device_id=0)

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features["TOKEN_PAIR"].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features["ATOM"].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    # Bond features
    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    trunk_bond_feat, structure_bond_feat = models_gpu0["bond_loss_input_proj"].forward(
        return_on_cpu=False,
        move_to_device=gpu0,
        crop_size=model_size,
        input=bond_ft,
    ).chunk(2, dim=-1)
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat
    token_pair_structure_input_feats = token_pair_structure_input_feats + structure_bond_feat

    # Token embedder on GPU0
    print("\nRunning token embedder on GPU0...")
    token_input_embedder_outputs = models_gpu0["token_input_embedder"].forward(
        return_on_cpu=False,
        move_to_device=gpu0,
        token_single_input_feats=token_single_input_feats,
        token_pair_input_feats=token_pair_input_feats,
        atom_single_input_feats=atom_single_input_feats,
        block_atom_pair_feat=block_atom_pair_input_feats,
        block_atom_pair_mask=inputs["block_atom_pair_mask"],
        block_indices_h=inputs["block_atom_pair_q_idces"],
        block_indices_w=inputs["block_atom_pair_kv_idces"],
        atom_single_mask=inputs["atom_exists_mask"],
        atom_token_indices=inputs["atom_token_index"].long(),
        crop_size=model_size,
    )
    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = token_input_embedder_outputs
    print_memory("After token embedder", device_id=0)

    # Trunk on GPU0
    print("\nRunning trunk (3 recycles) on GPU0...")
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    template_input_masks = und_self(inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2")

    for recycle_idx in range(3):
        (token_single_trunk_repr, token_pair_trunk_repr) = models_gpu0["trunk"].forward(
            move_to_device=gpu0,
            token_single_trunk_initial_repr=token_single_initial_repr,
            token_pair_trunk_initial_repr=token_pair_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            msa_input_feats=msa_input_feats,
            msa_mask=inputs["msa_mask"],
            template_input_feats=template_input_feats,
            template_input_masks=template_input_masks,
            token_single_mask=token_single_mask,
            token_pair_mask=token_pair_mask,
            crop_size=model_size,
        )
        print_memory(f"After trunk recycle {recycle_idx+1}/3", device_id=0)

    print("\n### PHASE 2: Prepare diffusion inputs and move to GPU1 ###\n")

    # Prepare static diffusion inputs on GPU0
    atom_single_mask = inputs["atom_exists_mask"].to(gpu0)
    static_diffusion_inputs_gpu0 = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=inputs["block_atom_pair_mask"],
        token_single_mask=token_single_mask,
        block_indices_h=inputs["block_atom_pair_q_idces"],
        block_indices_w=inputs["block_atom_pair_kv_idces"],
        atom_token_indices=inputs["atom_token_index"].long(),
    )
    print_memory("After preparing diffusion inputs on GPU0", device_id=0)

    # Move diffusion inputs to CPU, then to GPU1
    print("\nMoving diffusion inputs: GPU0 -> CPU -> GPU1...")
    static_diffusion_inputs_cpu = move_data_to_device(static_diffusion_inputs_gpu0, device=torch.device("cpu"))
    del static_diffusion_inputs_gpu0
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After moving to CPU (GPU0 cleared)", device_id=0)

    static_diffusion_inputs_gpu1 = move_data_to_device(static_diffusion_inputs_cpu, device=gpu1)
    del static_diffusion_inputs_cpu
    print_memory("Diffusion inputs on GPU1", device_id=1)

    # Load diffusion module on GPU1
    print("\nLoading diffusion_module on GPU1...")
    diffusion_module_gpu1 = load_exported("diffusion_module.pt", gpu1)
    print_memory("After loading diffusion_module", device_id=1)

    # Initialize atom positions on GPU1
    n_atoms = atom_single_mask.shape[-1]
    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=DiffusionConfig.S_tmin,
        p=7.0,
        sigma_data=DiffusionConfig.sigma_data,
    )
    noise_schedule = inference_noise_schedule.get_schedule(device=gpu1, num_timesteps=200)

    atom_pos = torch.randn(batch_size, num_samples, n_atoms, 3, device=gpu1) * noise_schedule[0]
    atom_pos = rearrange(atom_pos, "b s ... -> (b s) ...")
    print_memory("After initializing atom_pos", device_id=1)

    # Diffusion on GPU1
    print("\nRunning diffusion (200 steps) on GPU1...")

    def _denoise(atom_pos, sigma, ds):
        atom_noised_coords = rearrange(atom_pos, "(b ds) ... -> b ds ...", ds=ds).contiguous()
        noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=ds)
        return diffusion_module_gpu1.forward(
            atom_noised_coords=atom_noised_coords.float(),
            noise_sigma=noise_sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs_gpu1,
        )

    sample_steps = [0, 50, 100, 150, 199]
    for i in range(len(noise_schedule) - 1):
        sigma = noise_schedule[i]
        sigma_next = noise_schedule[i + 1]

        if DiffusionConfig.S_churn > 0 and sigma < DiffusionConfig.S_tmax and sigma > DiffusionConfig.S_tmin:
            gamma = min(DiffusionConfig.S_churn / len(noise_schedule), 2**0.5 - 1)
            sigma_hat = sigma * (1 + gamma)
            noise = torch.randn_like(atom_pos) * DiffusionConfig.S_noise
            atom_pos = atom_pos + noise * (sigma_hat**2 - sigma**2).sqrt()
            sigma = sigma_hat
        else:
            sigma_hat = sigma

        denoised = _denoise(atom_pos, sigma_hat.expand(batch_size), num_samples)
        d = (atom_pos - denoised) / sigma_hat
        atom_pos = atom_pos + d * (sigma_next - sigma_hat)

        if DiffusionConfig.second_order and sigma_next > 0:
            denoised_2 = _denoise(atom_pos, sigma_next.expand(batch_size), num_samples)
            d_2 = (atom_pos - denoised_2) / sigma_next
            atom_pos = atom_pos + (d_2 - d) * (sigma_next - sigma_hat) / 2

        if i in sample_steps:
            print_memory(f"Diffusion step {i}/199", device_id=1)

    atom_pos = rearrange(atom_pos, "(b s) ... -> b s ...", b=batch_size)
    print_memory("After diffusion complete", device_id=1)

    # Move atom_pos to CPU, clear GPU1
    print("\nMoving atom_pos: GPU1 -> CPU...")
    atom_pos_cpu = atom_pos.cpu()
    atom_single_mask_cpu = static_diffusion_inputs_gpu1["atom_single_mask"].cpu()

    del atom_pos, static_diffusion_inputs_gpu1, diffusion_module_gpu1
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After clearing GPU1", device_id=1)

    # Apply augmentation on CPU
    atom_pos_cpu = center_random_augmentation(
        atom_pos_cpu, atom_single_mask_cpu.unsqueeze(1).expand(-1, num_samples, -1)
    )

    print("\n### PHASE 3: GPU0 - Confidence head ###\n")

    # Move atom_pos back to GPU0
    atom_pos_gpu0 = atom_pos_cpu.to(gpu0)
    atom_single_mask_gpu0 = atom_single_mask_cpu.to(gpu0)

    print_memory("After moving atom_pos to GPU0", device_id=0)

    # Run confidence head on GPU0
    print("\nRunning confidence head on GPU0...")
    pae_logits, pde_logits, plddt_logits = models_gpu0["confidence_head"].forward(
        move_to_device=gpu0,
        token_single_input_repr=token_single_structure_input.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_repr=atom_single_structure_input_feats.float(),
        atom_single_mask=atom_single_mask_gpu0,
        atom_token_indices=inputs["atom_token_index"].long(),
        token_single_mask=token_single_mask,
        atom_coords=atom_pos_gpu0.float(),
        crop_size=model_size,
    )
    print_memory("After confidence head", device_id=0)

    # Move to CPU
    atom_pos_cpu = atom_pos_gpu0.cpu()
    pae_bins = _bin_centers(0, 32, 64)
    pde_bins = _bin_centers(0, 32, 64)
    plddt_bins = _bin_centers(0, 1, 50)

    pae = (pae_logits.softmax(dim=-1) * pae_bins.to(pae_logits.device)).sum(dim=-1).cpu()
    pde = (pde_logits.softmax(dim=-1) * pde_bins.to(pde_logits.device)).sum(dim=-1).cpu()
    plddt = (plddt_logits.softmax(dim=-1) * plddt_bins.to(plddt_logits.device)).sum(dim=-1).cpu()

    print_memory("After moving results to CPU", device_id=0)

    print("\n" + "=" * 100)
    print("PEAK MEMORY STATISTICS")
    print("=" * 100)
    print(f"GPU0 Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU0 Peak reserved:  {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
    print(f"GPU1 Peak allocated: {torch.cuda.max_memory_allocated(1) / 1024**3:.2f} GB")
    print(f"GPU1 Peak reserved:  {torch.cuda.max_memory_reserved(1) / 1024**3:.2f} GB")
    print("=" * 100)
    print("\nSUCCESS! Dual GPU approach works.")

if __name__ == "__main__":
    main()
