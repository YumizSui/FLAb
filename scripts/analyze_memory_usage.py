#!/usr/bin/env python3
"""Analyze GPU memory usage during Chai-1 inference."""

import torch
import gc
import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from predict_structures_chai import create_fasta

def get_gpu_memory_gb():
    """Get current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def print_memory(stage):
    """Print current memory usage."""
    allocated = get_gpu_memory_gb()
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{stage:50s} | Allocated: {allocated:6.2f} GB | Reserved: {reserved:6.2f} GB")

def main():
    # Load test data
    csv_path = Path("data/thermostability/garbinski2023_tm1.csv")
    df = pd.read_csv(csv_path)

    # Use first row
    row = df.iloc[0]
    heavy_seq = row["heavy"]
    light_seq = row["light"]

    # Create FASTA
    output_dir = Path("test_memory_analysis")
    output_dir.mkdir(exist_ok=True)
    fasta_path = output_dir / "test.fasta"
    create_fasta(heavy_seq, light_seq, fasta_path)

    device = "cuda"
    num_samples = 1

    print("=" * 100)
    print("GPU Memory Usage During Inference")
    print("=" * 100)

    torch.cuda.reset_peak_memory_stats()
    print_memory("1. Baseline")

    # Import after baseline measurement
    from chai_lab.chai1 import (
        load_exported,
        make_all_atom_feature_context,
        feature_factory,
        raise_if_too_many_tokens,
        raise_if_too_many_templates,
        raise_if_msa_too_deep,
        DiffusionConfig,
        StructureCandidates,
        _bin_centers,
    )
    from chai_lab.data.collate.collate import Collate
    from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
    from chai_lab.data.features.generators.token_bond import TokenBondRestraint
    from chai_lab.data.dataset.msas.utils import subsample_and_reorder_msa_feats_n_mask
    from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
    from chai_lab.model.utils import center_random_augmentation
    from chai_lab.ranking.frames import get_frames_and_mask
    from chai_lab.ranking.rank import get_scores, rank
    from chai_lab.data.io.cif_utils import save_to_cif
    from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self
    from einops import rearrange, repeat

    print_memory("2. After imports")

    # Load models
    torch_device = torch.device(device)
    print("\nLoading models...")
    models = {
        "feature_embedding": load_exported("feature_embedding.pt", torch_device),
        "bond_loss_input_proj": load_exported("bond_loss_input_proj.pt", torch_device),
        "token_input_embedder": load_exported("token_embedder.pt", torch_device),
        "trunk": load_exported("trunk.pt", torch_device),
        "diffusion_module": load_exported("diffusion_module.pt", torch_device),
        "confidence_head": load_exported("confidence_head.pt", torch_device),
    }
    print_memory("3. After loading models")

    # Create feature context
    print("\nCreating feature context...")
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=output_dir,
        use_esm_embeddings=True,
        use_msa_server=False,
        esm_device=torch_device,
    )
    print_memory("4. After feature context (includes ESM embeddings)")

    # Validate
    n_actual_tokens = feature_context.structure_context.num_tokens
    raise_if_too_many_tokens(n_actual_tokens)
    raise_if_too_many_templates(feature_context.template_context.num_templates)
    raise_if_msa_too_deep(feature_context.msa_context.depth)

    # Prepare batch
    print("\nPreparing batch...")
    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )

    feature_contexts = [feature_context]
    batch_size = len(feature_contexts)
    batch = collator(feature_contexts)

    low_memory = True
    if not low_memory:
        batch = move_data_to_device(batch, device=torch_device)

    print_memory("5. After batch collation")

    # Get features
    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]
    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2")
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

    _, _, model_size = msa_mask.shape

    feature_embedding = models["feature_embedding"]
    bond_loss_input_proj = models["bond_loss_input_proj"]
    token_input_embedder = models["token_input_embedder"]
    trunk = models["trunk"]
    diffusion_module = models["diffusion_module"]
    confidence_head = models["confidence_head"]

    # Feature embedding
    print("\nRunning feature embedding...")
    embedded_features = feature_embedding.forward(
        crop_size=model_size,
        move_to_device=torch_device,
        return_on_cpu=low_memory,
        **features,
    )
    print_memory("6. After feature embedding")

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features["TOKEN_PAIR"].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features["ATOM"].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    # Bond features
    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
        return_on_cpu=low_memory,
        move_to_device=torch_device,
        crop_size=model_size,
        input=bond_ft,
    ).chunk(2, dim=-1)
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat
    token_pair_structure_input_feats = token_pair_structure_input_feats + structure_bond_feat

    print_memory("7. After bond features")

    # Token input embedder
    print("\nRunning token input embedder...")
    token_input_embedder_outputs = token_input_embedder.forward(
        return_on_cpu=low_memory,
        move_to_device=torch_device,
        token_single_input_feats=token_single_input_feats.to(torch.bfloat16),
        token_pair_input_feats=token_pair_input_feats.to(torch.bfloat16),
        atom_single_input_feats=atom_single_input_feats.to(torch.bfloat16),
        block_atom_pair_feat=block_atom_pair_input_feats.to(torch.bfloat16),
        block_atom_pair_mask=block_atom_pair_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_single_mask=atom_single_mask,
        atom_token_indices=atom_token_indices,
        crop_size=model_size,
    )
    print_memory("8. After token input embedder")

    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = token_input_embedder_outputs

    # Trunk recycles
    print("\nRunning trunk recycles...")
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    num_trunk_recycles = 3

    for recycle_idx in range(num_trunk_recycles):
        (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
            move_to_device=torch_device,
            token_single_trunk_initial_repr=token_single_initial_repr.to(torch.bfloat16),
            token_pair_trunk_initial_repr=token_pair_initial_repr.to(torch.bfloat16),
            token_single_trunk_repr=token_single_trunk_repr.to(torch.bfloat16),
            token_pair_trunk_repr=token_pair_trunk_repr.to(torch.bfloat16),
            msa_input_feats=msa_input_feats.to(torch.bfloat16),
            msa_mask=msa_mask,
            template_input_feats=template_input_feats.to(torch.bfloat16),
            template_input_masks=template_input_masks,
            token_single_mask=token_single_mask,
            token_pair_mask=token_pair_mask,
            crop_size=model_size,
        )
        print_memory(f"9.{recycle_idx+1}. After trunk recycle {recycle_idx+1}/{num_trunk_recycles}")

    torch.cuda.empty_cache()
    print_memory("10. After trunk (and cache clear)")

    # Diffusion
    print("\nPreparing diffusion...")
    atom_single_mask = atom_single_mask.to(torch_device)

    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(static_diffusion_inputs, device=torch_device)

    print_memory("11. After preparing diffusion inputs")

    def _denoise(atom_pos, sigma, ds):
        atom_noised_coords = rearrange(atom_pos, "(b ds) ... -> b ds ...", ds=ds).contiguous()
        noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=ds)
        return diffusion_module.forward(
            atom_noised_coords=atom_noised_coords.float(),
            noise_sigma=noise_sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs,
        )

    num_diffn_timesteps = 200
    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=DiffusionConfig.S_tmin,
        p=7.0,
        sigma_data=DiffusionConfig.sigma_data,
    )
    noise_schedule = inference_noise_schedule.get_schedule(
        device=torch_device, num_timesteps=num_diffn_timesteps
    )

    n_atoms = atom_single_mask.shape[-1]
    atom_pos = (
        torch.randn(batch_size, num_samples, n_atoms, 3, device=torch_device)
        * noise_schedule[0]
    )
    atom_pos = rearrange(atom_pos, "b s ... -> (b s) ...")

    print(f"\nRunning diffusion ({num_diffn_timesteps} steps)...")

    # Sample a few timesteps to check memory
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
            print_memory(f"12.{sample_steps.index(i)+1}. Diffusion step {i}/{num_diffn_timesteps-1}")

    atom_pos = rearrange(atom_pos, "(b s) ... -> b s ...", b=batch_size)
    atom_pos = center_random_augmentation(
        atom_pos, atom_single_mask.unsqueeze(1).expand(-1, num_samples, -1)
    )

    print_memory("13. After diffusion complete")

    # Confidence head
    print("\nRunning confidence head...")
    pae_logits, pde_logits, plddt_logits = confidence_head.forward(
        move_to_device=torch_device,
        token_single_input_repr=token_single_structure_input.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_repr=atom_single_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_token_indices=atom_token_indices,
        token_single_mask=token_single_mask,
        atom_coords=atom_pos.float(),
        crop_size=model_size,
    )
    print_memory("14. After confidence head")

    # Move to CPU
    atom_pos = atom_pos.cpu()
    pae_bins = _bin_centers(0, 32, 64)
    pde_bins = _bin_centers(0, 32, 64)
    plddt_bins = _bin_centers(0, 1, 50)

    pae = (pae_logits.softmax(dim=-1) * pae_bins.to(pae_logits.device)).sum(dim=-1).cpu()
    pde = (pde_logits.softmax(dim=-1) * pde_bins.to(pde_logits.device)).sum(dim=-1).cpu()
    plddt = (plddt_logits.softmax(dim=-1) * plddt_bins.to(plddt_logits.device)).sum(dim=-1).cpu()

    print_memory("15. After moving to CPU")

    # Cleanup
    print("\nCleaning up...")
    del batch, features, inputs, embedded_features
    del token_single_input_feats, token_pair_input_feats, atom_single_input_feats
    del block_atom_pair_input_feats, template_input_feats, msa_input_feats
    del token_pair_structure_input_feats, atom_single_structure_input_feats
    del block_atom_pair_structure_input_feats
    del token_single_initial_repr, token_single_structure_input, token_pair_initial_repr
    del token_single_trunk_repr, token_pair_trunk_repr
    del static_diffusion_inputs, atom_pos
    del pae_logits, pde_logits, plddt_logits
    del feature_context, collator
    gc.collect()
    torch.cuda.empty_cache()

    print_memory("16. After cleanup")

    print("\n" + "=" * 100)
    print("PEAK MEMORY STATISTICS")
    print("=" * 100)
    print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print("=" * 100)

if __name__ == "__main__":
    main()
