#!/usr/bin/env python
"""
Compare PLL mode vs Confidence mode for sablm.
Also verify esmif-like computation.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats

# Add PoET-2 paths
POET2_ROOT = Path("/home/kfurui/workspace/AbEval/PoET-2")
sys.path.insert(0, str(POET2_ROOT / "src"))
sys.path.insert(0, str(POET2_ROOT / "scripts"))

from poet_2.training_ab_enc.preprocess_ab import load_pdb_structure
from run_ab_enc_inference import load_encoder_only_model, encode_sequence


def compute_confidence_score(model, alphabet, s3di_alphabet, sequence, atomx, plddt, s3di_tokens, device):
    """Compute confidence score (single inference, no masking) - like esmif."""
    device_obj = torch.device(device)

    result = encode_sequence(
        model=model,
        sequence=sequence,
        alphabet=alphabet,
        s3di_alphabet=s3di_alphabet,
        device=device_obj,
        atomx=atomx,
        plddt=plddt,
        s3di=s3di_tokens,
    )

    mlm_logits = result["mlm_logits"][0]  # (L, vocab)
    log_probs = F.log_softmax(mlm_logits, dim=-1).cpu().numpy()

    # Get log prob for each position's true amino acid
    total_log_prob = 0.0
    count = 0
    for i, char in enumerate(sequence):
        if char == '|':
            continue
        token_pos = i + 1  # +1 for start token
        true_aa_idx = alphabet.encode(char.encode())[0]
        total_log_prob += log_probs[token_pos, true_aa_idx]
        count += 1

    avg_ll = total_log_prob / count
    perplexity = np.exp(-avg_ll)
    return perplexity


def main():
    # Load model
    checkpoint_dir = POET2_ROOT / "checkpoints/var2/var2_str-0.5_nofocal_noaug_lora"
    best_checkpoint_file = checkpoint_dir / "best_checkpoint.json"

    import json
    with open(best_checkpoint_file) as f:
        best_info = json.load(f)
    checkpoint_path = POET2_ROOT / best_info["model_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model from {checkpoint_path}...")
    model, alphabet, s3di_alphabet = load_encoder_only_model(
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    print(f"Model loaded on {device}")

    # Load data
    csv_path = "/home/kfurui/workspace/FLAb/data/thermostability/jain2017biophysical_Tm.csv"
    structure_dir = "/home/kfurui/workspace/FLAb/structure/tm/Wittrup2017_CST_Tm"
    df = pd.read_csv(csv_path)

    results = []

    for idx in tqdm(range(len(df)), desc="Computing scores"):
        row = df.iloc[idx]
        heavy_seq = row['heavy']
        light_seq = row['light']
        fitness = row['fitness'] if 'fitness' in row else row.get('Tm', None)
        sequence = f"{heavy_seq}|{light_seq}"

        pdb_path = f"{structure_dir}/Wittrup2017_CST_Tm_{idx}.pdb"
        expected_seq = heavy_seq + light_seq

        # Load structure
        atomx, plddt, s3di_tokens = load_pdb_structure(
            Path(pdb_path),
            expected_sequence=expected_seq,
            heavy_chain="H",
            light_chain="L",
        )

        # Add separator padding
        if atomx is not None and plddt is not None:
            vh_len = len(heavy_seq)
            atomx_padded = np.full((len(expected_seq) + 1, 3, 3), np.nan, dtype=np.float32)
            plddt_padded = np.full(len(expected_seq) + 1, np.nan, dtype=np.float32)
            atomx_padded[:vh_len] = atomx[:vh_len]
            atomx_padded[vh_len + 1:] = atomx[vh_len:]
            plddt_padded[:vh_len] = plddt[:vh_len]
            plddt_padded[vh_len + 1:] = plddt[vh_len:]
            atomx = atomx_padded
            plddt = plddt_padded

        if s3di_tokens is not None:
            vh_len = len(heavy_seq)
            s3di_tokens = s3di_tokens[:vh_len] + 'X' + s3di_tokens[vh_len:]

        # Compute confidence scores (single inference, no masking)
        conf_str = compute_confidence_score(model, alphabet, s3di_alphabet, sequence,
                                             atomx, plddt, s3di_tokens, device)
        conf_nostr = compute_confidence_score(model, alphabet, s3di_alphabet, sequence,
                                               None, None, None, device)

        results.append({
            'sample_id': idx,
            'fitness': fitness,
            'conf_str': conf_str,
            'conf_nostr': conf_nostr,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Load existing PLL results for comparison
    pll_str = pd.read_csv('test_full_pyrosetta/sablm_str_ppl.csv')
    pll_nostr = pd.read_csv('test_full_pyrosetta/sablm_nostr_ppl.csv')

    results_df = results_df.merge(
        pll_str[['sample_id', 'perplexity']].rename(columns={'perplexity': 'pll_str'}),
        on='sample_id'
    )
    results_df = results_df.merge(
        pll_nostr[['sample_id', 'perplexity']].rename(columns={'perplexity': 'pll_nostr'}),
        on='sample_id'
    )

    # Save results
    results_df.to_csv('test_full_pyrosetta/sablm_confidence_comparison.csv', index=False)
    print(f"\nResults saved to test_full_pyrosetta/sablm_confidence_comparison.csv")

    # Compute correlations with fitness
    print(f"\n{'='*60}")
    print("Correlations with Fitness (Tm)")
    print(f"{'='*60}")

    for col in ['conf_str', 'conf_nostr', 'pll_str', 'pll_nostr']:
        rho, p = stats.spearmanr(results_df[col], results_df['fitness'])
        print(f"  {col:15s}: ρ={rho:+.3f}, p={p:.3f}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    for col in ['conf_str', 'conf_nostr', 'pll_str', 'pll_nostr']:
        print(f"  {col:15s}: mean={results_df[col].mean():.3f}, std={results_df[col].std():.3f}")

    # Compare conf vs pll
    print(f"\n{'='*60}")
    print("Comparison: Confidence vs PLL")
    print(f"{'='*60}")
    rho_str, _ = stats.spearmanr(results_df['conf_str'], results_df['pll_str'])
    rho_nostr, _ = stats.spearmanr(results_df['conf_nostr'], results_df['pll_nostr'])
    print(f"  conf_str vs pll_str:     ρ={rho_str:.3f}")
    print(f"  conf_nostr vs pll_nostr: ρ={rho_nostr:.3f}")


if __name__ == "__main__":
    main()
