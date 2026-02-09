"""Extract embeddings from antibody language models for few-shot learning.

Supports 6 models:
  - antiberty (single-chain, hidden=512, concat H+L → 1024)
  - esm2      (single-chain, hidden=1280, concat H+L → 2560)
  - ism       (single-chain, hidden=1280, concat H+L → 2560)
  - ablang2   (paired model, hidden=480, whole mean pool → 480)
  - sablm_str (paired + structure, hidden=1024, whole mean pool → 1024)
  - sablm_nostr (paired, no structure, hidden=1024, whole mean pool → 1024)

Usage:
    python scripts/extract_embeddings.py \
        --csv-path data/thermostability/jain2017biophysical_Tm.csv \
        --model antiberty --device cuda:0
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

FLAB_DIR = Path(__file__).resolve().parent.parent
MODELS = ["antiberty", "esm2", "ism", "ablang2", "sablm_str", "sablm_nostr"]


def get_args():
    parser = argparse.ArgumentParser(description="Extract embeddings for few-shot learning")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to dataset CSV (columns: heavy, light, fitness)")
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: embs/{model}/{category}/{dataset}/)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--structure-dir", type=str, default=None,
                        help="Structure directory (for sablm_str)")
    parser.add_argument("--checkpoint", type=str, default="poet2_ab_enc_pretrain",
                        help="PoET-2 checkpoint name (for sablm)")
    return parser.parse_args()


def infer_output_dir(csv_path: str, model: str) -> str:
    """Infer output directory from CSV path and model name."""
    parts = Path(csv_path).parts
    # e.g. data/thermostability/jain2017biophysical_Tm.csv
    category = parts[-2]
    dataset_name = Path(csv_path).stem
    return str(FLAB_DIR / "embs" / model / category / dataset_name)


# ============================================================================
# AntiBERTy embedding (single-chain, hidden=512)
# ============================================================================

def extract_antiberty(df, device, batch_size):
    """Extract AntiBERTy embeddings: H and L independently → mean pool → concat."""
    from antiberty import AntiBERTyRunner

    runner = AntiBERTyRunner()
    device_obj = torch.device(device)
    runner.device = device_obj
    runner.model = runner.model.to(device_obj)

    def embed_chain(sequences, bs):
        """Embed sequences, remove CLS/SEP, mean pool → [N, 512]."""
        all_embs = []
        for i in tqdm(range(0, len(sequences), bs), desc="AntiBERTy embed"):
            batch_seqs = sequences[i:i+bs]
            # embed returns list of tensors, each [L_with_special, 512] (attention_mask filtered)
            emb_list = runner.embed(batch_seqs, hidden_layer=-1)
            for emb in emb_list:
                # emb shape: [L_with_special, 512] - includes CLS (pos 0) and SEP (pos -1)
                # Remove CLS and SEP
                emb_no_special = emb[1:-1]  # [L, 512]
                pooled = emb_no_special.mean(dim=0)  # [512]
                all_embs.append(pooled.cpu())
        return torch.stack(all_embs)  # [N, 512]

    heavy_seqs = df["heavy"].tolist()
    heavy_embs = embed_chain(heavy_seqs, batch_size)

    has_light = "light" in df.columns and df["light"].notna().all()
    if has_light:
        light_seqs = df["light"].tolist()
        light_embs = embed_chain(light_seqs, batch_size)
        embeddings = torch.cat([heavy_embs, light_embs], dim=1)  # [N, 1024]
    else:
        embeddings = heavy_embs  # [N, 512]

    return embeddings


# ============================================================================
# ESM2 / ISM embedding (single-chain, hidden=1280)
# ============================================================================

def extract_esm2(df, device, batch_size, model_name="esm2"):
    """Extract ESM2/ISM embeddings: H and L independently → mean pool → concat."""
    import esm as esm_module

    device_obj = torch.device(device)

    if model_name == "ism":
        esm_model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
        ism_path = FLAB_DIR / "envs" / "models" / "ism_t33_650M_uc30pdb" / "checkpoint.pth"
        ckpt = torch.load(str(ism_path), map_location=device_obj)
        esm_model.load_state_dict(ckpt)
    else:
        esm_model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()

    esm_model = esm_model.to(device_obj)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = 33  # last layer

    def embed_chain(sequences, bs):
        """Embed sequences, remove BOS/EOS, mean pool → [N, 1280]."""
        all_embs = []
        for i in tqdm(range(0, len(sequences), bs), desc=f"{model_name} embed"):
            batch_seqs = sequences[i:i+bs]
            data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device_obj)

            with torch.no_grad():
                results = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)

            representations = results["representations"][repr_layer]  # [B, L+2, 1280]

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                # Remove BOS (pos 0) and EOS (pos seq_len+1)
                emb = representations[j, 1:seq_len+1, :]  # [seq_len, 1280]
                pooled = emb.mean(dim=0)
                all_embs.append(pooled.cpu())

        return torch.stack(all_embs)  # [N, 1280]

    heavy_seqs = df["heavy"].tolist()
    heavy_embs = embed_chain(heavy_seqs, batch_size)

    has_light = "light" in df.columns and df["light"].notna().all()
    if has_light:
        light_seqs = df["light"].tolist()
        light_embs = embed_chain(light_seqs, batch_size)
        embeddings = torch.cat([heavy_embs, light_embs], dim=1)  # [N, 2560]
    else:
        embeddings = heavy_embs  # [N, 1280]

    return embeddings


# ============================================================================
# AbLang2 embedding (paired model, hidden=480)
# ============================================================================

def extract_ablang2(df, device, batch_size):
    """Extract AbLang2 embeddings: paired input → mean pool (padding excluded) → [N, 480]."""
    import ablang2

    ablang_model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False,
                                       ncpu=1, device=device)
    pad_token = ablang_model.tokenizer.pad_token

    has_light = "light" in df.columns and df["light"].notna().all()
    all_embs = []

    for i in tqdm(range(0, len(df), batch_size), desc="AbLang2 embed"):
        batch_df = df.iloc[i:i+batch_size]
        seqs = []
        for _, row in batch_df.iterrows():
            h = row["heavy"]
            if has_light:
                l = row["light"]
                seqs.append(f"{h}|{l}")
            else:
                seqs.append(f"{h}|")
        # Tokenize: w_extra_tkns=False → no <> wrapping
        tokens = ablang_model.tokenizer(seqs, pad=True, w_extra_tkns=False, device=device)

        with torch.no_grad():
            hidden_states = ablang_model.AbLang.AbRep(tokens).last_hidden_states
            # hidden_states: [B, L, 480]

        # Mean pool excluding padding
        for j in range(hidden_states.size(0)):
            seq_len = len(seqs[j])  # actual sequence length (including |)
            emb = hidden_states[j, :seq_len, :]  # [seq_len, 480]
            pooled = emb.mean(dim=0)  # [480]
            all_embs.append(pooled.cpu())

    return torch.stack(all_embs)  # [N, 480]


# ============================================================================
# SABLm embedding (paired, hidden=1024)
# ============================================================================

def extract_sablm(df, device, batch_size, use_structure, structure_dir, checkpoint):
    """Extract SABLm embeddings using encoder_outputs → mean pool → [N, 1024]."""
    import numpy as np

    # Add PoET-2 paths
    POET2_ROOT = FLAB_DIR / "PoET-2"
    sys.path.insert(0, str(POET2_ROOT / "src"))
    sys.path.insert(0, str(POET2_ROOT / "scripts"))

    from run_ab_enc_inference import load_encoder_only_model, tokenize_sequence

    # Find checkpoint
    checkpoint_dir = POET2_ROOT / "checkpoints" / checkpoint
    best_checkpoint_file = checkpoint_dir / "best_checkpoint.json"

    import json
    if best_checkpoint_file.exists():
        with open(best_checkpoint_file) as f:
            best_info = json.load(f)
        checkpoint_path = POET2_ROOT / best_info["model_path"]
    else:
        model_files = sorted(checkpoint_dir.glob("model_step_*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {checkpoint_dir}")
        checkpoint_path = model_files[-1]

    device_str = device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    dtype = torch.float16 if "cuda" in device_str else torch.float32
    device_obj = torch.device(device_str)

    model, alphabet, s3di_alphabet = load_encoder_only_model(
        checkpoint_path=checkpoint_path,
        device=device_str,
        dtype=dtype,
    )

    has_light = "light" in df.columns and df["light"].notna().all()
    all_embs = []

    for idx in tqdm(range(len(df)), desc=f"SABLm {'str' if use_structure else 'nostr'} embed"):
        row = df.iloc[idx]
        heavy_seq = row["heavy"]
        light_seq = row["light"] if has_light else ""
        sequence = f"{heavy_seq}|{light_seq}" if light_seq else heavy_seq

        # Load structure if needed
        atomx, plddt, s3di_tokens = None, None, None
        if use_structure and structure_dir:
            try:
                from poet_2.training_ab_enc.preprocess_ab import load_pdb_structure

                # Find PDB file corresponding to this sample
                struct_path = Path(structure_dir)
                dataset_name = struct_path.name  # e.g., "Wittrup2017_CST_Tm"

                # Try pattern: {dataset_name}_{idx}.pdb
                pdb_file = struct_path / f"{dataset_name}_{idx}.pdb"

                if not pdb_file.exists():
                    warnings.warn(f"PDB file not found: {pdb_file}, skipping structure for idx={idx}")
                else:
                    expected_seq = heavy_seq + light_seq
                    atomx, plddt, s3di_tokens = load_pdb_structure(
                        pdb_file,
                        expected_sequence=expected_seq,
                        heavy_chain="H",
                        light_chain="L",
                    )
                    # Pad for separator
                    if atomx is not None and plddt is not None and light_seq:
                        vh_len = len(heavy_seq)
                        atomx_padded = np.full((len(expected_seq) + 1, 3, 3), np.nan, dtype=np.float32)
                        plddt_padded = np.full(len(expected_seq) + 1, np.nan, dtype=np.float32)
                        atomx_padded[:vh_len] = atomx[:vh_len]
                        atomx_padded[vh_len + 1:] = atomx[vh_len:]
                        plddt_padded[:vh_len] = plddt[:vh_len]
                        plddt_padded[vh_len + 1:] = plddt[vh_len:]
                        atomx, plddt = atomx_padded, plddt_padded
                    if s3di_tokens is not None and light_seq:
                        vh_len = len(heavy_seq)
                        s3di_tokens = s3di_tokens[:vh_len] + "X" + s3di_tokens[vh_len:]
            except Exception as e:
                warnings.warn(f"Structure loading failed for idx={idx}: {e}")
                atomx, plddt, s3di_tokens = None, None, None

        # Tokenize
        tokenized = tokenize_sequence(
            sequence=sequence,
            alphabet=alphabet,
            s3di_alphabet=s3di_alphabet,
            device=device_obj,
            atomx=atomx,
            plddt=plddt,
            s3di=s3di_tokens,
        )

        # Get encoder representations
        with torch.no_grad():
            output = model.encoder_outputs(
                xs=tokenized["seqs"],
                segment_sizes=tokenized["segment_sizes"],
                xs_plddts=tokenized["plddts"],
                xs_s3dis=tokenized["s3dis"],
                xs_atomxs=tokenized["atomxs"],
                xs_atombs=tokenized["atombs"],
                repr_layers=(-1,),
            )

        # Extract hidden states: PackedTensorSequences → padded tensor
        reprs = output.reprs[-1]
        reprs.make_to_paddedable()
        padded = reprs.to_padded()  # [1, L, 1024]

        hidden = padded[0]  # [L, 1024]
        # L = 1(start) + len(heavy) + 1(separator if paired) + len(light)
        # Exclude start token (pos 0), separator, and padding
        # Positions: 0=start, 1..vh_len=heavy, vh_len+1=separator, vh_len+2..=light
        vh_len = len(heavy_seq)
        if light_seq:
            # Take heavy (1..vh_len+1) and light (vh_len+2..vh_len+2+vl_len)
            vl_len = len(light_seq)
            seq_emb = torch.cat([
                hidden[1:vh_len+1],            # heavy
                hidden[vh_len+2:vh_len+2+vl_len],  # light
            ], dim=0)  # [vh_len + vl_len, 1024]
        else:
            seq_emb = hidden[1:vh_len+1]  # [vh_len, 1024]

        pooled = seq_emb.float().mean(dim=0).cpu()  # [1024]
        all_embs.append(pooled)

    return torch.stack(all_embs)  # [N, 1024]


# ============================================================================
# Main
# ============================================================================

HIDDEN_DIMS = {
    "antiberty": 1024,  # 512 * 2 (H+L concat)
    "esm2": 2560,       # 1280 * 2
    "ism": 2560,         # 1280 * 2
    "ablang2": 480,
    "sablm_str": 1024,
    "sablm_nostr": 1024,
}


def main():
    args = get_args()

    # Output directory
    if args.output_dir is None:
        output_dir = infer_output_dir(args.csv_path, args.model)
    else:
        output_dir = args.output_dir

    output_path = os.path.join(output_dir, "embeddings.pt")

    # Skip if already exists
    if os.path.exists(output_path):
        print(f"SKIP: {output_path} already exists")
        return

    # Device
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load data
    df = pd.read_csv(args.csv_path)
    print(f"Dataset: {args.csv_path} ({len(df)} samples)")
    print(f"Model: {args.model}, Device: {device}")

    # Extract embeddings
    if args.model == "antiberty":
        embeddings = extract_antiberty(df, device, args.batch_size)
    elif args.model == "esm2":
        embeddings = extract_esm2(df, device, args.batch_size, model_name="esm2")
    elif args.model == "ism":
        embeddings = extract_esm2(df, device, args.batch_size, model_name="ism")
    elif args.model == "ablang2":
        embeddings = extract_ablang2(df, device, args.batch_size)
    elif args.model == "sablm_str":
        if not args.structure_dir:
            warnings.warn("sablm_str requires --structure-dir. Skipping.")
            return
        embeddings = extract_sablm(df, device, args.batch_size,
                                    use_structure=True,
                                    structure_dir=args.structure_dir,
                                    checkpoint=args.checkpoint)
    elif args.model == "sablm_nostr":
        embeddings = extract_sablm(df, device, args.batch_size,
                                    use_structure=False,
                                    structure_dir=None,
                                    checkpoint=args.checkpoint)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_dict = {
        "embeddings": embeddings,
        "model": args.model,
        "hidden_dim": embeddings.shape[1],
        "n_samples": embeddings.shape[0],
        "csv_path": args.csv_path,
    }
    torch.save(save_dict, output_path)
    print(f"Saved: {output_path} (shape: {embeddings.shape})")


if __name__ == "__main__":
    main()
