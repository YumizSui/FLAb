#!/usr/bin/env python3
"""
Predict antibody structures using Chai-1 from CSV file.

Usage:
    python scripts/predict_structures_chai.py \
        --csv-path data/thermostability/rosace2023automated_tm1_golimumab.csv \
        --output-dir structure/thermostability/rosace2023automated_tm1_golimumab \
        --num-samples 1
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def create_fasta(heavy_seq: str, light_seq: str, output_path: Path):
    """
    Create a FASTA file for antibody heavy and light chains.

    Args:
        heavy_seq: Heavy chain sequence
        light_seq: Light chain sequence
        output_path: Path to save FASTA file
    """
    with open(output_path, 'w') as f:
        f.write(f">protein|name=H\n{heavy_seq}\n")
        f.write(f">protein|name=L\n{light_seq}\n")


def predict_structure(fasta_path: Path, output_dir: Path, num_samples: int = 1, device: str = "cuda:0"):
    """
    Run Chai-1 structure prediction.

    Args:
        fasta_path: Path to FASTA file
        output_dir: Directory to save output
        num_samples: Number of diffusion samples
        device: Device to use (cuda:0 or cpu)
    """
    from chai_lab.chai1 import run_inference

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    print(f"Running Chai-1 prediction for {fasta_path.name}...")
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_diffn_samples=num_samples,
        use_esm_embeddings=True,
        use_msa_server=False,  # Skip MSA for speed
        device=device,
        low_memory=True,
    )

    return candidates


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path):
    """
    Convert CIF file to PDB format using Biopython.
    Renames chains A, B to H, L for antibody heavy and light chains.

    Args:
        cif_path: Path to input CIF file
        pdb_path: Path to output PDB file
    """
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("antibody", cif_path)

    # Rename chains: A -> H (heavy), B -> L (light)
    chain_mapping = {"A": "H", "B": "L"}
    for model in structure:
        for chain in model:
            if chain.id in chain_mapping:
                chain.id = chain_mapping[chain.id]

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def _get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Predict antibody structures using Chai-1")

    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with heavy and light chain sequences"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for predicted structures"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of diffusion samples (default: 1 for speed)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0 or cpu)"
    )

    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to predict (for testing)"
    )

    return parser.parse_args()


def main():
    args = _get_args()

    # Read CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} sequences from {args.csv_path}")

    # Limit number of structures if specified
    if args.max_structures is not None:
        df = df.head(args.max_structures)
        print(f"Processing only first {len(df)} structures")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for FASTA files
    fasta_dir = output_dir / "fasta_inputs"
    fasta_dir.mkdir(exist_ok=True)

    # Process each antibody
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting structures"):
        heavy_seq = row['heavy']
        light_seq = row['light']

        # Create FASTA file
        fasta_path = fasta_dir / f"antibody_{idx}.fasta"
        create_fasta(heavy_seq, light_seq, fasta_path)

        # Create output directory for this structure
        structure_output_dir = output_dir / f"chai_output_{idx}"

        # Skip if already predicted
        final_pdb = output_dir / f"{Path(args.csv_path).stem}_{idx}.pdb"
        if final_pdb.exists():
            print(f"Skipping {idx}: {final_pdb} already exists")
            continue

        try:
            # Run prediction
            candidates = predict_structure(
                fasta_path=fasta_path,
                output_dir=structure_output_dir,
                num_samples=args.num_samples,
                device=args.device,
            )

            # Find the best (rank 0) CIF file
            cif_files = list(structure_output_dir.glob("pred.model_idx_*.cif"))
            if not cif_files:
                print(f"Warning: No CIF files found for sequence {idx}")
                continue

            # Use the first prediction (usually pred.model_idx_0.cif)
            best_cif = sorted(cif_files)[0]

            # Convert to PDB
            print(f"Converting {best_cif.name} to PDB...")
            convert_cif_to_pdb(best_cif, final_pdb)

            print(f"✓ Saved: {final_pdb}")

        except Exception as e:
            print(f"✗ Error predicting structure {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! Predicted structures saved to {output_dir}")
    print(f"Total PDB files: {len(list(output_dir.glob('*.pdb')))}")


if __name__ == '__main__':
    main()
