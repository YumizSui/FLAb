#!/usr/bin/env python3
"""
Batch predict antibody structures using Chai-1 for a single dataset CSV.

Supports chunk-based processing for large datasets and parallel execution via array jobs.
Uses chai_lab.run_inference() to avoid memory accumulation issues.

Usage:
    # Process all entries in a dataset
    python scripts/predict_structures_chai_batch.py --input data/thermostability/garbinski2023_tm1.csv

    # Dry-run (show chunk info)
    python scripts/predict_structures_chai_batch.py \
        --input data/thermostability/garbinski2023_tm1.csv \
        --chunk-size 1000 \
        --dry-run

    # Process specific chunk (for array jobs)
    python scripts/predict_structures_chai_batch.py \
        --input data/thermostability/garbinski2023_tm1.csv \
        --chunk-index 0 \
        --chunk-size 1000
"""

import argparse
import gc
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import traceback
from math import ceil

import torch

# Import functions from existing script
from predict_structures_chai import create_fasta, convert_cif_to_pdb, predict_structure

import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"


def cleanup_intermediate_files(structure_output_dir: Path):
    """
    Remove intermediate files after successful prediction.

    Args:
        structure_output_dir: Directory containing chai_output files
    """
    # Remove chai_output directory (keep FASTA file)
    if structure_output_dir.exists():
        shutil.rmtree(structure_output_dir)


def process_chunk(
    chunk_df,
    dataset_stem: str,
    category: str,
    output_base_dir: Path,
    chunk_idx: int | None,
    total_chunks: int,
    num_samples: int,
    device: str,
    skip_existing: bool,
):
    """
    Process a single chunk of data.

    Args:
        chunk_df: DataFrame chunk to process
        dataset_stem: Dataset stem name (e.g., "garbinski2023_tm1")
        category: Dataset category (e.g., "thermostability")
        output_base_dir: Base output directory path (e.g., Path("new_structure"))
        chunk_idx: Chunk index (None if not chunking)
        total_chunks: Total number of chunks (None if not chunking)
        num_samples: Number of diffusion samples
        device: Device to use
        skip_existing: Whether to skip existing structures

    Returns:
        dict: Statistics {success: N, failed: M, skipped: K}
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}

    # Build output directory
    output_dir = output_base_dir / category / dataset_stem

    # Add chunk subdirectory if chunking
    if chunk_idx is not None:
        output_dir = output_dir / f"chunk_{chunk_idx}"
        print(f"  Chunk {chunk_idx}/{total_chunks - 1}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each row in chunk
    for idx, row in tqdm(
        chunk_df.iterrows(),
        total=len(chunk_df),
        desc=f"  {dataset_stem} chunk {chunk_idx}" if chunk_idx is not None else f"  {dataset_stem}",
        leave=False,
    ):
        # Build output filename (same base name for PDB and FASTA)
        if chunk_idx is not None:
            base_filename = f"{dataset_stem}_chunk_{chunk_idx}_{idx}"
        else:
            base_filename = f"{dataset_stem}_{idx}"

        final_pdb = output_dir / f"{base_filename}.pdb"
        final_fasta = output_dir / f"{base_filename}.fasta"

        # Skip if exists
        if skip_existing and final_pdb.exists():
            stats["skipped"] += 1
            continue

        try:
            # Get sequences
            heavy_seq = row["heavy"]
            light_seq = row["light"]

            # Create FASTA (same name as PDB, in same directory)
            create_fasta(heavy_seq, light_seq, final_fasta)

            # Create temporary output directory for Chai-1
            structure_output_dir = output_dir / f"chai_output_{idx}"

            # Run prediction using standard run_inference (loads/unloads models each time)
            predict_structure(
                fasta_path=final_fasta,
                output_dir=structure_output_dir,
                num_samples=num_samples,
                device=device,
            )

            # Find best CIF file
            cif_files = list(structure_output_dir.glob("pred.model_idx_*.cif"))
            if not cif_files:
                raise ValueError(f"No CIF files found in {structure_output_dir}")

            # Use first prediction (best ranked)
            best_cif = sorted(cif_files)[0]

            # Convert to PDB (with H/L chain names)
            convert_cif_to_pdb(best_cif, final_pdb)

            print(f"  Saved: {final_pdb}, {final_fasta}")
            stats["success"] += 1

            # Cleanup intermediate files on success (keep FASTA, delete chai_output)
            cleanup_intermediate_files(structure_output_dir)

            # Memory cleanup after each prediction
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at row {idx}: {e}")
            traceback.print_exc()
            stats["failed"] += 1
            # Cleanup GPU memory even on failure
            gc.collect()
            torch.cuda.empty_cache()
            # Keep intermediate files for debugging on failure
            continue

    return stats


def print_chunk_info(input_path: Path, df: pd.DataFrame, chunk_size: int):
    """Print chunk information for the dataset."""
    num_chunks = ceil(len(df) / chunk_size)

    print("\nDataset Chunk Information:")
    print("-" * 80)
    print(f"Input file: {input_path}")
    print(f"Total entries: {len(df)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Total chunks: {num_chunks}")
    print("-" * 80)

    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, len(df))
        print(f"  Chunk {chunk_idx}: rows {start_row}-{end_row - 1} ({end_row - start_row} entries)")

    print("-" * 80)


def _get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch predict antibody structures using Chai-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all entries
    python scripts/predict_structures_chai_batch.py --input data/thermostability/garbinski2023_tm1.csv

    # Dry-run to see chunk info
    python scripts/predict_structures_chai_batch.py --input data/thermostability/garbinski2023_tm1.csv --dry-run

    # Process specific chunk (for array jobs)
    python scripts/predict_structures_chai_batch.py --input data/thermostability/garbinski2023_tm1.csv --chunk-index 0 --chunk-size 1000
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset CSV file with heavy and light chain sequences",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="new_structure",
        help="Base output directory (default: new_structure)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of entries per chunk (default: 1000)",
    )

    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Dataset-local chunk index for array jobs (optional, 0-indexed)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of diffusion samples (default: 1)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip existing PDB files (default: True)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing PDB files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunk info only (no processing)",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    # Read the input CSV directly
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} entries from {input_path}")

    # Derive output directory structure from input path
    # e.g., data/thermostability/garbinski2023_tm1.csv
    #    -> new_structure/thermostability/garbinski2023_tm1/
    category = input_path.parent.name  # "thermostability"
    dataset_stem = input_path.stem      # "garbinski2023_tm1"

    output_dir = Path(args.output_dir)

    # Calculate chunks for this dataset only
    num_chunks = ceil(len(df) / args.chunk_size)

    # Dry run mode
    if args.dry_run:
        print_chunk_info(input_path, df, args.chunk_size)
        return

    # Validate chunk index if provided
    if args.chunk_index is not None:
        if args.chunk_index < 0 or args.chunk_index >= num_chunks:
            print(f"Error: Invalid chunk index {args.chunk_index}")
            print(f"Valid range: 0-{num_chunks - 1}")
            sys.exit(1)

    # Overall statistics
    overall_stats = {"success": 0, "failed": 0, "skipped": 0}

    # MODE 1: Chunk index mode (specific chunk)
    if args.chunk_index is not None:
        print(f"\n=== Processing chunk {args.chunk_index}/{num_chunks - 1} ===")

        start_row = args.chunk_index * args.chunk_size
        end_row = min(start_row + args.chunk_size, len(df))
        chunk_df = df.iloc[start_row:end_row]

        print(f"Dataset: {dataset_stem}")
        print(f"Category: {category}")
        print(f"Rows: {start_row}-{end_row - 1} ({len(chunk_df)} entries)")

        stats = process_chunk(
            chunk_df=chunk_df,
            dataset_stem=dataset_stem,
            category=category,
            output_base_dir=output_dir,
            chunk_idx=args.chunk_index if num_chunks > 1 else None,
            total_chunks=num_chunks,
            num_samples=args.num_samples,
            device=args.device,
            skip_existing=args.skip_existing,
        )

        overall_stats = stats

    # MODE 2: Normal mode (process all chunks sequentially)
    else:
        print(f"\n=== Processing all {num_chunks} chunks ===")

        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * args.chunk_size
            end_row = min(start_row + args.chunk_size, len(df))
            chunk_df = df.iloc[start_row:end_row]

            print(f"\nChunk {chunk_idx}/{num_chunks - 1}: rows {start_row}-{end_row - 1}")

            try:
                stats = process_chunk(
                    chunk_df=chunk_df,
                    dataset_stem=dataset_stem,
                    category=category,
                    output_base_dir=output_dir,
                    chunk_idx=chunk_idx if num_chunks > 1 else None,
                    total_chunks=num_chunks,
                    num_samples=args.num_samples,
                    device=args.device,
                    skip_existing=args.skip_existing,
                )

                # Accumulate stats
                for key in stats:
                    overall_stats[key] += stats[key]

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                traceback.print_exc()
                overall_stats["failed"] += len(chunk_df)
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: {dataset_stem}")
    print(f"Category: {category}")
    print(f"Success:  {overall_stats['success']}")
    print(f"Failed:   {overall_stats['failed']}")
    print(f"Skipped:  {overall_stats['skipped']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
