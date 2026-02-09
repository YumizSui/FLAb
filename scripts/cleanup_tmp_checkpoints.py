#!/usr/bin/env python3
"""Clean up tmp checkpoint directories when _ppl.csv exists.

Walks through score/ directory and removes tmp/ subdirectories
for any dataset that has a completed _ppl.csv file.

Usage:
    python scripts/cleanup_tmp_checkpoints.py              # dry-run (show what would be deleted)
    python scripts/cleanup_tmp_checkpoints.py --execute    # actually delete
"""

import argparse
import os
import shutil
from pathlib import Path


def find_tmp_dirs_to_clean(score_root="score"):
    """Find all tmp directories where parent has a _ppl.csv file.

    Returns:
        List of tuples: (tmp_dir_path, ppl_csv_path, dataset_name)
    """
    to_clean = []
    score_path = Path(score_root)

    if not score_path.exists():
        print(f"Warning: {score_root} does not exist")
        return to_clean

    # Find all tmp directories
    for tmp_dir in score_path.rglob("tmp"):
        if not tmp_dir.is_dir():
            continue

        parent_dir = tmp_dir.parent
        dataset_name = parent_dir.name
        ppl_csv = parent_dir / f"{dataset_name}_ppl.csv"

        if ppl_csv.exists():
            chunk_count = len(list(tmp_dir.glob("*.csv")))
            to_clean.append((tmp_dir, ppl_csv, dataset_name, chunk_count))

    return to_clean


def main():
    parser = argparse.ArgumentParser(
        description="Clean up tmp checkpoint directories when _ppl.csv exists"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete tmp directories (default: dry-run only)"
    )
    parser.add_argument(
        "--score-dir",
        default="score",
        help="Root directory to scan (default: score)"
    )
    args = parser.parse_args()

    to_clean = find_tmp_dirs_to_clean(args.score_dir)

    if not to_clean:
        print("No tmp directories to clean (no _ppl.csv files found with tmp/)")
        return

    print(f"Found {len(to_clean)} tmp directories to clean:")
    print()

    total_chunks = 0
    for tmp_dir, ppl_csv, dataset_name, chunk_count in to_clean:
        print(f"  {tmp_dir}")
        print(f"    -> {dataset_name}_ppl.csv exists")
        print(f"    -> {chunk_count} chunk files")
        total_chunks += chunk_count

    print()
    print(f"Total: {len(to_clean)} tmp directories, {total_chunks} chunk CSV files")
    print()

    if args.execute:
        print("EXECUTING deletion...")
        for tmp_dir, _, _, _ in to_clean:
            try:
                shutil.rmtree(tmp_dir)
                print(f"  DELETED: {tmp_dir}")
            except Exception as e:
                print(f"  ERROR deleting {tmp_dir}: {e}")
        print()
        print("Done.")
    else:
        print("DRY-RUN mode (no files deleted)")
        print("Run with --execute to actually delete these directories")


if __name__ == "__main__":
    main()
