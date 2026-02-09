#!/usr/bin/env python3
"""
One-hot encoding特徴抽出スクリプト

配列をone-hot encodingに変換（ベースラインモデル用）
- 20アミノ酸 + padding + separator (22次元)
- Heavy + separator + Light → concat
- 出力: [N, max_len, 22] → flatten → [N, max_len*22]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE_DATASETS_CSV = ROOT / "data" / "fewshot_eligible_datasets.csv"
OUTPUT_DIR = ROOT / "embs" / "onehot"

# 20標準アミノ酸 + padding + separator
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN = "<PAD>"
SEP_TOKEN = "|"
ALL_TOKENS = AMINO_ACIDS + [PAD_TOKEN, SEP_TOKEN]
TOKEN_TO_IDX = {token: i for i, token in enumerate(ALL_TOKENS)}

# ============================================================================
# One-hot Encoding
# ============================================================================

def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """
    配列をone-hot encodingに変換

    Args:
        seq: アミノ酸配列（例: "EVQL..."|"DIQ..."）
        max_len: 最大長（padding用）

    Returns:
        one_hot: [max_len, 22] array
    """
    # Heavy|Light split
    if "|" in seq:
        parts = seq.split("|")
        heavy = parts[0]
        light = parts[1] if len(parts) > 1 else ""
        # Heavy + separator + Light
        full_seq = heavy + SEP_TOKEN + light
    else:
        full_seq = seq

    # Truncate if too long
    full_seq = full_seq[:max_len]

    # One-hot encoding
    one_hot = np.zeros((max_len, len(ALL_TOKENS)), dtype=np.float32)

    for i, aa in enumerate(full_seq):
        if aa in TOKEN_TO_IDX:
            one_hot[i, TOKEN_TO_IDX[aa]] = 1.0
        else:
            # Unknown amino acid → treat as padding
            one_hot[i, TOKEN_TO_IDX[PAD_TOKEN]] = 1.0

    # Padding for remaining positions
    for i in range(len(full_seq), max_len):
        one_hot[i, TOKEN_TO_IDX[PAD_TOKEN]] = 1.0

    return one_hot


def extract_onehot_features(csv_path: str, output_dir: Path) -> None:
    """
    データセットのone-hot encodingを抽出

    Args:
        csv_path: CSVファイルパス
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "embeddings.pt"

    # Skip if already exists
    if output_file.exists():
        print(f"✓ Already exists: {output_file}")
        return

    # Load CSV
    df = pd.read_csv(csv_path)

    if 'sequence' not in df.columns:
        print(f"⚠ Warning: 'sequence' column not found in {csv_path}")
        return

    sequences = df['sequence'].tolist()
    n_samples = len(sequences)

    # Compute max length
    max_len = max(len(seq.replace("|", "") + "|") for seq in sequences)

    print(f"Dataset: {csv_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Max length: {max_len}")

    # Extract one-hot encodings
    all_features = []

    for seq in tqdm(sequences, desc="Extracting one-hot", leave=False):
        one_hot = encode_sequence(seq, max_len)  # [max_len, 22]
        # Flatten: [max_len * 22]
        flattened = one_hot.flatten()
        all_features.append(flattened)

    all_features = np.stack(all_features, axis=0)  # [N, max_len*22]

    # Save
    torch.save({
        "embeddings": torch.from_numpy(all_features),  # [N, D]
        "model": "onehot",
        "hidden_dim": all_features.shape[1],
        "max_len": max_len,
        "n_tokens": len(ALL_TOKENS),
    }, output_file)

    print(f"✓ Saved: {output_file} (shape: {all_features.shape})")


# ============================================================================
# Batch Processing
# ============================================================================

def extract_all_datasets():
    """全eligible datasetのone-hot encodingを抽出"""
    eligible_df = pd.read_csv(ELIGIBLE_DATASETS_CSV)

    print(f"\nTotal datasets: {len(eligible_df)}")

    for _, row in eligible_df.iterrows():
        csv_path = ROOT / row['csv_path']
        category = row['category']
        dataset_name = row['dataset_name']

        output_dir = OUTPUT_DIR / category / dataset_name

        extract_onehot_features(str(csv_path), output_dir)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract one-hot encoding features")
    parser.add_argument("--csv-path", type=str, help="CSV file path")
    parser.add_argument("--category", type=str, help="Category name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--run-all", action="store_true", help="Process all eligible datasets")

    args = parser.parse_args()

    if args.run_all:
        extract_all_datasets()
    elif args.csv_path and args.category and args.dataset:
        output_dir = OUTPUT_DIR / args.category / args.dataset
        extract_onehot_features(args.csv_path, output_dir)
    else:
        print("Error: Either --run-all or (--csv-path, --category, --dataset) required")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
