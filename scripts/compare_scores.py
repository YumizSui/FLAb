#!/usr/bin/env python
"""
既存のscore/antibertyと再現結果score_reprodを比較するスクリプト

使用方法:
    python scripts/compare_scores.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def compare_scores(original_dir='score/antiberty', reprod_dir='score_reprod'):
    """
    既存の結果と再現結果を比較
    """
    original_path = Path(original_dir)
    reprod_path = Path(reprod_dir)

    if not original_path.exists():
        print(f"エラー: {original_dir}が見つかりません")
        return

    if not reprod_path.exists():
        print(f"エラー: {reprod_dir}が見つかりません")
        return

    # すべての_ppl.csvファイルを探す
    original_files = list(original_path.rglob('*_ppl.csv'))

    matches = []
    mismatches = []
    missing = []

    print(f"比較対象: {len(original_files)} ファイル\n")

    for orig_file in original_files:
        # 相対パスを取得
        rel_path = orig_file.relative_to(original_path)
        reprod_file = reprod_path / rel_path

        if not reprod_file.exists():
            missing.append(str(rel_path))
            continue

        try:
            # CSVファイルを読み込んで比較
            df_orig = pd.read_csv(orig_file)
            df_reprod = pd.read_csv(reprod_file)

            # 列名を正規化（大文字小文字を無視）
            df_orig.columns = df_orig.columns.str.lower()
            df_reprod.columns = df_reprod.columns.str.lower()

            # 必要な列が存在するか確認
            required_cols = ['heavy_perplexity', 'light_perplexity', 'average_perplexity']
            if not all(col in df_orig.columns for col in required_cols):
                mismatches.append((str(rel_path), "必要な列が見つかりません"))
                continue

            if not all(col in df_reprod.columns for col in required_cols):
                mismatches.append((str(rel_path), "再現結果に必要な列が見つかりません"))
                continue

            # 行数が一致するか確認
            if len(df_orig) != len(df_reprod):
                mismatches.append((str(rel_path), f"行数が異なります: {len(df_orig)} vs {len(df_reprod)}"))
                continue

            # perplexity値を比較
            max_diff = 0
            for col in required_cols:
                orig_vals = df_orig[col].values
                reprod_vals = df_reprod[col].values

                # NaNを除外
                mask = ~(np.isnan(orig_vals) | np.isnan(reprod_vals))
                if mask.sum() > 0:
                    diff = np.abs(orig_vals[mask] - reprod_vals[mask])
                    max_diff = max(max_diff, np.max(diff))

            # 許容誤差（浮動小数点誤差を考慮）
            tolerance = 1e-5
            if max_diff <= tolerance:
                matches.append((str(rel_path), max_diff))
            else:
                mismatches.append((str(rel_path), f"最大差分: {max_diff:.2e}"))

        except Exception as e:
            mismatches.append((str(rel_path), f"エラー: {e}"))

    # 結果を表示
    print("="*60)
    print("比較結果")
    print("="*60)
    print(f"一致: {len(matches)}/{len(original_files)}")
    print(f"不一致: {len(mismatches)}")
    print(f"欠落: {len(missing)}")
    print()

    if matches:
        print("一致したファイル:")
        for path, diff in matches[:10]:  # 最初の10個のみ表示
            print(f"  ✓ {path} (最大差分: {diff:.2e})")
        if len(matches) > 10:
            print(f"  ... 他 {len(matches) - 10} ファイル")
        print()

    if mismatches:
        print("不一致またはエラー:")
        for path, reason in mismatches[:10]:  # 最初の10個のみ表示
            print(f"  ✗ {path}: {reason}")
        if len(mismatches) > 10:
            print(f"  ... 他 {len(mismatches) - 10} ファイル")
        print()

    if missing:
        print("再現結果が見つからないファイル:")
        for path in missing[:10]:  # 最初の10個のみ表示
            print(f"  - {path}")
        if len(missing) > 10:
            print(f"  ... 他 {len(missing) - 10} ファイル")
        print()

    # サマリー
    total_compared = len(matches) + len(mismatches)
    if total_compared > 0:
        match_rate = len(matches) / total_compared * 100
        print(f"一致率: {match_rate:.1f}% ({len(matches)}/{total_compared})")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='既存の結果と再現結果を比較',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--original-dir',
        type=str,
        default='score/antiberty',
        help='既存の結果ディレクトリ（デフォルト: score/antiberty）'
    )

    parser.add_argument(
        '--reprod-dir',
        type=str,
        default='score_reprod',
        help='再現結果ディレクトリ（デフォルト: score_reprod）'
    )

    args = parser.parse_args()

    compare_scores(args.original_dir, args.reprod_dir)

if __name__ == '__main__':
    main()










