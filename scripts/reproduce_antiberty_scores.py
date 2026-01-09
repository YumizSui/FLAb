#!/usr/bin/env python
"""
既存のscore/antibertyの結果を再現し、score_reprodに保存するスクリプト

使用方法:
    python scripts/reproduce_antiberty_scores.py

既存のscore/antibertyにあるすべてのデータセットを再実行し、
score_reprodに同じ結果を生成します。
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

def find_existing_results():
    """
    score/antibertyにある既存の結果をすべて見つける
    戻り値: [(fitness_dir, dataset_name, csv_path), ...]
    """
    import pandas as pd

    results = []
    score_dir = Path('score/antiberty')

    if not score_dir.exists():
        print("score/antibertyディレクトリが見つかりません")
        return results

    # すべての_ppl.csvファイルを探す
    for ppl_file in score_dir.rglob('*_ppl.csv'):
        # パスを解析: score/antiberty/{fitness_dir}/{dataset_name}/{dataset_name}_ppl.csv
        parts = ppl_file.parts
        if len(parts) >= 4:
            fitness_dir = parts[2]  # score/antiberty/{fitness_dir}/...
            dataset_name = parts[3]  # score/antiberty/{fitness_dir}/{dataset_name}/...

            # まず、直接的なパスを試す
            csv_path = None

            # 1. 直接的なパス（fitness_dir/dataset_name.csv）
            direct_path = Path('data') / fitness_dir / f"{dataset_name}.csv"
            if direct_path.exists():
                csv_path = direct_path
            else:
                # 2. thermostabilityディレクトリを確認（tm -> thermostability）
                if fitness_dir == 'tm':
                    thermostability_path = Path('data') / 'thermostability' / f"{dataset_name}.csv"
                    if thermostability_path.exists():
                        csv_path = thermostability_path

                # 3. 既存の結果CSVからheavy/light列を読み込んで、dataディレクトリ内のCSVと比較
                if csv_path is None:
                    try:
                        df_result = pd.read_csv(ppl_file, nrows=1)
                        if 'heavy' in df_result.columns and 'light' in df_result.columns:
                            result_heavy = df_result['heavy'].iloc[0] if len(df_result) > 0 else None
                            result_light = df_result['light'].iloc[0] if len(df_result) > 0 else None

                            if result_heavy and result_light:
                                # dataディレクトリ内のすべてのCSVファイルを検索
                                data_dirs = ['data/thermostability', 'data/expression', 'data/binding',
                                           'data/aggregation', 'data/polyreactivity', 'data/immunogenicity',
                                           'data/pharmacokinetics']

                                for data_dir in data_dirs:
                                    data_path = Path(data_dir)
                                    if data_path.exists():
                                        for csv_file in data_path.glob('*.csv'):
                                            try:
                                                df_data = pd.read_csv(csv_file, nrows=1)
                                                if 'heavy' in df_data.columns and 'light' in df_data.columns:
                                                    data_heavy = df_data['heavy'].iloc[0] if len(df_data) > 0 else None
                                                    data_light = df_data['light'].iloc[0] if len(df_data) > 0 else None

                                                    if data_heavy == result_heavy and data_light == result_light:
                                                        csv_path = csv_file
                                                        break
                                            except:
                                                continue

                                    if csv_path:
                                        break
                    except Exception as e:
                        pass

            if csv_path and csv_path.exists():
                results.append((fitness_dir, dataset_name, str(csv_path)))
            else:
                print(f"警告: CSVファイルが見つかりません: {fitness_dir}/{dataset_name}, {csv_path}")

    return results

def reproduce_scores(results, output_base='score_reprod', device=None):
    """
    既存の結果を再現する
    """
    script_path = Path('scripts/score_antiberty.py')

    if not script_path.exists():
        print(f"エラー: スクリプトが見つかりません: {script_path}")
        sys.exit(1)

    total = len(results)
    print(f"再現するデータセット数: {total}\n")

    success_count = 0
    fail_count = 0
    failed_datasets = []

    for i, (fitness_dir, dataset_name, csv_path) in enumerate(results, 1):
        print(f"[{i}/{total}] 処理中: {fitness_dir}/{dataset_name}")
        print(f"  CSV: {csv_path}")

        # 出力ディレクトリを指定
        output_dir = f"{output_base}/{fitness_dir}/{dataset_name}"

        # コマンドを構築
        cmd = [
            sys.executable,
            str(script_path),
            csv_path,
            '--output-dir', output_dir
        ]

        if device:
            cmd.extend(['--device', device])

        try:
            # スクリプトを実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )

            if result.returncode == 0:
                print(f"  ✓ 成功: {output_dir}")
                success_count += 1
            else:
                print(f"  ✗ 失敗 (終了コード: {result.returncode})")
                print(f"  エラー: {result.stderr[:200]}")
                fail_count += 1
                failed_datasets.append((fitness_dir, dataset_name))
        except Exception as e:
            print(f"  ✗ 例外発生: {e}")
            fail_count += 1
            failed_datasets.append((fitness_dir, dataset_name))

        print()

    # サマリーを表示
    print("="*60)
    print("再現結果サマリー")
    print("="*60)
    print(f"成功: {success_count}/{total}")
    print(f"失敗: {fail_count}/{total}")

    if failed_datasets:
        print("\n失敗したデータセット:")
        for fitness_dir, dataset_name in failed_datasets:
            print(f"  - {fitness_dir}/{dataset_name}")

    print(f"\n結果は {output_base} に保存されました")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='既存のscore/antibertyの結果を再現',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # すべての結果を再現（デフォルトでGPU使用）
  python scripts/reproduce_antiberty_scores.py

  # CPUで実行
  python scripts/reproduce_antiberty_scores.py --device cpu

  # 特定のデバイスを指定
  python scripts/reproduce_antiberty_scores.py --device cuda:0
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='score_reprod',
        help='出力ディレクトリ（デフォルト: score_reprod）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='使用するデバイス（cuda:0, cpuなど。デフォルト: CUDAが利用可能ならcuda:0、そうでなければcpu）'
    )

    parser.add_argument(
        '--fitness-dir',
        type=str,
        default=None,
        help='特定のfitnessディレクトリのみ処理（例: tm, expression）'
    )

    args = parser.parse_args()

    # 既存の結果を探す
    print("既存の結果を検索中...")
    results = find_existing_results()

    if not results:
        print("既存の結果が見つかりませんでした")
        sys.exit(1)

    # フィルタリング
    if args.fitness_dir:
        results = [(f, d, c) for f, d, c in results if f == args.fitness_dir]
        print(f"フィルタ: {args.fitness_dir} のみ処理")

    print(f"見つかったデータセット数: {len(results)}\n")

    # 再現実行
    reproduce_scores(results, output_base=args.output_dir, device=args.device)

if __name__ == '__main__':
    main()

