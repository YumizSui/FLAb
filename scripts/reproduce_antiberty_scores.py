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
import json
from pathlib import Path

def get_dataset_mapping():
    """
    データセット名のマッピングルールを返す
    戻り値: dict[tuple[fitness_dir, dataset_name], tuple[alt_fitness_dir, alt_dataset_name]]
    """
    mapping = {}

    # 完全一致マッピング
    # Koenig2017_g6_*
    mapping[('expression', 'Koenig2017_g6_er')] = ('expression', 'koenig2017mutational_er_g6')
    mapping[('binding', 'Koenig2017_g6_Kd')] = ('binding', 'koenig2017mutational_kd_g6')

    # Hie2022_* -> hie2023efficient_* (tm)
    mapping[('tm', 'Hie2022_C143_Tm')] = ('thermostability', 'hie2023efficient_C143_Tm')
    mapping[('tm', 'Hie2022_mAb114_Tm')] = ('thermostability', 'hie2023efficient_mAb114_Tm')
    mapping[('tm', 'Hie2022_mAb114UCA_Tm')] = ('thermostability', 'hie2023efficient_mAb114UCA_Tm')
    mapping[('tm', 'Hie2022_MEDI8852_Tm')] = ('thermostability', 'hie2023efficient_MEDI_Tm')
    mapping[('tm', 'Hie2022_MEDI8852UCA_Tm')] = ('thermostability', 'hie2023efficient_MEDIUCA_Tm')
    mapping[('tm', 'Hie2022_REGN10987_Tm')] = ('thermostability', 'hie2023efficient_REGN10987_Tm')
    mapping[('tm', 'Hie2022_S309_Tm')] = ('thermostability', 'hie2023efficient_S309_Tm')

    # Hie2022_* -> hie2023efficient_* (binding)
    mapping[('binding', 'Hie2022_C143_Kd')] = ('binding', 'hie2023efficient_CoV2Beta_C143_Kd')
    mapping[('binding', 'Hie2022_REGN10987_Kd')] = ('binding', 'hie2023efficient_CoV2Beta_REGN10987_Kd')
    mapping[('binding', 'Hie2022_S309_Kd')] = ('binding', 'hie2023efficient_CoV2_S309_Kd')
    mapping[('binding', 'Hie2022_mAb114_Kd')] = ('binding', 'hie2023efficient_ebola_mab114_Kd')
    mapping[('binding', 'Hie2022_MEDI8852_Kd')] = ('binding', 'hie2023efficient_MEDI_H4Hubei_Kd')
    mapping[('binding', 'Hie2022_MEDI8852UCA_Kd')] = ('binding', 'hie2023efficient_MEDIUCA_H1Solomon_Kd')

    # Rosace2023_*
    mapping[('tm', 'Rosace2023_Adalimumab_Tm')] = ('thermostability', 'rosace2023automated_tm1_adalimumab')
    mapping[('tm', 'Rosace2023_Golimumab_Tm')] = ('thermostability', 'rosace2023automated_tm1_golimumab')
    mapping[('binding', 'Rosace2023_Adalimumab_Kd')] = ('binding', 'rosace2023automated_kd_adalimumab')
    mapping[('binding', 'Rosace2023_Golimumab_Kd')] = ('binding', 'rosace2023automated_kd_golimumab')

    # Wittrup2017_CST_* -> jain2017biophysical_*
    mapping[('expression', 'Wittrup2017_CST_HEK')] = ('expression', 'jain2017biophysical_HEK')
    mapping[('expression', 'Wittrup2017_CST_BVP')] = ('polyreactivity', 'jain2017biophysical_BVPELISA')
    mapping[('tm', 'Wittrup2017_CST_Tm')] = ('thermostability', 'jain2017biophysical_Tm')
    mapping[('polyreactivity', 'Wittrup2017_CST_PSR')] = ('polyreactivity', 'jain2017biophysical_PSR')
    mapping[('polyreactivity', 'Wittrup2017_CST_ELISA')] = ('polyreactivity', 'jain2017biophysical_ELISA')
    mapping[('polyreactivity', 'Wittrup2017_CST_CIC')] = ('polyreactivity', 'jain2017biophysical_CICRT')
    mapping[('polyreactivity', 'Wittrup2017_CST_SMAC')] = ('polyreactivity', 'jain2017biophysical_SMACRT')
    mapping[('aggregation', 'Wittrup2017_CST_SMAC')] = ('polyreactivity', 'jain2017biophysical_SMACRT')
    mapping[('aggregation', 'Wittrup2017_CST_CSI')] = ('aggregation', 'jain2017biophysical_CSIBLI')
    mapping[('aggregation', 'Wittrup2017_CST_SAS')] = ('aggregation', 'jain2017biophysical_SAS')
    mapping[('aggregation', 'Wittrup2017_CST_HIC')] = ('aggregation', 'jain2017biophyscial_HICRT')  # typo in filename
    mapping[('aggregation', 'Wittrup2017_CST_SGAC')] = ('aggregation', 'jain2017biophysical_SGACSINS')
    mapping[('aggregation', 'Wittrup2017_CST_ACSINS')] = ('aggregation', 'jain2017biophysical_ACSINS')

    # Shanehsazzadeh2023_trastuzumab_*
    mapping[('binding', 'Shanehsazzadeh2023_trastuzumab_zero_kd')] = ('binding', 'shanehsazzadeh2023unlocking_zerokd_trastuzumab')

    # Warszawski2019_d44_Kd
    mapping[('binding', 'Warszawski2019_d44_Kd')] = ('binding', 'warszawski2019_d44_Kd')

    # Prihoda2021_mAb_immunogenicity
    mapping[('immunogenicity', 'Prihoda2021_mAb_immunogenicity')] = ('immunogenicity', 'marks2021humanization_immunogenicity')

    # gsk2023_* -> garbinski2023_* (パターンマッチングを辞書に追加)
    # expression
    mapping[('expression', 'gsk2023_RSB1_exp')] = ('expression', 'garbinski2023_exp')
    mapping[('expression', 'gsk2023_AM14_exp')] = ('expression', 'garbinski2023_exp')
    mapping[('expression', 'gsk2023_D25_exp')] = ('expression', 'garbinski2023_exp')
    mapping[('expression', 'gsk2023_MOTA_exp')] = ('expression', 'garbinski2023_exp')

    # binding
    mapping[('binding', 'gsk2023_RSB1_Kd')] = ('binding', 'garbinski2023_kd')
    mapping[('binding', 'gsk2023_AM14_Kd')] = ('binding', 'garbinski2023_kd')
    mapping[('binding', 'gsk2023_D25_Kd')] = ('binding', 'garbinski2023_kd')
    mapping[('binding', 'gsk2023_MOTA_Kd')] = ('binding', 'garbinski2023_kd')

    # tm
    mapping[('tm', 'gsk2023_RSB1_Tm')] = ('thermostability', 'garbinski2023_tm1')
    mapping[('tm', 'gsk2023_AM14_Tm')] = ('thermostability', 'garbinski2023_tm1')
    mapping[('tm', 'gsk2023_D25_Tm')] = ('thermostability', 'garbinski2023_tm1')
    mapping[('tm', 'gsk2023_MOTA_Tm')] = ('thermostability', 'garbinski2023_tm1')

    return mapping


def find_existing_results(fitness_dir_filter=None):
    """
    score/antibertyにある既存の結果をすべて見つける
    戻り値: [(fitness_dir, dataset_name, csv_path), ...]
    """
    import pandas as pd

    results = []
    score_dir = Path('score/antiberty')
    mapping = get_dataset_mapping()

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

            # 早期フィルタリング
            if fitness_dir_filter and fitness_dir != fitness_dir_filter:
                continue

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

                # 3. パターンマッチング: gsk2023_* -> garbinski2023_*
                if csv_path is None and dataset_name.startswith('gsk2023_'):
                    suffix = dataset_name.replace('gsk2023_', '')
                    if suffix.endswith('_exp'):
                        alt_name = 'garbinski2023_exp'
                        alt_path = Path('data') / fitness_dir / f"{alt_name}.csv"
                        if alt_path.exists():
                            csv_path = alt_path
                    elif suffix.endswith('_Tm') or fitness_dir == 'tm':
                        alt_name = 'garbinski2023_tm1'
                        alt_path = Path('data') / 'thermostability' / f"{alt_name}.csv"
                        if alt_path.exists():
                            csv_path = alt_path
                    elif suffix.endswith('_Kd') or fitness_dir == 'binding':
                        alt_name = 'garbinski2023_kd'
                        alt_path = Path('data') / fitness_dir / f"{alt_name}.csv"
                        if alt_path.exists():
                            csv_path = alt_path

                # 4. 辞書ベースのマッピング
                if csv_path is None:
                    key = (fitness_dir, dataset_name)
                    if key in mapping:
                        alt_fitness_dir, alt_dataset_name = mapping[key]
                        alt_path = Path('data') / alt_fitness_dir / f"{alt_dataset_name}.csv"
                        if alt_path.exists():
                            csv_path = alt_path

                # 5. Hie2022_* のパターンマッチング（tmディレクトリ用）
                if csv_path is None and dataset_name.startswith('Hie2022_') and fitness_dir == 'tm':
                    alt_name = dataset_name.replace('Hie2022_', 'hie2023efficient_')
                    if 'MEDI8852' in alt_name:
                        alt_name = alt_name.replace('MEDI8852', 'MEDI')
                    alt_path = Path('data') / 'thermostability' / f"{alt_name}.csv"
                    if alt_path.exists():
                        csv_path = alt_path

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

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際の実行をスキップして、マッピングの確認のみ行う'
    )

    parser.add_argument(
        '--export-mapping',
        type=str,
        default=None,
        help='マッピングをJSONファイルに出力（例: --export-mapping mapping.json）'
    )

    args = parser.parse_args()

    # マッピングをJSONに出力
    if args.export_mapping:
        # 実際に見つかったデータセットのマッピングを生成
        results = find_existing_results(fitness_dir_filter=args.fitness_dir)
        json_mapping = {}

        for fitness_dir, dataset_name, csv_path in results:
            key = f"{fitness_dir}/{dataset_name}"
            # csv_pathからfitness_dirとdataset_nameを抽出
            csv_path_obj = Path(csv_path)
            alt_fitness_dir = csv_path_obj.parent.name
            alt_dataset_name = csv_path_obj.stem
            json_mapping[key] = {
                "fitness_dir": alt_fitness_dir,
                "dataset_name": alt_dataset_name,
                "csv_path": csv_path
            }

        output_path = Path(args.export_mapping)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_mapping, f, indent=2, ensure_ascii=False)
        print(f"マッピングを {output_path} に出力しました")
        print(f"合計: {len(json_mapping)}個のマッピングルール")
        return

    # 既存の結果を探す
    print("既存の結果を検索中...")
    results = find_existing_results(fitness_dir_filter=args.fitness_dir)

    if not results:
        print("既存の結果が見つかりませんでした")
        sys.exit(1)

    # フィルタリング（既にfind_existing_resultsでフィルタリング済み）
    if args.fitness_dir:
        print(f"フィルタ: {args.fitness_dir} のみ処理")

    print(f"見つかったデータセット数: {len(results)}\n")

    # マッピングの確認のみ
    if args.dry_run:
        print("="*60)
        print("マッピング確認結果")
        print("="*60)
        for fitness_dir, dataset_name, csv_path in results:
            print(f"  {fitness_dir}/{dataset_name} -> {csv_path}")
        print(f"\n合計: {len(results)}個のデータセットが見つかりました")
        return

    # 再現実行
    reproduce_scores(results, output_base=args.output_dir, device=args.device)

if __name__ == '__main__':
    main()

