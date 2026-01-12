#!/usr/bin/env python
"""
AntiBERTyを使用して抗体配列のperplexityを計算し、fitnessとの相関を評価するスクリプト

使用方法:
    python scripts/score_antiberty.py data/thermostability/hie2023efficient_C143_Tm.csv
    python scripts/score_antiberty.py data/thermostability/hie2023efficient_C143_Tm.csv --ppl-only

出力:
    - score/antiberty/{fitness_dir}/{dataset_name}_ppl.csv: perplexityスコア（常に出力）
    - score/antiberty/{fitness_dir}/{dataset_name}_corr.csv: 相関係数（--ppl-only指定時はスキップ）
    - score/antiberty/{fitness_dir}/{dataset_name}_plot.png: 散布図（--ppl-only指定時はスキップ）
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # バックエンドを明示的に指定（GUI不要）
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# スクリプトディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from models import antiberty_score
from extra import dir_create


def get_fitness_metric(fitness_dir, df):
    """
    fitnessディレクトリ名に基づいて適切なfitness metricを設定
    """
    if fitness_dir == 'binding':
        fitness_metric = 'negative log Kd'
        df[fitness_metric] = -np.log(df['fitness'])
    elif fitness_dir == 'immunogenicity':
        fitness_metric = 'immunogenic response (%)'
        df[fitness_metric] = df['fitness']
    elif fitness_dir == 'tm' or fitness_dir == 'thermostability':
        fitness_metric = 'Melting temperature (°C)'
        df[fitness_metric] = df['fitness']
    elif fitness_dir == 'expression':
        fitness_metric = 'negative log expression'
        df[fitness_metric] = -np.log(df['fitness'])
    elif fitness_dir == 'aggregation':
        fitness_metric = 'aggregation metric'
        df[fitness_metric] = df['fitness']
    elif fitness_dir == 'polyreactivity':
        fitness_metric = 'polyreactivity metric'
        df[fitness_metric] = df['fitness']
    elif fitness_dir == 'pharmacokinetics':
        fitness_metric = 'pharmacokinetics metric'
        df[fitness_metric] = df['fitness']
    else:
        fitness_metric = 'fitness'
        df[fitness_metric] = df['fitness']

    return fitness_metric


def main():
    parser = argparse.ArgumentParser(
        description='AntiBERTyを使用して抗体配列をスコアリング',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python scripts/score_antiberty.py data/thermostability/hie2023efficient_C143_Tm.csv
  python scripts/score_antiberty.py data/expression/jain2017biophysical_HEK.csv
        """
    )

    parser.add_argument(
        'csv_path',
        type=str,
        help='CSVファイルのパス（heavy, light, fitness列が必要）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力ディレクトリ（デフォルト: score/antiberty/{fitness_dir}/{dataset_name}）'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='AntiBERTyのバッチサイズ（デフォルト: 16）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='使用するデバイス（cuda:0, cpuなど。デフォルト: CUDAが利用可能ならcuda:0、そうでなければcpu）'
    )

    parser.add_argument(
        '--ppl-only',
        action='store_true',
        help='perplexityのみを計算し、プロットと相関係数の計算をスキップする'
    )

    args = parser.parse_args()

    # CSVファイルの存在確認
    if not os.path.exists(args.csv_path):
        print(f"エラー: ファイルが見つかりません: {args.csv_path}")
        sys.exit(1)

    # パスの解析
    csv_path = os.path.abspath(args.csv_path)
    dir_name, filename = os.path.split(csv_path)
    data_dir, fitness_dir = os.path.split(dir_name)
    name_only, _ = os.path.splitext(filename)

    # 出力ディレクトリの設定
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # デフォルトの出力パス
        score_dir = 'score'
        dir_create(score_dir)
        dir_create(score_dir, 'antiberty')
        dir_create(score_dir, 'antiberty', fitness_dir)
        dir_create(score_dir, 'antiberty', fitness_dir, name_only)
        output_dir = os.path.join(score_dir, 'antiberty', fitness_dir, name_only)

    os.makedirs(output_dir, exist_ok=True)

    print(f"データセット: {name_only}")
    print(f"カテゴリ: {fitness_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    # CSVの読み込み
    print(f"\nCSVファイルを読み込み中: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        sys.exit(1)

    # 必要な列の確認
    required_columns = ['heavy', 'light', 'fitness']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"エラー: 必要な列が見つかりません: {missing_columns}")
        print(f"利用可能な列: {list(df.columns)}")
        sys.exit(1)

    print(f"データ数: {len(df)} 配列")

    # Perplexityの計算
    print("\nAntiBERTyでperplexityを計算中...")
    try:
        import torch
        if args.device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"GPUを使用: {device}")
            else:
                device = 'cpu'
                print("CPUを使用")
        else:
            device = args.device
            print(f"デバイスを指定: {device}")

        df = antiberty_score(df, batch_size=args.batch_size, device=device)
        print("計算完了")
    except Exception as e:
        print(f"エラー: perplexityの計算に失敗しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Perplexityスコアの保存
    ppl_path = os.path.join(output_dir, f'{name_only}_ppl.csv')
    df.to_csv(ppl_path, index=False)
    print(f"Perplexityスコアを保存: {ppl_path}")

    # ppl-onlyオプションが指定されている場合はここで終了
    if args.ppl_only:
        print(f"\nPerplexityスコアは {ppl_path} に保存されました")
        return

    # Fitness metricの設定
    fitness_metric = get_fitness_metric(fitness_dir, df)

    # プロットの作成
    print("\nプロットを作成中...")
    plt.figure(figsize=(10, 6))

    df['is_first_row'] = df.index == 0
    colors = ['orange' if row else 'blue' for row in df['is_first_row']]
    plt.scatter(df[fitness_metric], df['average_perplexity'],
                c=colors, alpha=0.6, s=50)

    # wildtypeを強調
    if len(df) > 0:
        plt.scatter(df[fitness_metric].iloc[0], df['average_perplexity'].iloc[0],
                   c='orange', s=100, label='wildtype', zorder=5, edgecolors='black', linewidths=1)

    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='orange', label='wildtype', linestyle='None', markersize=8),
        plt.Line2D([], [], marker='o', color='blue', label='mutants', linestyle='None', markersize=8)
    ])
    plt.xlabel(fitness_metric)
    plt.ylabel('average perplexity')
    plt.title(f'{name_only} - AntiBERTy')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, f'{name_only}_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"プロットを保存: {plot_path}")

    # 相関係数の計算
    print("\n相関係数を計算中...")
    name_list = ['pearson', 'spearman', 'kendall tau']
    correlation_list = []
    p_list = []

    # Pearson相関
    pearson_corr, pearson_p_value = pearsonr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(pearson_corr)
    p_list.append(pearson_p_value)

    # Spearman相関
    spearman_corr, spearman_p_value = spearmanr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(spearman_corr)
    p_list.append(spearman_p_value)

    # Kendall相関
    kendall_corr, kendall_p_value = kendalltau(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(kendall_corr)
    p_list.append(kendall_p_value)

    df_corr = pd.DataFrame({
        'correlation_name': name_list,
        'value': correlation_list,
        'p-value': p_list
    })

    # 相関係数の保存
    corr_path = os.path.join(output_dir, f'{name_only}_corr.csv')
    df_corr.to_csv(corr_path, index=False)
    print(f"相関係数を保存: {corr_path}")

    # 結果の表示
    print("\n" + "="*60)
    print("結果サマリー")
    print("="*60)
    print(df_corr.to_string(index=False))
    print("="*60)
    print(f"\nすべての結果は {output_dir} に保存されました")


if __name__ == '__main__':
    main()

