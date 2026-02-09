#!/usr/bin/env python3
"""
FLAb1 ベンチマーク評価スクリプト

9モデル × 52データセットの予測性能を評価:
- データ数の整合性確認
- 相関計算（生の値 + 符号補正）
- AntiBERTy論文値との検証
- 可視化（ヒートマップ、バープロット、ボックスプロット、ランキング）
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "score"
DATA_DIR = ROOT / "data" / "FLAb_0"
OUTPUT_DIR = ROOT / "reports" / "flab1_evaluation"
MASTER_TABLE = DATA_DIR / "FLAb1_paper_table1_structure_used.csv"

# 9モデル（FLAb1）
MODELS = [
    "ablang2",
    "antiberty",
    "esm2",
    "esmif",
    "ism",
    "mpnn",
    "pyrosetta",
    "sablm_nostr",
    "sablm_str"
]

# esm2, ismは650Mサブディレクトリを使用
MODELS_WITH_SUBDIR = {
    "esm2": "650M",
    "ism": "650M"
}

# 符号補正ルール（論文: "The sign was inverted for aggregation, expression, and thermostability"）
# 基準: corr(-perplexity, fitness) を計算（低perplexity=良い→正の相関=良い予測）
# Figure 2のheatmapではagg/exp/tmのみ符号反転が適用される
# これは、これらのタスクで「低い値=悪い」という関係性のため
SIGN_CORRECTION = {
    "expression": -1,      # 論文Figure 2で反転
    "tm": -1,              # 論文Figure 2で反転
    "aggregation": -1,     # 論文Figure 2で反転
    "binding": 1,          # そのまま
    "immunogenicity": 1,   # そのまま
    "polyreactivity": 1,   # そのまま
}

# AntiBERTy論文期待値（Spearman、Tables 3-4から抽出）
# 注: 論文値と既存_corr.csvファイルに不一致があるため、
# このスクリプトでは既存_corr.csvとの整合性を優先する
# 詳細な検証は antiberty_validation.csv を参照
ANTIBERTY_PAPER_VALUES = {
    # Expression (Table 3)
    ("antiberty", "expression", "Koenig2017_g6_er"): 0.24,
    ("antiberty", "expression", "gsk2023_AM14_exp"): 0.12,  # CA1
    ("antiberty", "expression", "gsk2023_D25_exp"): 0.60,   # CA2
    ("antiberty", "expression", "gsk2023_MOTA_exp"): 0.55,  # CA3
    ("antiberty", "expression", "gsk2023_RSB1_exp"): 0.39,  # CA4
    ("antiberty", "expression", "Wittrup2017_CST_HEK"): 0.05,

    # Tm (Table 4)
    ("antiberty", "tm", "Hie2022_C143_Tm"): -1.00,
    ("antiberty", "tm", "Hie2022_mAb114_Tm"): -0.11,
    ("antiberty", "tm", "Hie2022_mAb114UCA_Tm"): -0.43,
    ("antiberty", "tm", "Hie2022_MEDI8852_Tm"): -1.00,
    ("antiberty", "tm", "Hie2022_MEDI8852UCA_Tm"): -0.09,
    ("antiberty", "tm", "Hie2022_REGN10987_Tm"): 0.29,
    ("antiberty", "tm", "Hie2022_S309_Tm"): 0.25,
    ("antiberty", "tm", "Rosace2023_Adalimumab_Tm"): -0.41,
    ("antiberty", "tm", "Rosace2023_CR3022_Tm"): -0.09,
    ("antiberty", "tm", "Rosace2023_Golimumab_Tm"): 0.40,
    ("antiberty", "tm", "Wittrup2017_CST_Tm"): 0.34,
    ("antiberty", "tm", "gsk2023_AM14_Tm"): 0.38,
    ("antiberty", "tm", "gsk2023_D25_Tm"): 0.35,
    ("antiberty", "tm", "gsk2023_MOTA_Tm"): 0.81,
    ("antiberty", "tm", "gsk2023_RSB1_Tm"): 0.42,

    # Binding (Table 4)
    ("antiberty", "binding", "Hie2022_C143_Kd"): -0.02,
    ("antiberty", "binding", "Hie2022_mAb114_Kd"): -0.19,
    ("antiberty", "binding", "Hie2022_MEDI8852_Kd"): -0.27,
    ("antiberty", "binding", "Hie2022_MEDI8852UCA_Kd"): -0.11,
    ("antiberty", "binding", "Hie2022_REGN10987_Kd"): -0.45,
    ("antiberty", "binding", "Hie2022_S309_Kd"): -0.20,
    ("antiberty", "binding", "Koenig2017_g6_Kd"): 0.11,
    ("antiberty", "binding", "Rosace2023_Adalimumab_Kd"): -0.43,
    ("antiberty", "binding", "Rosace2023_CR3022_Kd"): 0.03,
    ("antiberty", "binding", "Rosace2023_Golimumab_Kd"): 0.80,
    ("antiberty", "binding", "Shanehsazzadeh2023_trastuzumab_multi_kd"): 0.33,
    ("antiberty", "binding", "Shanehsazzadeh2023_trastuzumab_zero_kd"): 0.14,
    ("antiberty", "binding", "Warszawski2019_d44_Kd"): 0.10,
    ("antiberty", "binding", "gsk2023_AM14_Kd"): -0.42,
    ("antiberty", "binding", "gsk2023_D25_Kd"): -0.37,
    ("antiberty", "binding", "gsk2023_MOTA_Kd"): 0.24,
    ("antiberty", "binding", "gsk2023_RSB1_Kd"): -0.35,

    # Aggregation (Table 4)
    ("antiberty", "aggregation", "Wittrup2017_CST_ACSINS"): -0.11,
    ("antiberty", "aggregation", "Wittrup2017_CST_CSI"): -0.10,
    ("antiberty", "aggregation", "Wittrup2017_CST_HIC"): -0.04,
    ("antiberty", "aggregation", "Wittrup2017_CST_SAS"): 0.02,
    ("antiberty", "aggregation", "Wittrup2017_CST_SGAC"): -0.17,
    ("antiberty", "aggregation", "Wittrup2017_CST_SMAC"): -0.01,

    # Immunogenicity (Table 4)
    ("antiberty", "immunogenicity", "Prihoda2021_mAb_immunogenicity"): 0.13,

    # Polyreactivity (Table 4)
    ("antiberty", "polyreactivity", "Rosace2023_Adalimumab_CIC"): -0.19,
    ("antiberty", "polyreactivity", "Rosace2023_CR3022_CIC"): -0.43,
    ("antiberty", "polyreactivity", "Rosace2023_Golimumab_CIC"): -0.10,
    ("antiberty", "polyreactivity", "Wittrup2017_CST_CIC"): 0.15,
    ("antiberty", "polyreactivity", "Wittrup2017_CST_ELISA"): 0.12,
    ("antiberty", "polyreactivity", "Wittrup2017_CST_PSR"): 0.16,
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_score_path(model: str, category: str, folder: str) -> Path:
    """スコアCSVのパスを取得"""
    if model in MODELS_WITH_SUBDIR:
        subdir = MODELS_WITH_SUBDIR[model]
        return SCORE_DIR / model / subdir / category / folder / f"{folder}_ppl.csv"
    else:
        return SCORE_DIR / model / category / folder / f"{folder}_ppl.csv"


def get_data_path(category: str, folder: str) -> Path:
    """データCSVのパスを取得"""
    return DATA_DIR / category / f"{folder}.csv"


def load_master_table() -> pd.DataFrame:
    """マスターテーブルを読み込み"""
    df = pd.read_csv(MASTER_TABLE)
    return df


def compute_correlation(score_df: pd.DataFrame, min_n: int = 3) -> dict:
    """
    average_perplexity vs fitness の相関を計算

    Note: 既存の_corr.csvファイルと論文の補足テーブルは
    corr(perplexity, fitness)の生の値を報告している。
    Figure 2のheatmapでのみ符号補正が適用される。

    Returns:
        dict with keys: spearman, pearson, spearman_p, pearson_p, n
    """
    # 有効なデータのみ使用
    valid = score_df[['average_perplexity', 'fitness']].dropna()
    n = len(valid)

    if n < min_n:
        return {
            'spearman': np.nan,
            'pearson': np.nan,
            'spearman_p': np.nan,
            'pearson_p': np.nan,
            'n': n
        }

    try:
        # perplexityとfitnessの生の相関
        s_corr, s_pval = spearmanr(valid['average_perplexity'], valid['fitness'])
        p_corr, p_pval = pearsonr(valid['average_perplexity'], valid['fitness'])
    except:
        return {
            'spearman': np.nan,
            'pearson': np.nan,
            'spearman_p': np.nan,
            'pearson_p': np.nan,
            'n': n
        }

    return {
        'spearman': s_corr,
        'pearson': p_corr,
        'spearman_p': s_pval,
        'pearson_p': p_pval,
        'n': n
    }


# ============================================================================
# Step 1: Data Integrity Check
# ============================================================================

def verify_data_counts():
    """データ数の整合性確認"""
    print("\n" + "="*80)
    print("Step 1: データ数の整合性確認")
    print("="*80)

    master_df = load_master_table()
    results = []

    for _, row in master_df.iterrows():
        category = row['Category']
        folder = row['Folder_name']
        expected_count = row['Fasta_count']

        # データCSVの行数
        data_path = get_data_path(category, folder)
        if data_path.exists():
            data_df = pd.read_csv(data_path)
            data_count = len(data_df)
        else:
            data_count = -1

        # 各モデルのスコアCSVの行数
        for model in MODELS:
            score_path = get_score_path(model, category, folder)

            if score_path.exists():
                score_df = pd.read_csv(score_path)
                score_count = len(score_df)
            else:
                score_count = -1

            results.append({
                'model': model,
                'category': category,
                'folder': folder,
                'expected_fasta': expected_count,
                'data_csv': data_count,
                'score_csv': score_count,
                'match': (data_count == score_count == expected_count) if (data_count > 0 and score_count > 0) else False,
                'missing': score_count < 0
            })

    results_df = pd.DataFrame(results)

    # 保存
    output_path = OUTPUT_DIR / "verification_report.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ 検証レポート保存: {output_path}")

    # サマリー
    total_pairs = len(results_df)
    missing_pairs = results_df['missing'].sum()
    mismatch_pairs = ((~results_df['match']) & (~results_df['missing'])).sum()

    print(f"\n総ペア数: {total_pairs}")
    print(f"欠損ペア: {missing_pairs} ({missing_pairs/total_pairs*100:.1f}%)")
    print(f"データ数不一致: {mismatch_pairs} ({mismatch_pairs/total_pairs*100:.1f}%)")

    if mismatch_pairs > 0:
        print("\n⚠ データ数不一致のペア:")
        mismatch_df = results_df[(~results_df['match']) & (~results_df['missing'])]
        for _, row in mismatch_df.head(10).iterrows():
            print(f"  {row['model']:15s} {row['category']:15s} {row['folder']:40s} "
                  f"expected={row['expected_fasta']:4d} data={row['data_csv']:4d} score={row['score_csv']:4d}")

    return results_df


# ============================================================================
# Step 2: Correlation Calculation
# ============================================================================

def calculate_correlations():
    """全モデル×データセットの相関を計算（生の値）"""
    print("\n" + "="*80)
    print("Step 2: 相関計算（生の値）")
    print("="*80)

    master_df = load_master_table()
    results = []

    for _, row in master_df.iterrows():
        category = row['Category']
        folder = row['Folder_name']
        antibody_set = row['Antibody_set']
        metric = row['Metric']

        for model in MODELS:
            score_path = get_score_path(model, category, folder)

            if not score_path.exists():
                results.append({
                    'model': model,
                    'category': category,
                    'folder': folder,
                    'antibody_set': antibody_set,
                    'metric': metric,
                    'spearman_raw': np.nan,
                    'pearson_raw': np.nan,
                    'spearman_p': np.nan,
                    'pearson_p': np.nan,
                    'n': 0
                })
                continue

            # スコアCSV読み込み（fitnessカラム含む）
            try:
                score_df = pd.read_csv(score_path)

                # fitnessカラムが存在するか確認
                if 'fitness' not in score_df.columns:
                    print(f"⚠ Warning: 'fitness' column missing in {score_path}")
                    results.append({
                        'model': model,
                        'category': category,
                        'folder': folder,
                        'antibody_set': antibody_set,
                        'metric': metric,
                        'spearman_raw': np.nan,
                        'pearson_raw': np.nan,
                        'spearman_p': np.nan,
                        'pearson_p': np.nan,
                        'n': 0
                    })
                    continue

                # 相関計算
                corr_result = compute_correlation(score_df)
            except Exception as e:
                print(f"⚠ Error loading {score_path}: {e}")
                results.append({
                    'model': model,
                    'category': category,
                    'folder': folder,
                    'antibody_set': antibody_set,
                    'metric': metric,
                    'spearman_raw': np.nan,
                    'pearson_raw': np.nan,
                    'spearman_p': np.nan,
                    'pearson_p': np.nan,
                    'n': 0
                })
                continue

            results.append({
                'model': model,
                'category': category,
                'folder': folder,
                'antibody_set': antibody_set,
                'metric': metric,
                'spearman_raw': corr_result['spearman'],
                'pearson_raw': corr_result['pearson'],
                'spearman_p': corr_result['spearman_p'],
                'pearson_p': corr_result['pearson_p'],
                'n': corr_result['n']
            })

    results_df = pd.DataFrame(results)

    # 保存
    output_path = OUTPUT_DIR / "correlations_raw.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ 生の相関値保存: {output_path}")

    # 統計
    valid_count = results_df['spearman_raw'].notna().sum()
    print(f"\n有効な相関値: {valid_count}/{len(results_df)}")

    return results_df


# ============================================================================
# Step 3: Sign Correction and Validation
# ============================================================================

def validate_antiberty(corr_df: pd.DataFrame):
    """AntiBERTy計算値を論文値と比較"""
    print("\n" + "="*80)
    print("Step 3: AntiBERTy論文値との検証")
    print("="*80)

    validation_results = []

    for (model, category, folder), paper_value in ANTIBERTY_PAPER_VALUES.items():
        # 計算値を取得
        match = corr_df[
            (corr_df['model'] == model) &
            (corr_df['category'] == category) &
            (corr_df['folder'] == folder)
        ]

        if len(match) == 0:
            calculated_value = np.nan
        else:
            calculated_value = match.iloc[0]['spearman_raw']

        diff = calculated_value - paper_value if not np.isnan(calculated_value) else np.nan

        validation_results.append({
            'category': category,
            'folder': folder,
            'paper_spearman': paper_value,
            'calculated_spearman': calculated_value,
            'difference': diff,
            'match': abs(diff) < 0.05 if not np.isnan(diff) else False
        })

    validation_df = pd.DataFrame(validation_results)

    # 保存
    output_path = OUTPUT_DIR / "antiberty_validation.csv"
    validation_df.to_csv(output_path, index=False)
    print(f"\n✓ 検証結果保存: {output_path}")

    # サマリー
    valid_comparisons = validation_df['difference'].notna().sum()
    matches = validation_df['match'].sum()

    print(f"\n比較可能: {valid_comparisons}/{len(validation_df)}")
    print(f"一致（±0.05以内）: {matches}/{valid_comparisons}")

    if matches < valid_comparisons:
        print("\n⚠ 不一致のデータセット:")
        mismatch = validation_df[~validation_df['match'] & validation_df['difference'].notna()]
        for _, row in mismatch.head(10).iterrows():
            print(f"  {row['category']:15s} {row['folder']:45s} "
                  f"paper={row['paper_spearman']:6.2f} calc={row['calculated_spearman']:6.2f} "
                  f"diff={row['difference']:6.3f}")

    return validation_df


def apply_sign_correction(corr_df: pd.DataFrame):
    """符号補正を適用"""
    print("\n" + "="*80)
    print("Step 3 (続き): 符号補正の適用")
    print("="*80)

    # 符号補正後の値を計算
    corr_df['spearman_corrected'] = corr_df.apply(
        lambda row: row['spearman_raw'] * SIGN_CORRECTION.get(row['category'], 1),
        axis=1
    )
    corr_df['pearson_corrected'] = corr_df.apply(
        lambda row: row['pearson_raw'] * SIGN_CORRECTION.get(row['category'], 1),
        axis=1
    )

    # 保存
    output_path = OUTPUT_DIR / "correlations_corrected.csv"
    corr_df.to_csv(output_path, index=False)
    print(f"\n✓ 符号補正後の相関値保存: {output_path}")

    # 符号補正の影響を表示
    print("\n符号補正ルール:")
    for category, sign in SIGN_CORRECTION.items():
        sign_str = "反転" if sign == -1 else "そのまま"
        print(f"  {category:15s}: {sign_str}")

    return corr_df


# ============================================================================
# Step 4: Per-Task Summary
# ============================================================================

def summarize_per_task(corr_df: pd.DataFrame):
    """タスク別の平均・標準偏差を計算"""
    print("\n" + "="*80)
    print("Step 4: タスク別サマリー")
    print("="*80)

    # タスク別に集計（符号補正後の値を使用）
    summary = corr_df.groupby(['model', 'category']).agg({
        'spearman_corrected': ['mean', 'std', 'count'],
        'pearson_corrected': ['mean', 'std', 'count']
    }).reset_index()

    # カラム名を整理
    summary.columns = ['model', 'category',
                      'spearman_mean', 'spearman_std', 'spearman_count',
                      'pearson_mean', 'pearson_std', 'pearson_count']

    # 保存
    output_path = OUTPUT_DIR / "correlations_per_task.csv"
    summary.to_csv(output_path, index=False)
    print(f"\n✓ タスク別サマリー保存: {output_path}")

    # タスク別の平均を表示
    print("\nタスク別平均（Spearman、符号補正後）:")
    pivot = summary.pivot(index='model', columns='category', values='spearman_mean')
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "  NaN"))

    return summary


# ============================================================================
# Step 5: Model Ranking
# ============================================================================

def calculate_model_ranks(corr_df: pd.DataFrame):
    """モデルランキングを計算（符号補正後の値で順位付け）"""
    print("\n" + "="*80)
    print("Step 5: モデルランキング")
    print("="*80)

    # 各データセットでモデルをランク付け（大きい相関=良い予測=上位ランク=小さい数字）
    rank_data = []

    for category in corr_df['category'].unique():
        for folder in corr_df['folder'].unique():
            subset = corr_df[
                (corr_df['category'] == category) &
                (corr_df['folder'] == folder)
            ].copy()

            if len(subset) == 0:
                continue

            # Spearmanランク（降順: 大きい値=1位）
            subset['spearman_rank'] = subset['spearman_corrected'].rank(ascending=False, method='average')
            # Pearsonランク
            subset['pearson_rank'] = subset['pearson_corrected'].rank(ascending=False, method='average')

            for _, row in subset.iterrows():
                rank_data.append({
                    'model': row['model'],
                    'category': category,
                    'folder': folder,
                    'spearman_rank': row['spearman_rank'],
                    'pearson_rank': row['pearson_rank']
                })

    rank_df = pd.DataFrame(rank_data)

    # モデル別、カテゴリ別の平均ランク
    rank_summary = rank_df.groupby(['model', 'category']).agg({
        'spearman_rank': 'mean',
        'pearson_rank': 'mean'
    }).reset_index()

    # 全体平均ランク
    overall_rank = rank_df.groupby('model').agg({
        'spearman_rank': 'mean',
        'pearson_rank': 'mean'
    }).reset_index()
    overall_rank = overall_rank.sort_values('spearman_rank')

    # 保存
    output_path = OUTPUT_DIR / "model_ranks.csv"
    rank_summary.to_csv(output_path, index=False)
    print(f"\n✓ モデルランキング保存: {output_path}")

    print("\n全体平均ランク（Spearman、小さい方が良い）:")
    for _, row in overall_rank.iterrows():
        print(f"  {row['model']:15s}: {row['spearman_rank']:.2f}")

    return rank_df, rank_summary, overall_rank


# ============================================================================
# Step 6: Visualization
# ============================================================================

def plot_heatmap(corr_df: pd.DataFrame, method: str = 'spearman'):
    """ヒートマップ（Figure 2スタイル）"""
    print(f"\n生成中: {method} heatmap...")

    # ピボットテーブル作成（符号補正後）
    pivot = corr_df.pivot(
        index='model',
        columns='folder',
        values=f'{method}_corrected'
    )

    # カテゴリごとにソート
    category_map = corr_df[['folder', 'category']].drop_duplicates().set_index('folder')['category'].to_dict()

    # カラムをカテゴリ順にソート
    category_order = ['expression', 'tm', 'binding', 'aggregation', 'immunogenicity', 'polyreactivity']
    sorted_cols = []
    for cat in category_order:
        cat_cols = [col for col in pivot.columns if category_map.get(col) == cat]
        sorted_cols.extend(sorted(cat_cols))

    pivot = pivot[sorted_cols]

    # プロット
    fig, ax = plt.subplots(figsize=(24, 8))

    # ヒートマップ
    sns.heatmap(
        pivot,
        cmap='RdBu_r',  # 青=正=良い
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        cbar_kws={'label': f'{method.capitalize()} Correlation (Sign-Corrected)'},
        ax=ax,
        linewidths=0.5,
        linecolor='lightgray'
    )

    # カテゴリ境界線を追加
    current_cat = None
    for i, col in enumerate(pivot.columns):
        cat = category_map.get(col)
        if cat != current_cat and current_cat is not None:
            ax.axvline(i, color='black', linewidth=2)
        current_cat = cat

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'FLAb1 Benchmark: {method.capitalize()} Correlation Heatmap\n'
                 f'(Sign-corrected: positive = better prediction)',
                 fontsize=14, pad=20)

    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"heatmap_{method}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_per_task_barplot(summary_df: pd.DataFrame):
    """タスク別バープロット（Spearman、モデルごとに色分け）"""
    print("\n生成中: per-task barplot (Spearman)...")

    # モデル順序を固定
    all_models = sorted(summary_df['model'].unique())

    # 手法別に色を定義（似た手法は同じ系統の色）
    model_colors = {
        # Language models (PLM系) - 青系
        'ablang2': '#1f77b4',      # 濃い青
        'antiberty': '#6baed6',    # 中間の青
        'esm2': '#9ecae1',         # 薄い青
        'ism': '#c6dbef',          # とても薄い青
        # Structure-based - 赤/オレンジ系
        'esmif': '#d62728',        # 赤
        'mpnn': '#ff7f0e',         # オレンジ
        'pyrosetta': '#ff9896',    # 薄い赤
        # Generative (SA-based) - 緑系
        'sablm_nostr': '#2ca02c',  # 濃い緑
        'sablm_str': '#98df8a',    # 薄い緑
    }

    categories = summary_df['category'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(sorted(categories)):
        ax = axes[i]
        data = summary_df[summary_df['category'] == category]

        # モデル順序でソート
        data = data.set_index('model').reindex(all_models).reset_index()

        x = np.arange(len(all_models))
        bar_colors = [model_colors[model] for model in all_models]

        means = data['spearman_mean'].fillna(0)
        stds = data['spearman_std'].fillna(0)

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Spearman', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # 未使用サブプロットを非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb1: Per-Task Average Correlation (Sign-Corrected, Spearman)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_barplot_spearman.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_per_task_barplot_pearson(summary_df: pd.DataFrame):
    """タスク別バープロット（Pearson、モデルごとに色分け）"""
    print("\n生成中: per-task barplot (Pearson)...")

    # モデル順序を固定
    all_models = sorted(summary_df['model'].unique())

    # 手法別に色を定義
    model_colors = {
        'ablang2': '#1f77b4', 'antiberty': '#6baed6', 'esm2': '#9ecae1', 'ism': '#c6dbef',
        'esmif': '#d62728', 'mpnn': '#ff7f0e', 'pyrosetta': '#ff9896',
        'sablm_nostr': '#2ca02c', 'sablm_str': '#98df8a',
    }

    categories = summary_df['category'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(sorted(categories)):
        ax = axes[i]
        data = summary_df[summary_df['category'] == category]

        # モデル順序でソート
        data = data.set_index('model').reindex(all_models).reset_index()

        x = np.arange(len(all_models))
        bar_colors = [model_colors[model] for model in all_models]

        means = data['pearson_mean'].fillna(0)
        stds = data['pearson_std'].fillna(0)

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Pearson', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # 未使用サブプロットを非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb1: Per-Task Average Correlation (Sign-Corrected, Pearson)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_barplot_pearson.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_per_task_barplot_abs(corr_df: pd.DataFrame):
    """タスク別バープロット（絶対値、論文スタイル、モデルごとに色分け）"""
    print("\n生成中: per-task barplot (Absolute Spearman)...")

    # 絶対値で集計
    corr_df_abs = corr_df.copy()
    corr_df_abs['spearman_abs'] = corr_df_abs['spearman_raw'].abs()

    summary_abs = corr_df_abs.groupby(['model', 'category']).agg({
        'spearman_abs': ['mean', 'std', 'count']
    }).reset_index()
    summary_abs.columns = ['model', 'category', 'abs_mean', 'abs_std', 'abs_count']

    # モデル順序を固定
    all_models = sorted(summary_abs['model'].unique())

    # 手法別に色を定義
    model_colors = {
        'ablang2': '#1f77b4', 'antiberty': '#6baed6', 'esm2': '#9ecae1', 'ism': '#c6dbef',
        'esmif': '#d62728', 'mpnn': '#ff7f0e', 'pyrosetta': '#ff9896',
        'sablm_nostr': '#2ca02c', 'sablm_str': '#98df8a',
    }

    categories = summary_abs['category'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(sorted(categories)):
        ax = axes[i]
        data = summary_abs[summary_abs['category'] == category]

        # モデル順序でソート
        data = data.set_index('model').reindex(all_models).reset_index()

        x = np.arange(len(all_models))
        bar_colors = [model_colors[model] for model in all_models]

        # NaNを0として扱う（欠損データ用）
        means = data['abs_mean'].fillna(0)
        stds = data['abs_std'].fillna(0)

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean |Spearman|', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 0.8)  # より見やすい範囲

    # 未使用サブプロットを非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb1: Per-Task Average |Correlation| (Absolute Spearman, Paper Style)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_barplot_abs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_per_task_boxplot(corr_df: pd.DataFrame):
    """タスク別ボックスプロット"""
    print("\n生成中: per-task boxplot...")

    categories = corr_df['category'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(sorted(categories)):
        ax = axes[i]
        data = corr_df[corr_df['category'] == category]

        # ボックスプロット
        models = sorted(data['model'].unique())
        plot_data = [data[data['model'] == m]['spearman_corrected'].dropna().values for m in models]

        bp = ax.boxplot(plot_data, labels=models, patch_artist=True)

        # ジッタープロット
        for j, (model, values) in enumerate(zip(models, plot_data)):
            if len(values) > 0:
                x = np.random.normal(j+1, 0.04, len(values))
                ax.scatter(x, values, alpha=0.4, s=20)

        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title(category.capitalize())
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # 未使用サブプロットを非表示
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb1: Per-Task Correlation Distribution (Sign-Corrected)', fontsize=16)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_rank_barplot(rank_summary: pd.DataFrame, overall_rank: pd.DataFrame):
    """ランキングバープロット（2つの独立した図）"""

    # 手法別の色定義
    model_colors = {
        'ablang2': '#1f77b4', 'antiberty': '#6baed6', 'esm2': '#9ecae1', 'ism': '#c6dbef',
        'esmif': '#d62728', 'mpnn': '#ff7f0e', 'pyrosetta': '#ff9896',
        'sablm_nostr': '#2ca02c', 'sablm_str': '#98df8a',
    }

    # 図1: タスク別平均ランク（横幅広く）
    print("\n生成中: rank barplot (per task)...")

    categories = sorted(rank_summary['category'].unique())
    models = sorted(rank_summary['model'].unique())

    fig, ax = plt.subplots(figsize=(20, 8))  # 横幅を広く

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        ranks = []
        for cat in categories:
            match = rank_summary[(rank_summary['model'] == model) & (rank_summary['category'] == cat)]
            if len(match) > 0:
                ranks.append(match.iloc[0]['spearman_rank'])
            else:
                ranks.append(np.nan)
        ax.bar(x + i*width, ranks, width, label=model, color=model_colors.get(model, 'gray'),
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Task Category', fontsize=12)
    ax.set_ylabel('Average Rank (lower = better)', fontsize=12)
    ax.set_title('FLAb1: Average Rank by Task Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models)-1) / 2)
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=0, ha='center', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, ncol=3, frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "rank_barplot_per_task.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()

    # 図2: 全体平均ランク
    print("\n生成中: rank barplot (overall)...")

    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_overall = overall_rank.sort_values('spearman_rank')
    colors_list = [model_colors.get(model, 'gray') for model in sorted_overall['model']]

    ax.barh(range(len(sorted_overall)), sorted_overall['spearman_rank'],
            color=colors_list, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_overall)))
    ax.set_yticklabels(sorted_overall['model'], fontsize=11)
    ax.set_xlabel('Average Rank (lower = better)', fontsize=12)
    ax.set_title('FLAb1: Overall Average Rank', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "rank_barplot_overall.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """メイン処理"""
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FLAb1 ベンチマーク評価スクリプト")
    print("="*80)
    print(f"\n対象モデル数: {len(MODELS)}")
    print(f"対象データセット数: 52 (予定)")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")

    # Step 1: データ数確認
    verification_df = verify_data_counts()

    # Step 2: 相関計算（生の値）
    corr_df = calculate_correlations()

    # Step 3: 検証と符号補正
    validation_df = validate_antiberty(corr_df)
    corr_df = apply_sign_correction(corr_df)

    # Step 4: タスク別サマリー
    summary_df = summarize_per_task(corr_df)

    # Step 5: ランキング
    rank_df, rank_summary, overall_rank = calculate_model_ranks(corr_df)

    # Step 6: 可視化
    print("\n" + "="*80)
    print("Step 6: 可視化")
    print("="*80)

    plot_heatmap(corr_df, 'spearman')
    plot_heatmap(corr_df, 'pearson')
    plot_per_task_barplot(summary_df)
    plot_per_task_barplot_pearson(summary_df)
    plot_per_task_barplot_abs(corr_df)
    plot_per_task_boxplot(corr_df)
    plot_rank_barplot(rank_summary, overall_rank)

    print("\n" + "="*80)
    print("✓ 全ての処理が完了しました")
    print("="*80)
    print(f"\n結果は以下に保存されました: {OUTPUT_DIR}")
    print("\n生成されたファイル:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
