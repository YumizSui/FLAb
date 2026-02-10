#!/usr/bin/env python3
"""
FLAb Few-shot ベンチマーク評価スクリプト

6モデル × 19データセット（N>=30、構造あり）の教師あり転移学習性能を評価:
- Few-shot (MLP regression)結果の集計
- Zero-shot (perplexity)との比較
- タスク別サマリー
- 可視化（ヒートマップ、バープロット、比較プロット）
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
FEWSHOT_DIR = ROOT / "fewshot"
SCORE_DIR = ROOT / "score"
DATA_DIR = ROOT / "data" / "FLAb_0"
OUTPUT_DIR = ROOT / "reports" / "fewshot_evaluation"
ELIGIBLE_DATASETS_CSV = ROOT / "data" / "fewshot_eligible_datasets.csv"

# 7モデル（few-shot対象）
MODELS = [
    "ablang2",
    "antiberty",
    "esm2",
    "ism",
    "sablm_nostr",
    "sablm_str",
    "onehot"
]

# esm2, ismは650Mサブディレクトリを使用（zero-shot用）
MODELS_WITH_SUBDIR = {
    "esm2": "650M",
    "ism": "650M"
}

# sablm_str → var2_str-0.4チェックポイントのスコアを使用
SABLM_VAR2_CHECKPOINT = "var2_str-0.4_nofocal_noaug_lora"
SABLM_VAR2_MODELS = {
    "sablm_str": ("sablm_var2", SABLM_VAR2_CHECKPOINT)
}
SABLM_VAR2_FEWSHOT_MODELS = {
    "sablm_str": ("sablm_str_var2", SABLM_VAR2_CHECKPOINT)
}

# 符号補正ルール（zero-shotとの比較用）
SIGN_CORRECTION = {
    "expression": -1,
    "tm": -1,
    "aggregation": -1,
    "binding": 1,
    "immunogenicity": 1,
    "polyreactivity": 1,
}

# モデル表示順序
MODEL_ORDER = [
    "onehot",
    "antiberty",
    "esm2",
    "ism",
    "ablang2",
    "sablm_nostr",
    "sablm_str"
]

# モデル色定義
MODEL_COLORS = {
    'onehot': '#969696',
    'ablang2': '#1f77b4',
    'antiberty': '#6baed6',
    'esm2': '#9ecae1',
    'ism': '#c6dbef',
    'sablm_nostr': '#2ca02c',
    'sablm_str': '#98df8a',
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_fewshot_results() -> pd.DataFrame:
    """Few-shot結果を読み込み"""
    print("\n" + "="*80)
    print("Few-shot結果の読み込み")
    print("="*80)

    results = []

    for model in MODELS:
        # sablm_str → sablm_str_var2/var2_str-0.4_nofocal_noaug_lora
        if model in SABLM_VAR2_FEWSHOT_MODELS:
            base_model, checkpoint = SABLM_VAR2_FEWSHOT_MODELS[model]
            model_dir = FEWSHOT_DIR / base_model / checkpoint
        else:
            model_dir = FEWSHOT_DIR / model

        if not model_dir.exists():
            print(f"⚠ Warning: {model_dir} not found")
            continue

        # 全カテゴリ・データセットを探索
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for dataset_dir in category_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                dataset = dataset_dir.name
                results_file = dataset_dir / "results.json"

                if not results_file.exists():
                    continue

                # JSONファイル読み込み
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    results.append({
                        'model': model,
                        'category': category,
                        'dataset': dataset,
                        'n_samples': data.get('n_samples', 0),
                        'embedding_dim': data.get('embedding_dim', 0),
                        'n_trials': data.get('n_trials', 0),
                        'mean_spearman': data.get('mean_spearman_rho', np.nan),
                        'std_spearman': data.get('std_spearman_rho', np.nan),
                        'mean_spearman_p': data.get('mean_spearman_p', np.nan),
                    })
                except Exception as e:
                    print(f"⚠ Error loading {results_file}: {e}")

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠ Warning: No few-shot results found")
        return df

    print(f"\n✓ {len(df)} few-shot results loaded")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Categories: {sorted(df['category'].unique())}")

    return df


def get_zeroshot_score_path(model: str, category: str, dataset: str) -> Path:
    """Zero-shotスコアCSVのパスを取得"""
    if model in SABLM_VAR2_MODELS:
        # sablm_str → sablm_var2/var2_str-0.4_nofocal_noaug_lora
        base_model, checkpoint = SABLM_VAR2_MODELS[model]
        return SCORE_DIR / base_model / checkpoint / category / dataset / f"{dataset}_ppl.csv"
    elif model in MODELS_WITH_SUBDIR:
        subdir = MODELS_WITH_SUBDIR[model]
        return SCORE_DIR / model / subdir / category / dataset / f"{dataset}_ppl.csv"
    else:
        return SCORE_DIR / model / category / dataset / f"{dataset}_ppl.csv"


def load_zeroshot_results(fewshot_df: pd.DataFrame) -> pd.DataFrame:
    """Zero-shot結果を読み込み（few-shotと同じデータセットのみ）"""
    print("\n" + "="*80)
    print("Zero-shot結果の読み込み")
    print("="*80)

    from scipy.stats import spearmanr

    results = []

    # few-shotで評価したmodel×datasetのみ対象
    for _, row in fewshot_df.iterrows():
        model = row['model']
        category = row['category']
        dataset = row['dataset']

        score_path = get_zeroshot_score_path(model, category, dataset)

        if not score_path.exists():
            print(f"⚠ Warning: {score_path} not found")
            results.append({
                'model': model,
                'category': category,
                'dataset': dataset,
                'spearman_zeroshot': np.nan,
            })
            continue

        # スコアCSV読み込み
        try:
            score_df = pd.read_csv(score_path)

            if 'average_perplexity' not in score_df.columns or 'fitness' not in score_df.columns:
                print(f"⚠ Warning: Missing columns in {score_path}")
                results.append({
                    'model': model,
                    'category': category,
                    'dataset': dataset,
                    'spearman_zeroshot': np.nan,
                })
                continue

            # 有効なデータのみ使用
            valid = score_df[['average_perplexity', 'fitness']].dropna()

            if len(valid) < 3:
                results.append({
                    'model': model,
                    'category': category,
                    'dataset': dataset,
                    'spearman_zeroshot': np.nan,
                })
                continue

            # Spearman相関計算（perplexity vs fitness）
            rho, _ = spearmanr(valid['average_perplexity'], valid['fitness'])

            results.append({
                'model': model,
                'category': category,
                'dataset': dataset,
                'spearman_zeroshot': rho,
            })

        except Exception as e:
            print(f"⚠ Error loading {score_path}: {e}")
            results.append({
                'model': model,
                'category': category,
                'dataset': dataset,
                'spearman_zeroshot': np.nan,
            })

    df = pd.DataFrame(results)

    print(f"\n✓ {len(df)} zero-shot results loaded")
    valid_count = df['spearman_zeroshot'].notna().sum()
    print(f"  Valid correlations: {valid_count}/{len(df)}")

    return df


def merge_fewshot_zeroshot(fewshot_df: pd.DataFrame, zeroshot_df: pd.DataFrame) -> pd.DataFrame:
    """Few-shotとZero-shotを結合"""
    merged = pd.merge(
        fewshot_df,
        zeroshot_df,
        on=['model', 'category', 'dataset'],
        how='left'
    )

    # 符号補正を適用（zero-shot側のみ）
    merged['spearman_zeroshot_corrected'] = merged.apply(
        lambda row: row['spearman_zeroshot'] * SIGN_CORRECTION.get(row['category'], 1),
        axis=1
    )

    # Few-shotは生のfitnessで学習しているため符号補正不要
    # (fitnessが既に「高い=良い」に処理済みなら常に正の相関が正しい)
    merged['spearman_fewshot_corrected'] = merged['mean_spearman']

    return merged


# ============================================================================
# Analysis
# ============================================================================

def compute_task_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """タスク別サマリー"""
    print("\n" + "="*80)
    print("タスク別サマリー")
    print("="*80)

    summary = merged_df.groupby(['model', 'category']).agg({
        'spearman_fewshot_corrected': ['mean', 'std', 'count'],
        'spearman_zeroshot_corrected': ['mean', 'std', 'count']
    }).reset_index()

    summary.columns = [
        'model', 'category',
        'fewshot_mean', 'fewshot_std', 'fewshot_count',
        'zeroshot_mean', 'zeroshot_std', 'zeroshot_count'
    ]

    # 改善率を計算
    summary['improvement'] = summary['fewshot_mean'] - summary['zeroshot_mean']

    # 保存
    output_path = OUTPUT_DIR / "task_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"\n✓ タスク別サマリー保存: {output_path}")

    return summary


def compute_overall_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """モデル全体のサマリー"""
    print("\n" + "="*80)
    print("モデル全体サマリー")
    print("="*80)

    summary = merged_df.groupby('model').agg({
        'spearman_fewshot_corrected': ['mean', 'std', 'count'],
        'spearman_zeroshot_corrected': ['mean', 'std', 'count']
    }).reset_index()

    summary.columns = [
        'model',
        'fewshot_mean', 'fewshot_std', 'fewshot_count',
        'zeroshot_mean', 'zeroshot_std', 'zeroshot_count'
    ]

    # 改善率
    summary['improvement'] = summary['fewshot_mean'] - summary['zeroshot_mean']

    # 並び替え（few-shot性能順）
    summary = summary.sort_values('fewshot_mean', ascending=False)

    # 保存
    output_path = OUTPUT_DIR / "overall_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"\n✓ 全体サマリー保存: {output_path}")

    print("\nモデル全体平均（Spearman、符号補正後）:")
    for _, row in summary.iterrows():
        print(f"  {row['model']:15s}: Few-shot={row['fewshot_mean']:6.3f}  Zero-shot={row['zeroshot_mean']:6.3f}  Δ={row['improvement']:+6.3f}")

    return summary


# ============================================================================
# Ranking
# ============================================================================

def calculate_model_ranks(merged_df: pd.DataFrame):
    """モデルランキングを計算（Few-shotとZero-shotの両方）"""
    print("\n" + "="*80)
    print("モデルランキング")
    print("="*80)

    # 各データセットでモデルをランク付け
    rank_data = []

    for category in merged_df['category'].unique():
        for dataset in merged_df['dataset'].unique():
            subset = merged_df[
                (merged_df['category'] == category) &
                (merged_df['dataset'] == dataset)
            ].copy()

            if len(subset) == 0:
                continue

            # Few-shotランク（降順: 大きい値=1位）
            subset['fewshot_rank'] = subset['spearman_fewshot_corrected'].rank(ascending=False, method='average')
            # Zero-shotランク
            subset['zeroshot_rank'] = subset['spearman_zeroshot_corrected'].rank(ascending=False, method='average')

            for _, row in subset.iterrows():
                rank_data.append({
                    'model': row['model'],
                    'category': category,
                    'dataset': dataset,
                    'fewshot_rank': row['fewshot_rank'],
                    'zeroshot_rank': row['zeroshot_rank']
                })

    rank_df = pd.DataFrame(rank_data)

    # モデル別、カテゴリ別の平均ランク
    rank_summary = rank_df.groupby(['model', 'category']).agg({
        'fewshot_rank': 'mean',
        'zeroshot_rank': 'mean'
    }).reset_index()

    # 全体平均ランク
    overall_rank = rank_df.groupby('model').agg({
        'fewshot_rank': 'mean',
        'zeroshot_rank': 'mean'
    }).reset_index()
    overall_rank = overall_rank.sort_values('fewshot_rank')

    # 保存
    output_path = OUTPUT_DIR / "model_ranks.csv"
    rank_summary.to_csv(output_path, index=False)
    print(f"\n✓ モデルランキング保存: {output_path}")

    print("\n全体平均ランク（小さい方が良い）:")
    print("\nFew-shot:")
    for _, row in overall_rank.iterrows():
        print(f"  {row['model']:15s}: {row['fewshot_rank']:.2f}")
    print("\nZero-shot:")
    for _, row in overall_rank.sort_values('zeroshot_rank').iterrows():
        print(f"  {row['model']:15s}: {row['zeroshot_rank']:.2f}")

    return rank_df, rank_summary, overall_rank


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison_heatmap(merged_df: pd.DataFrame):
    """Few-shot vs Zero-shot比較ヒートマップ（2段）"""
    print("\n生成中: comparison heatmap...")

    # Few-shotピボット
    pivot_fewshot = merged_df.pivot(
        index='model',
        columns='dataset',
        values='spearman_fewshot_corrected'
    )

    # Zero-shotピボット
    pivot_zeroshot = merged_df.pivot(
        index='model',
        columns='dataset',
        values='spearman_zeroshot_corrected'
    )

    # カテゴリマップ
    category_map = merged_df[['dataset', 'category']].drop_duplicates().set_index('dataset')['category'].to_dict()

    # カテゴリ順にソート
    category_order = ['expression', 'tm', 'binding', 'aggregation', 'immunogenicity', 'polyreactivity']
    sorted_cols = []
    for cat in category_order:
        cat_cols = [col for col in pivot_fewshot.columns if category_map.get(col) == cat]
        sorted_cols.extend(sorted(cat_cols))

    pivot_fewshot = pivot_fewshot[sorted_cols]
    pivot_zeroshot = pivot_zeroshot[sorted_cols]

    # モデル順序
    model_order = [m for m in MODEL_ORDER if m in pivot_fewshot.index]
    pivot_fewshot = pivot_fewshot.reindex(model_order)
    pivot_zeroshot = pivot_zeroshot.reindex(model_order)

    # プロット（縦に2段）
    fig, axes = plt.subplots(2, 1, figsize=(24, 10))

    # Few-shot
    sns.heatmap(
        pivot_fewshot,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        cbar_kws={'label': 'Spearman Correlation'},
        ax=axes[0],
        linewidths=0.5,
        linecolor='lightgray'
    )
    axes[0].set_title('Few-shot (MLP Regression)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Model', fontsize=12)
    axes[0].set_xticklabels([])  # x軸ラベルは下段のみ

    # Zero-shot
    sns.heatmap(
        pivot_zeroshot,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        cbar_kws={'label': 'Spearman Correlation'},
        ax=axes[1],
        linewidths=0.5,
        linecolor='lightgray'
    )
    axes[1].set_title('Zero-shot (Perplexity)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, ha='right', fontsize=8)

    plt.suptitle('FLAb Few-shot vs Zero-shot Comparison\n(Sign-corrected: positive = better prediction)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "comparison_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_comparison_scatter(merged_df: pd.DataFrame):
    """Few-shot vs Zero-shot散布図（データセットごと）"""
    print("\n生成中: comparison scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    for model in MODEL_ORDER:
        data = merged_df[merged_df['model'] == model]

        ax.scatter(
            data['spearman_zeroshot_corrected'],
            data['spearman_fewshot_corrected'],
            label=model,
            color=MODEL_COLORS.get(model, 'gray'),
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    # 対角線
    ax.plot([-1, 1], [-1, 1], 'k--', linewidth=1, alpha=0.5, label='y=x')

    ax.set_xlabel('Zero-shot Spearman (Sign-corrected)', fontsize=12)
    ax.set_ylabel('Few-shot Spearman (Sign-corrected)', fontsize=12)
    ax.set_title('Few-shot vs Zero-shot Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "comparison_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_improvement_barplot(task_summary: pd.DataFrame):
    """タスク別改善率バープロット"""
    print("\n生成中: improvement barplot...")

    categories = sorted(task_summary['category'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        data = task_summary[task_summary['category'] == category]

        # モデル順序
        model_order_subset = [m for m in MODEL_ORDER if m in data['model'].values]
        data = data.set_index('model').reindex(model_order_subset).reset_index()

        x = np.arange(len(model_order_subset))
        width = 0.35

        bar_colors = [MODEL_COLORS.get(m, 'gray') for m in model_order_subset]

        # Zero-shot
        ax.bar(x - width/2, data['zeroshot_mean'], width, label='Zero-shot',
               color='lightgray', edgecolor='black', linewidth=0.5)

        # Few-shot
        ax.bar(x + width/2, data['fewshot_mean'], width, label='Few-shot',
               color=bar_colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_order_subset, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Spearman', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    # 未使用サブプロット
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb: Per-Task Improvement (Zero-shot → Few-shot)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "improvement_barplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_overall_barplot(overall_summary: pd.DataFrame):
    """モデル全体のバープロット"""
    print("\n生成中: overall barplot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # モデル順序
    model_order_subset = [m for m in MODEL_ORDER if m in overall_summary['model'].values]
    overall_summary = overall_summary.set_index('model').reindex(model_order_subset).reset_index()

    x = np.arange(len(model_order_subset))
    width = 0.35

    bar_colors = [MODEL_COLORS.get(m, 'gray') for m in model_order_subset]

    # Zero-shot
    ax.bar(x - width/2, overall_summary['zeroshot_mean'], width, label='Zero-shot',
           color='lightgray', edgecolor='black', linewidth=0.5,
           yerr=overall_summary['zeroshot_std'], capsize=5)

    # Few-shot
    ax.bar(x + width/2, overall_summary['fewshot_mean'], width, label='Few-shot',
           color=bar_colors, edgecolor='black', linewidth=0.5,
           yerr=overall_summary['fewshot_std'], capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order_subset, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Mean Spearman (across all datasets)', fontsize=12)
    ax.set_title('Overall Performance: Few-shot vs Zero-shot', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "overall_barplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_individual_barplots(task_summary: pd.DataFrame):
    """Few-shotとZero-shotの個別バープロット（2つの独立した図）"""

    # Few-shot単独
    print("\n生成中: fewshot individual barplot...")
    categories = sorted(task_summary['category'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        data = task_summary[task_summary['category'] == category]

        # モデル順序
        model_order_subset = [m for m in MODEL_ORDER if m in data['model'].values]
        data = data.set_index('model').reindex(model_order_subset).reset_index()

        x = np.arange(len(model_order_subset))
        bar_colors = [MODEL_COLORS.get(m, 'gray') for m in model_order_subset]

        means = data['fewshot_mean'].fillna(0)
        stds = data['fewshot_std'].fillna(0)

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order_subset, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Spearman', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # 未使用サブプロット
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb Few-shot: Per-Task Average Correlation',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_barplot_fewshot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()

    # Zero-shot単独
    print("\n生成中: zeroshot individual barplot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        data = task_summary[task_summary['category'] == category]

        # モデル順序
        model_order_subset = [m for m in MODEL_ORDER if m in data['model'].values]
        data = data.set_index('model').reindex(model_order_subset).reset_index()

        x = np.arange(len(model_order_subset))
        bar_colors = [MODEL_COLORS.get(m, 'gray') for m in model_order_subset]

        means = data['zeroshot_mean'].fillna(0)
        stds = data['zeroshot_std'].fillna(0)

        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order_subset, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Spearman', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # 未使用サブプロット
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb Zero-shot: Per-Task Average Correlation',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_barplot_zeroshot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_rank_barplot(rank_summary: pd.DataFrame, overall_rank: pd.DataFrame):
    """ランキングバープロット（2つの独立した図）"""

    # 図1: タスク別平均ランク（Few-shot）
    print("\n生成中: rank barplot (per task, fewshot)...")

    categories = sorted(rank_summary['category'].unique())
    models = sorted(rank_summary['model'].unique())

    fig, ax = plt.subplots(figsize=(20, 8))

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        ranks = []
        for cat in categories:
            match = rank_summary[(rank_summary['model'] == model) & (rank_summary['category'] == cat)]
            if len(match) > 0:
                ranks.append(match.iloc[0]['fewshot_rank'])
            else:
                ranks.append(np.nan)
        ax.bar(x + i*width, ranks, width, label=model, color=MODEL_COLORS.get(model, 'gray'),
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Task Category', fontsize=12)
    ax.set_ylabel('Average Rank (lower = better)', fontsize=12)
    ax.set_title('FLAb Few-shot: Average Rank by Task Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models)-1) / 2)
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=0, ha='center', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, ncol=3, frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "rank_barplot_per_task_fewshot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()

    # 図2: 全体平均ランク（Few-shot）
    print("\n生成中: rank barplot (overall, fewshot)...")

    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_overall = overall_rank.sort_values('fewshot_rank')
    colors_list = [MODEL_COLORS.get(model, 'gray') for model in sorted_overall['model']]

    ax.barh(range(len(sorted_overall)), sorted_overall['fewshot_rank'],
            color=colors_list, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_overall)))
    ax.set_yticklabels(sorted_overall['model'], fontsize=11)
    ax.set_xlabel('Average Rank (lower = better)', fontsize=12)
    ax.set_title('FLAb Few-shot: Overall Average Rank', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "rank_barplot_overall_fewshot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def plot_r2_barplot(merged_df: pd.DataFrame):
    """タスク別R²バープロット（負の相関を0にしてから二乗）"""
    print("\n生成中: R² barplot...")

    from scipy.stats import pearsonr

    # 各model×datasetでPearson相関を計算してR²化
    r2_results = []

    for _, row in merged_df.iterrows():
        model = row['model']
        category = row['category']
        dataset = row['dataset']

        # Few-shot R² (mean_spearmanをそのまま使う代わりに、Pearson相関想定で計算)
        # Note: Few-shotはSpearmanで評価されているが、ここではZero-shotとの比較のためPearsonベースで統一
        fewshot_corr = row['spearman_fewshot_corrected']  # 実際はSpearmanだが近似として使用
        fewshot_r2 = max(0, fewshot_corr) ** 2 if not np.isnan(fewshot_corr) else np.nan

        # Zero-shot: 元データからPearson相関を再計算してR²化
        score_path = get_zeroshot_score_path(model, category, dataset)
        if score_path.exists():
            try:
                score_df = pd.read_csv(score_path)
                valid = score_df[['average_perplexity', 'fitness']].dropna()
                if len(valid) >= 3:
                    p_corr, _ = pearsonr(valid['average_perplexity'], valid['fitness'])
                    # 符号補正を適用
                    p_corr_corrected = p_corr * SIGN_CORRECTION.get(category, 1)
                    zeroshot_r2 = max(0, p_corr_corrected) ** 2
                else:
                    zeroshot_r2 = np.nan
            except:
                zeroshot_r2 = np.nan
        else:
            zeroshot_r2 = np.nan

        r2_results.append({
            'model': model,
            'category': category,
            'dataset': dataset,
            'fewshot_r2': fewshot_r2,
            'zeroshot_r2': zeroshot_r2
        })

    r2_df = pd.DataFrame(r2_results)

    # タスク別に集計
    task_r2_summary = r2_df.groupby(['model', 'category']).agg({
        'fewshot_r2': ['mean', 'std'],
        'zeroshot_r2': ['mean', 'std']
    }).reset_index()
    task_r2_summary.columns = ['model', 'category', 'fewshot_r2_mean', 'fewshot_r2_std',
                                'zeroshot_r2_mean', 'zeroshot_r2_std']

    # プロット
    categories = sorted(task_r2_summary['category'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        data = task_r2_summary[task_r2_summary['category'] == category]

        # モデル順序
        model_order_subset = [m for m in MODEL_ORDER if m in data['model'].values]
        data = data.set_index('model').reindex(model_order_subset).reset_index()

        x = np.arange(len(model_order_subset))
        width = 0.35

        bar_colors = [MODEL_COLORS.get(m, 'gray') for m in model_order_subset]

        # Zero-shot
        ax.bar(x - width/2, data['zeroshot_r2_mean'], width, label='Zero-shot',
               color='lightgray', edgecolor='black', linewidth=0.5)

        # Few-shot
        ax.bar(x + width/2, data['fewshot_r2_mean'], width, label='Few-shot',
               color=bar_colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_order_subset, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Pearson R²', fontsize=10)
        ax.set_title(category.capitalize(), fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 0.5)

    # 未使用サブプロット
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('FLAb: Per-Task Pearson R² Comparison (Negative correlations set to 0)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_task_r2_barplot.png"
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
    print("FLAb Few-shot ベンチマーク評価スクリプト")
    print("="*80)
    print(f"\n対象モデル数: {len(MODELS)}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")

    # Step 1: Few-shot結果読み込み
    fewshot_df = load_fewshot_results()

    if len(fewshot_df) == 0:
        print("\n⚠ Error: No few-shot results found. Exiting.")
        return

    # Step 2: Zero-shot結果読み込み
    zeroshot_df = load_zeroshot_results(fewshot_df)

    # Step 3: マージ
    merged_df = merge_fewshot_zeroshot(fewshot_df, zeroshot_df)

    # 保存
    output_path = OUTPUT_DIR / "merged_results.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ マージ結果保存: {output_path}")

    # Step 4: サマリー
    task_summary = compute_task_summary(merged_df)
    overall_summary = compute_overall_summary(merged_df)

    # Step 5: ランキング
    rank_df, rank_summary, overall_rank = calculate_model_ranks(merged_df)

    # Step 6: 可視化
    print("\n" + "="*80)
    print("可視化")
    print("="*80)

    plot_comparison_heatmap(merged_df)
    plot_comparison_scatter(merged_df)
    plot_individual_barplots(task_summary)
    plot_improvement_barplot(task_summary)
    plot_overall_barplot(overall_summary)
    plot_rank_barplot(rank_summary, overall_rank)
    plot_r2_barplot(merged_df)

    print("\n" + "="*80)
    print("✓ 全ての処理が完了しました")
    print("="*80)
    print(f"\n結果は以下に保存されました: {OUTPUT_DIR}")
    print("\n生成されたファイル:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
