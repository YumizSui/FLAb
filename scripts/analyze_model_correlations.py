#!/usr/bin/env python
"""
Analyze correlations between model predictions and fitness values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import argparse


def load_results(output_dir, models, csv_name='jain2017biophysical_Tm_ppl.csv'):
    """Load results from all models"""
    data = {}

    for model in models:
        # Try new structure first: output_dir/model/csv_name
        result_file = Path(output_dir) / model / csv_name
        if not result_file.exists():
            # Fall back to old structure: output_dir/model_ppl.csv
            result_file = Path(output_dir) / f"{model}_ppl.csv"

        if result_file.exists():
            df = pd.read_csv(result_file)
            if 'status' in df.columns:
                df = df[df['status'] == 'success'].copy()

            # Look for perplexity column (try different names)
            ppl_col = None
            for col in ['average_perplexity', 'perplexity', 'ppl']:
                if col in df.columns:
                    ppl_col = col
                    break

            if ppl_col and len(df) > 0:
                data[model] = df[ppl_col].values
            else:
                print(f"Warning: No perplexity column found for {model}")
        else:
            print(f"Warning: Results not found for {model}")

    return data


def load_fitness_data(csv_path):
    """Load fitness (Tm) values"""
    df = pd.read_csv(csv_path)

    # Find Tm column
    tm_cols = [col for col in df.columns if 'Tm' in col or 'tm' in col or 'fitness' in col.lower()]
    if tm_cols:
        return df[tm_cols[0]].values
    else:
        print("Warning: Could not find Tm/fitness column")
        return None


def compute_correlations(data, fitness=None):
    """Compute pairwise correlations between models and fitness"""

    models = list(data.keys())
    n_models = len(models)

    # Add fitness if available
    if fitness is not None:
        models.append('Tm (Fitness)')
        data['Tm (Fitness)'] = fitness
        n_models += 1

    # Initialize correlation matrices
    spearman_corr = np.zeros((n_models, n_models))
    pearson_corr = np.zeros((n_models, n_models))
    spearman_pval = np.zeros((n_models, n_models))
    pearson_pval = np.zeros((n_models, n_models))

    # Compute correlations
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                spearman_corr[i, j] = 1.0
                pearson_corr[i, j] = 1.0
            else:
                # Get valid pairs (both non-NaN)
                valid = ~(np.isnan(data[model1]) | np.isnan(data[model2]))
                if valid.sum() > 2:
                    s_corr, s_pval = spearmanr(data[model1][valid], data[model2][valid])
                    p_corr, p_pval = pearsonr(data[model1][valid], data[model2][valid])

                    spearman_corr[i, j] = s_corr
                    pearson_corr[i, j] = p_corr
                    spearman_pval[i, j] = s_pval
                    pearson_pval[i, j] = p_pval
                else:
                    spearman_corr[i, j] = np.nan
                    pearson_corr[i, j] = np.nan

    return {
        'models': models,
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'spearman_pval': spearman_pval,
        'pearson_pval': pearson_pval
    }


def plot_correlation_heatmap(corr_results, output_dir, method='spearman'):
    """Plot correlation heatmap"""

    models = corr_results['models']
    corr_matrix = corr_results[method]
    pval_matrix = corr_results[f'{method}_pval']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=models,
        yticklabels=models,
        cbar_kws={'label': f'{method.capitalize()} Correlation'},
        ax=ax
    )

    # Add significance stars
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                pval = pval_matrix[i, j]
                if pval < 0.001:
                    text = '***'
                elif pval < 0.01:
                    text = '**'
                elif pval < 0.05:
                    text = '*'
                else:
                    text = ''

                if text:
                    ax.text(j + 0.5, i + 0.7, text, ha='center', va='center',
                           color='black', fontsize=8, fontweight='bold')

    plt.title(f'{method.capitalize()} Correlation Matrix\n(*, **, *** for p<0.05, 0.01, 0.001)',
              fontsize=14, pad=20)
    plt.tight_layout()

    # Save
    output_file = Path(output_dir) / f'correlation_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {method} correlation heatmap to: {output_file}")
    plt.close()


def plot_fitness_correlations(data, fitness, output_dir):
    """Plot scatter plots of model predictions vs fitness"""

    models = [m for m in data.keys() if m != 'Tm (Fitness)']
    n_models = len(models)

    # Determine grid size
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Get valid data
        valid = ~(np.isnan(data[model]) | np.isnan(fitness))
        x = data[model][valid]
        y = fitness[valid]

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30)

        # Compute correlation
        if len(x) > 2:
            s_corr, s_pval = spearmanr(x, y)
            p_corr, p_pval = pearsonr(x, y)

            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

            # Add correlation text
            text = f"Spearman ρ = {s_corr:.3f} (p={s_pval:.2e})\n"
            text += f"Pearson r = {p_corr:.3f} (p={p_pval:.2e})"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{model} (Perplexity)', fontsize=12)
        ax.set_ylabel('Tm (°C)', fontsize=12)
        ax.set_title(model, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Model Predictions vs Fitness (Tm)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    output_file = Path(output_dir) / 'fitness_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved fitness correlation plots to: {output_file}")
    plt.close()


def save_correlation_table(corr_results, output_dir):
    """Save correlation results as CSV"""

    models = corr_results['models']

    # Spearman correlations
    spearman_df = pd.DataFrame(
        corr_results['spearman'],
        index=models,
        columns=models
    )
    spearman_file = Path(output_dir) / 'correlation_spearman.csv'
    spearman_df.to_csv(spearman_file)
    print(f"Saved Spearman correlations to: {spearman_file}")

    # Pearson correlations
    pearson_df = pd.DataFrame(
        corr_results['pearson'],
        index=models,
        columns=models
    )
    pearson_file = Path(output_dir) / 'correlation_pearson.csv'
    pearson_df.to_csv(pearson_file)
    print(f"Saved Pearson correlations to: {pearson_file}")

    # Summary table (correlations with fitness only)
    if 'Tm (Fitness)' in models:
        fitness_idx = models.index('Tm (Fitness)')

        summary_data = []
        for i, model in enumerate(models):
            if model != 'Tm (Fitness)':
                summary_data.append({
                    'Model': model,
                    'Spearman_ρ': corr_results['spearman'][i, fitness_idx],
                    'Spearman_p': corr_results['spearman_pval'][i, fitness_idx],
                    'Pearson_r': corr_results['pearson'][i, fitness_idx],
                    'Pearson_p': corr_results['pearson_pval'][i, fitness_idx]
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Spearman_ρ', key=abs, ascending=False)

        summary_file = Path(output_dir) / 'fitness_correlation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved fitness correlation summary to: {summary_file}")

        # Print summary
        print("\n" + "="*60)
        print("Correlation with Fitness (Tm)")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze model correlations')
    parser.add_argument('--output-dir', type=str,
                       default='test_results',
                       help='Directory with model results')
    parser.add_argument('--csv-path', type=str,
                       default='data/thermostability/jain2017biophysical_Tm.csv',
                       help='Path to fitness data CSV')
    parser.add_argument('--models', nargs='+',
                       default=['antiberty', 'esm2', 'ablang2', 'ism', 'sablm_nostr', 'sablm_str',
                               'pyrosetta', 'esmif', 'mpnn'],
                       help='Models to analyze')

    args = parser.parse_args()

    print("="*60)
    print("Model Correlation Analysis")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"CSV path: {args.csv_path}")
    print(f"Models: {', '.join(args.models)}")
    print("")

    # Load data
    print("Loading model results...")
    data = load_results(args.output_dir, args.models)
    print(f"Loaded {len(data)} models")

    print("\nLoading fitness data...")
    fitness = load_fitness_data(args.csv_path)

    if len(data) == 0:
        print("Error: No model results found!")
        return

    # Compute correlations
    print("\nComputing correlations...")
    corr_results = compute_correlations(data, fitness)

    # Save correlation tables
    print("\nSaving correlation tables...")
    save_correlation_table(corr_results, args.output_dir)

    # Plot heatmaps
    print("\nGenerating correlation heatmaps...")
    plot_correlation_heatmap(corr_results, args.output_dir, method='spearman')
    plot_correlation_heatmap(corr_results, args.output_dir, method='pearson')

    # Plot fitness correlations
    if fitness is not None:
        print("\nGenerating fitness correlation plots...")
        plot_fitness_correlations(data, fitness, args.output_dir)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
