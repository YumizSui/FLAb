#!/usr/bin/env python
"""
Analyze AbEval PoET-2 results: compare str vs nostr performance.
"""
import json
import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np

def load_result(result_dir):
    """Load all_results.json from a result directory."""
    json_path = result_dir / "all_results.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Extract result (assuming single model entry)
    for model_name, model_data in data.items():
        for task_name, task_data in model_data.items():
            return task_data
    return None


def main():
    abeval_results_dir = Path("/home/kfurui/workspace/AbEval/results/poet2/var2")

    # Find all checkpoint directories
    checkpoints = []
    for d in sorted(abeval_results_dir.iterdir()):
        if d.is_dir() and d.name.startswith("var2_str-"):
            checkpoints.append(d.name)

    print(f"Found {len(checkpoints)} checkpoints")

    # Parse checkpoint names and group str/nostr pairs
    pairs = {}
    for ckpt in checkpoints:
        if ckpt.endswith("-nostr"):
            base_name = ckpt[:-6]  # Remove "-nostr"
            if base_name not in pairs:
                pairs[base_name] = {}
            pairs[base_name]['nostr'] = ckpt
        else:
            if ckpt not in pairs:
                pairs[ckpt] = {}
            pairs[ckpt]['str'] = ckpt

    print(f"\nFound {len(pairs)} checkpoint pairs")

    # Load results for each pair
    results = []

    for base_name, pair in sorted(pairs.items()):
        if 'str' not in pair or 'nostr' not in pair:
            continue

        str_dir = abeval_results_dir / pair['str']
        nostr_dir = abeval_results_dir / pair['nostr']

        # Check which tasks exist
        for task_dir in sorted(str_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == 'figures' or task_dir.name == 'old':
                continue

            task_name = task_dir.name
            nostr_task_dir = nostr_dir / task_name

            if not nostr_task_dir.exists():
                continue

            # Load results
            str_result = load_result(task_dir)
            nostr_result = load_result(nostr_task_dir)

            if str_result is None or nostr_result is None:
                continue

            # Extract key metrics
            dataset = str_result.get('dataset', 'unknown')
            str_spearman = str_result.get('spearman_r', None)
            nostr_spearman = nostr_result.get('spearman_r', None)
            str_pvalue = str_result.get('spearman_pvalue', None)
            nostr_pvalue = nostr_result.get('spearman_pvalue', None)
            n_mutations = str_result.get('n_mutations', None)

            if str_spearman is None or nostr_spearman is None:
                continue

            results.append({
                'checkpoint': base_name,
                'task': task_name,
                'dataset': dataset,
                'n_mutations': n_mutations,
                'str_spearman': str_spearman,
                'nostr_spearman': nostr_spearman,
                'str_pvalue': str_pvalue,
                'nostr_pvalue': nostr_pvalue,
                'diff': str_spearman - nostr_spearman,
                'abs_str': abs(str_spearman),
                'abs_nostr': abs(nostr_spearman),
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No results found!")
        return

    # Save raw results
    output_dir = Path("/home/kfurui/workspace/FLAb/abeval_analysis")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "str_vs_nostr_raw.csv", index=False)
    print(f"\nRaw results saved to {output_dir / 'str_vs_nostr_raw.csv'}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(df)}")
    print(f"Unique checkpoints: {df['checkpoint'].nunique()}")
    print(f"Unique tasks: {df['task'].nunique()}")
    print(f"Unique datasets: {df['dataset'].nunique()}")

    # Overall comparison
    print(f"\n{'='*60}")
    print("Overall Performance")
    print(f"{'='*60}")
    print(f"Mean |Spearman| (str):   {df['abs_str'].mean():.4f}")
    print(f"Mean |Spearman| (nostr): {df['abs_nostr'].mean():.4f}")
    print(f"Mean difference (str - nostr): {df['diff'].mean():.4f}")

    # Count wins
    str_better = (df['abs_str'] > df['abs_nostr']).sum()
    nostr_better = (df['abs_nostr'] > df['abs_str']).sum()
    tie = (df['abs_str'] == df['abs_nostr']).sum()

    print(f"\nWin counts (by |Spearman|):")
    print(f"  Str better:   {str_better} ({100*str_better/len(df):.1f}%)")
    print(f"  Nostr better: {nostr_better} ({100*nostr_better/len(df):.1f}%)")
    print(f"  Tie:          {tie} ({100*tie/len(df):.1f}%)")

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(df['abs_str'], df['abs_nostr'])
    print(f"\nPaired t-test (|Spearman|):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pvalue:.4e}")

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pvalue = stats.wilcoxon(df['abs_str'], df['abs_nostr'])
    print(f"\nWilcoxon signed-rank test:")
    print(f"  statistic: {w_stat:.4f}")
    print(f"  p-value: {w_pvalue:.4e}")

    # Per-task analysis
    print(f"\n{'='*60}")
    print("Per-Task Analysis")
    print(f"{'='*60}")

    task_summary = df.groupby('task').agg({
        'abs_str': 'mean',
        'abs_nostr': 'mean',
        'diff': 'mean',
        'dataset': 'first',
    }).round(4)
    task_summary['n_ckpts'] = df.groupby('task').size()
    task_summary = task_summary.sort_values('diff', ascending=False)

    print(task_summary.to_string())
    task_summary.to_csv(output_dir / "str_vs_nostr_per_task.csv")

    # Per-checkpoint analysis
    print(f"\n{'='*60}")
    print("Per-Checkpoint Analysis")
    print(f"{'='*60}")

    ckpt_summary = df.groupby('checkpoint').agg({
        'abs_str': 'mean',
        'abs_nostr': 'mean',
        'diff': 'mean',
    }).round(4)
    ckpt_summary['n_tasks'] = df.groupby('checkpoint').size()
    ckpt_summary = ckpt_summary.sort_values('diff', ascending=False)

    print(ckpt_summary.head(10).to_string())
    print(f"...")
    print(ckpt_summary.tail(10).to_string())
    ckpt_summary.to_csv(output_dir / "str_vs_nostr_per_checkpoint.csv")

    # Best performing checkpoint
    print(f"\n{'='*60}")
    print("Best Performing Checkpoint")
    print(f"{'='*60}")

    best_str = ckpt_summary['abs_str'].idxmax()
    best_nostr = ckpt_summary['abs_nostr'].idxmax()

    print(f"Best Str:   {best_str} (|ρ|={ckpt_summary.loc[best_str, 'abs_str']:.4f})")
    print(f"Best Nostr: {best_nostr} (|ρ|={ckpt_summary.loc[best_nostr, 'abs_nostr']:.4f})")

    # Detailed comparison for best checkpoint
    if best_str in df['checkpoint'].values:
        print(f"\nDetailed results for best str checkpoint: {best_str}")
        best_df = df[df['checkpoint'] == best_str][['task', 'dataset', 'str_spearman', 'nostr_spearman', 'diff']]
        print(best_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
