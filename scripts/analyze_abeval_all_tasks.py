#!/usr/bin/env python
"""
Analyze AbEval PoET-2 results for ALL tasks: compare str vs nostr performance.
"""
import json
import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np

def load_results_from_dir(task_dir):
    """Load all_results.json recursively from a task directory."""
    results = []

    # Check if all_results.json exists directly
    json_path = task_dir / "all_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        # Extract result
        for model_name, model_data in data.items():
            for task_name, task_data in model_data.items():
                results.append(task_data)

    # Check subdirectories
    for subdir in task_dir.iterdir():
        if subdir.is_dir():
            json_path = subdir / "all_results.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                for model_name, model_data in data.items():
                    for task_name, task_data in model_data.items():
                        results.append(task_data)

    return results


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

    print(f"Found {len(pairs)} checkpoint pairs")

    # Load results for all tasks
    all_results = []

    for base_name, pair in sorted(pairs.items()):
        if 'str' not in pair or 'nostr' not in pair:
            continue

        str_dir = abeval_results_dir / pair['str']
        nostr_dir = abeval_results_dir / pair['nostr']

        # Get all str results
        str_results = load_results_from_dir(str_dir)
        nostr_results = load_results_from_dir(nostr_dir)

        # Match by task and dataset
        for str_res in str_results:
            task = str_res.get('task', 'unknown')
            dataset = str_res.get('dataset', 'unknown')

            # Find matching nostr result
            nostr_res = None
            for nr in nostr_results:
                if nr.get('task') == task and nr.get('dataset') == dataset:
                    nostr_res = nr
                    break

            if nostr_res is None:
                continue

            str_spearman = str_res.get('spearman_r', None)
            nostr_spearman = nostr_res.get('spearman_r', None)

            if str_spearman is None or nostr_spearman is None:
                continue

            all_results.append({
                'checkpoint': base_name,
                'task': task,
                'dataset': dataset,
                'n_mutations': str_res.get('n_mutations', None),
                'with_antigen': str_res.get('with_antigen', False),
                'str_spearman': str_spearman,
                'nostr_spearman': nostr_spearman,
                'str_pvalue': str_res.get('spearman_pvalue', None),
                'nostr_pvalue': nostr_res.get('spearman_pvalue', None),
                'diff': str_spearman - nostr_spearman,
                'abs_str': abs(str_spearman),
                'abs_nostr': abs(nostr_spearman),
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No results found!")
        return

    # Save raw results
    output_dir = Path("/home/kfurui/workspace/FLAb/abeval_analysis")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "all_tasks_str_vs_nostr_raw.csv", index=False)
    print(f"\nRaw results saved to {output_dir / 'all_tasks_str_vs_nostr_raw.csv'}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(df)}")
    print(f"Unique checkpoints: {df['checkpoint'].nunique()}")
    print(f"Unique tasks: {df['task'].nunique()}")
    print(f"Unique datasets: {df['dataset'].nunique()}")
    print(f"\nTasks: {sorted(df['task'].unique())}")

    # Overall comparison
    print(f"\n{'='*60}")
    print("Overall Performance (all tasks)")
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

    # Per-task analysis
    print(f"\n{'='*60}")
    print("Per-Task Analysis")
    print(f"{'='*60}")

    task_summary = df.groupby(['task', 'dataset']).agg({
        'abs_str': ['mean', 'std'],
        'abs_nostr': ['mean', 'std'],
        'diff': ['mean', 'std'],
        'checkpoint': 'count',
    }).round(4)
    task_summary.columns = ['_'.join(col).strip() for col in task_summary.columns.values]
    task_summary = task_summary.rename(columns={'checkpoint_count': 'n_ckpts'})
    task_summary = task_summary.sort_values('diff_mean', ascending=False)

    print(task_summary.to_string())
    task_summary.to_csv(output_dir / "all_tasks_per_task_summary.csv")

    # Per-checkpoint analysis (across all tasks)
    print(f"\n{'='*60}")
    print("Per-Checkpoint Analysis (averaged across tasks)")
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
    ckpt_summary.to_csv(output_dir / "all_tasks_per_checkpoint_summary.csv")

    # Statistical test per task
    print(f"\n{'='*60}")
    print("Statistical Tests Per Task")
    print(f"{'='*60}")

    for (task, dataset), group in df.groupby(['task', 'dataset']):
        if len(group) < 3:
            continue
        t_stat, t_pvalue = stats.ttest_rel(group['abs_str'], group['abs_nostr'])
        w_stat, w_pvalue = stats.wilcoxon(group['abs_str'], group['abs_nostr'], zero_method='zsplit')

        str_better_pct = 100 * (group['abs_str'] > group['abs_nostr']).sum() / len(group)

        print(f"\n{task} ({dataset}):")
        print(f"  N={len(group)}")
        print(f"  Mean |ρ| (str):   {group['abs_str'].mean():.4f}")
        print(f"  Mean |ρ| (nostr): {group['abs_nostr'].mean():.4f}")
        print(f"  Str better: {str_better_pct:.1f}%")
        print(f"  t-test p-value: {t_pvalue:.4e}")
        print(f"  Wilcoxon p-value: {w_pvalue:.4e}")

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
