#!/usr/bin/env python
"""
Analyze all PoET-2 Tm results and compare str vs nostr.
"""
import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np

def main():
    results_dir = Path("/home/kfurui/workspace/FLAb/poet2_tm_results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all result files
    all_results = []

    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        ckpt_name = ckpt_dir.name

        str_file = ckpt_dir / "str_ppl.csv"
        nostr_file = ckpt_dir / "nostr_ppl.csv"

        if not str_file.exists() or not nostr_file.exists():
            continue

        # Load results
        str_df = pd.read_csv(str_file)
        nostr_df = pd.read_csv(nostr_file)

        # Check success
        if (str_df['status'] != 'success').all() or (nostr_df['status'] != 'success').all():
            print(f"Warning: {ckpt_name} has errors")
            continue

        # Compute correlation with fitness
        str_rho, str_p = stats.spearmanr(str_df['perplexity'], str_df['fitness'])
        nostr_rho, nostr_p = stats.spearmanr(nostr_df['perplexity'], nostr_df['fitness'])

        all_results.append({
            'checkpoint': ckpt_name,
            'str_mean_ppl': str_df['perplexity'].mean(),
            'nostr_mean_ppl': nostr_df['perplexity'].mean(),
            'str_spearman': str_rho,
            'nostr_spearman': nostr_rho,
            'str_pvalue': str_p,
            'nostr_pvalue': nostr_p,
            'abs_str': abs(str_rho),
            'abs_nostr': abs(nostr_rho),
            'diff': str_rho - nostr_rho,
            'n_samples': len(str_df),
        })

    if len(all_results) == 0:
        print("No complete results found!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    output_file = results_dir / "summary_all_configs.csv"
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

    # Overall statistics
    print(f"\n{'='*60}")
    print("Overall Statistics")
    print(f"{'='*60}")
    print(f"Total configs: {len(df)}")
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

    # Statistical tests
    if len(df) >= 3:
        t_stat, t_pvalue = stats.ttest_rel(df['abs_str'], df['abs_nostr'])
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.4e}")

        w_stat, w_pvalue = stats.wilcoxon(df['abs_str'], df['abs_nostr'])
        print(f"\nWilcoxon signed-rank test:")
        print(f"  statistic: {w_stat:.4f}")
        print(f"  p-value: {w_pvalue:.4e}")

    # Best checkpoints
    print(f"\n{'='*60}")
    print("Best Checkpoints")
    print(f"{'='*60}")

    df_sorted = df.sort_values('abs_str', ascending=False)
    print("\nTop 5 by |Spearman| (str):")
    print(df_sorted[['checkpoint', 'str_spearman', 'str_pvalue', 'abs_str']].head(5).to_string(index=False))

    df_sorted = df.sort_values('abs_nostr', ascending=False)
    print("\nTop 5 by |Spearman| (nostr):")
    print(df_sorted[['checkpoint', 'nostr_spearman', 'nostr_pvalue', 'abs_nostr']].head(5).to_string(index=False))

    # Compare with FLab original result (var2_str-0.5_nofocal_noaug_lora)
    print(f"\n{'='*60}")
    print("Comparison with FLab Original")
    print(f"{'='*60}")

    original = df[df['checkpoint'] == 'var2_str-0.5_nofocal_noaug_lora']
    if len(original) > 0:
        print(f"\nvar2_str-0.5_nofocal_noaug_lora (FLab original):")
        print(f"  Str:   ρ={original.iloc[0]['str_spearman']:.4f}, p={original.iloc[0]['str_pvalue']:.4f}")
        print(f"  Nostr: ρ={original.iloc[0]['nostr_spearman']:.4f}, p={original.iloc[0]['nostr_pvalue']:.4f}")

    print(f"\n{'='*60}")
    print("Analysis complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
