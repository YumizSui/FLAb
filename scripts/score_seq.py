# imports
import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from models import iglm_score, antiberty_score, progen_score, esm2_score, ablang2_score
from extra import dir_create

"""
Input: relative path to csv file (columns: heavy,light,fitness)

Output:
    csv_1 - perplexity scores
    csv_2 - correlations
    pdf - perplexity plots
"""

def _get_args():
    """Gets command line arguments"""

    desc = ("""Script for scoring antibody sequences using various language models.

    Supported score methods:
    - iglm: IgLM model
    - antiberty: AntiBERTy model
    - progen: ProGen model (requires model_variant: small/medium/large)
    - esm2: ESM2 model (requires model_variant: 650M/3B)
    - ism: ISM model (requires model_variant: 650M)
    - ablang2: AbLang2 model

    Example usage:
        python score_seq.py --csv-path data/binding/example.csv --score-method iglm
        python score_seq.py --csv-path data/binding/example.csv --score-method progen --model-variant small --device cpu
        python score_seq.py --csv-path data/binding/example.csv --score-method esm2 --model-variant 650M --device cuda:0
    """)
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--csv-path",
                      type=str,
                      required=True,
                      help="Path to CSV file with columns: heavy, light, fitness")

    parser.add_argument("--score-method",
                      type=str,
                      required=True,
                      choices=['iglm', 'antiberty', 'progen', 'esm2', 'ism', 'ablang2'],
                      help="Model for scoring sequences")

    parser.add_argument("--model-variant",
                      type=str,
                      default=None,
                      help="Model variant (required for progen/esm2/ism): progen (small/medium/large), esm2 (650M/3B), ism (650M)")

    parser.add_argument("--device",
                      type=str,
                      default='cpu',
                      help="Device to use: 'cpu' (default) or 'cuda:0' for GPU")

    parser.add_argument("--output-dir",
                      type=str,
                      default=None,
                      help="Output directory path. If not specified, uses default score/ structure")

    parser.add_argument("--ppl-only",
                      action='store_true',
                      help="Only output perplexity CSV file, skip plots and correlations")
    return parser.parse_args()

def _cli():
    args = _get_args()

    csv_path = args.csv_path
    score_method = args.score_method
    model_variant = args.model_variant
    device = args.device
    output_dir_arg = args.output_dir
    ppl_only = args.ppl_only

    # CREATE DIRECTORY PATH

    # split csv_path data/fitness/filename into variables
    dir_name, filename = os.path.split(csv_path)
    data_dir, fitness_dir = os.path.split(dir_name)

    # remove file extension from filename
    name_only, extension = os.path.splitext(filename)

    # Use provided output_dir or construct default path
    if output_dir_arg:
        output_dir = output_dir_arg
        os.makedirs(output_dir, exist_ok=True)
    else:
        score_dir = 'score'
        # create score/
        dir_create(score_dir)

        # create score/model/
        dir_create(score_dir, score_method)

        # Models with variants need subdirectory
        if score_method in ['progen', 'esm2', 'ism']:
            # create score/model/variant/
            dir_create(score_dir, score_method, model_variant)

            # create score/model/variant/fitness/
            dir_create(score_dir, score_method, model_variant, fitness_dir)

            # create score/model/variant/fitness/csv/
            dir_create(score_dir, score_method, model_variant, fitness_dir, name_only)

            # construct the 'score/score_method/variant/fitness/name/' directory path
            output_dir = os.path.join('.', score_dir, score_method, model_variant, fitness_dir, name_only)
        else:
            # create score/model/fitness/
            dir_create(score_dir, score_method, fitness_dir)

            # create score/model/fitness/csv/
            dir_create(score_dir, score_method, fitness_dir, name_only)

            # construct the 'score/score_method/fitness/name/' directory path
            output_dir = os.path.join('.', score_dir, score_method, fitness_dir, name_only)

    df = pd.read_csv(csv_path)

    # CALCULATE PERPLEXITY
    if score_method == 'iglm':
        df = iglm_score(df)

    elif score_method == 'antiberty':
        df = antiberty_score(df)

    elif score_method == 'progen':
        df = progen_score(df, model_variant, device)

    elif score_method == 'esm2':
        # Map variant to model name
        if model_variant == '650M':
            model_name = 'esm2_t33_650M_UR50D'
        elif model_variant == '3B':
            model_name = 'esm2_t36_3B_UR50D'
        else:
            model_name = 'esm2_t33_650M_UR50D'  # default
        df = esm2_score(df, model_name=model_name, device=device)

    elif score_method == 'ism':
        # ISM uses ESM2 architecture with custom weights
        ism_weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'envs', 'models', 'ism_t33_650M_uc30pdb', 'checkpoint.pth'
        )
        df = esm2_score(df, model_name='ism', device=device, ism_weights_path=ism_weights_path)

    elif score_method == 'ablang2':
        df = ablang2_score(df, device=device)

    # SAVE PERPLEXITY
    ppl_output_path = os.path.join(output_dir, f"{name_only}_ppl.csv")
    df.to_csv(ppl_output_path, index=False)

    # If ppl_only flag is set, skip plots and correlations
    if ppl_only:
        return

    # PLOT PERPLEXITY

    # Determine x axis
    if fitness_dir == 'binding':
        fitness_metric = 'negative log Kd'
        df[fitness_metric] = -np.log(df['fitness'])

    elif fitness_dir == 'immunogenicity':
        fitness_metric = 'immunogenic response (%)'
        df[fitness_metric] = df['fitness']

    elif fitness_dir == 'tm':
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

    elif fitness_dir == 'thermostability':
        fitness_metric = 'Melting temperature (°C)'
        df[fitness_metric] = df['fitness']

    else:
        # Default case for unknown fitness directories
        fitness_metric = 'fitness'
        df[fitness_metric] = df['fitness']

    df['is_first_row'] = df.index == 0
    plt.scatter(df[fitness_metric], df['average_perplexity'], c=['orange' if row else 'blue' for row in df['is_first_row']])
    plt.scatter(df[fitness_metric][0], df['average_perplexity'][0], c=['orange'])
    plt.legend(handles=[plt.Line2D([], [], marker='o', color='orange', label='wildtype'),
                        plt.Line2D([], [], marker='o', color='blue', label='mutants')])
    plt.xlabel(fitness_metric)
    plt.ylabel('average perplexity')
    plt.title(name_only)

    plot_output_path = os.path.join(output_dir, f"{name_only}_plot.png")
    plt.savefig(plot_output_path)
    plt.show()

    # CALCULATE CORRELATION

    name_list, correlation_list, p_list = ['pearson', 'spearman', 'kendal tau'], [], []

    # Pearson: linear relation between 2 variables
    pearson_corr, pearson_p_value = pearsonr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(pearson_corr)
    p_list.append(pearson_p_value)

    # Spearman's rank measures monotonic relationship between 2 variables
    spearman_corr, spearman_p_value = spearmanr(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(spearman_corr)
    p_list.append(spearman_p_value)

    # Kendall's tau measures ordinal relationship between 2 variables
    kendall_corr, kendall_p_value = kendalltau(df[fitness_metric], df['average_perplexity'])
    correlation_list.append(kendall_corr)
    p_list.append(kendall_p_value)

    df_corr = pd.DataFrame({'correlation_name': name_list, 'value': correlation_list, 'p-value': p_list})

    # SAVE CORRELATION
    corr_output_path = os.path.join(output_dir, f"{name_only}_corr.csv")
    df_corr.to_csv(corr_output_path, index=False)


if __name__=='__main__':
    _cli()

