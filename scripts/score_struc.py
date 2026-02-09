# imports
import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from models import esmif_score, pyrosetta_score, mpnn_score, sablm_score
from extra import dir_create, extract_last_digits

"""
Input: relative path to csv file (columns: heavy,light,fitness)

Output:
    csv_1 - perplexity scores
    csv_2 - correlations
    pdf - perplexity plots
"""

def _get_args():
    """Gets command line arguments"""

    desc = ("""Script for scoring antibody sequences""")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--csv-path",
                      type=str,
                      required=True,
                      help="csv file with heavy and light chain sequences, and fitness metric.")

    parser.add_argument("--score-method",
                      type=str,
                      required=True,
                      help="model for scoring (ex: esmif, pyrosetta, mpnn, sablm).")

    parser.add_argument("--device",
                      type=str,
                      default=None,
                      help="Device to use: 'cuda:0' for GPU (default: auto-detect GPU if available, else CPU) or 'cpu'")

    parser.add_argument("--output-dir",
                      type=str,
                      default=None,
                      help="Output directory path. If not specified, uses default score/ structure")

    parser.add_argument("--ppl-only",
                      action='store_true',
                      help="Only output perplexity CSV file, skip plots and correlations")

    parser.add_argument("--variant",
                      type=str,
                      default=None,
                      help="Model variant (for sablm: 'str' for structure, 'nostr' for sequence-only)")

    parser.add_argument("--structure-dir",
                      type=str,
                      default=None,
                      help="Directory containing structure files (default: auto-constructed from CSV path)")

    parser.add_argument("--checkpoint",
                      type=str,
                      default="poet2_ab_enc_pretrain",
                      help="PoET-2 checkpoint name or path (for sablm models)")

    parser.add_argument("--pll-batch-size",
                      type=int,
                      default=64,
                      help="Batch size for PLL computation in sablm (default: 64)")
    return parser.parse_args()

def _cli():
    args = _get_args()

    csv_path = args.csv_path
    score_method = args.score_method
    device = args.device
    output_dir_arg = args.output_dir
    ppl_only = args.ppl_only
    variant = args.variant
    structure_dir_arg = args.structure_dir
    checkpoint = args.checkpoint
    pll_batch_size = args.pll_batch_size

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

        # check score/model
        dir_create(score_dir, score_method)

        # create score/model/fitness
        dir_create(score_dir, score_method, fitness_dir)

        # check score/model/fitness/csv
        dir_create(score_dir, score_method, fitness_dir, name_only)

        # construct the 'score/score_method/fitness/name/' directory path
        output_dir = os.path.join('.', score_dir, score_method, fitness_dir, name_only)

    df_og = pd.read_csv(csv_path)

    # path to PDB files
    if structure_dir_arg:
        pdb_dir = structure_dir_arg
    else:
        structure_dir ='structure'
        pdb_dir = f'{structure_dir}/{fitness_dir}/{name_only}'

    # Collect all PDB files (including in subdirectories)
    pdb_files = []
    for root, dirs, files in os.walk(pdb_dir):
        for file in files:
            if file.endswith(".pdb"):
                pdb_files.append(os.path.join(root, file))

    # Sort PDBs by their numeric ID for deterministic chunk ordering
    pdb_files.sort(key=lambda p: extract_last_digits(os.path.splitext(os.path.basename(p))[0]))

    # Process in chunks with tmp checkpoints
    CHUNK_SIZE = 100
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    all_chunk_files = []
    for chunk_start in range(0, len(pdb_files), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(pdb_files))
        tmp_path = os.path.join(tmp_dir, f"{chunk_start:04d}.csv")
        all_chunk_files.append(tmp_path)

        if os.path.exists(tmp_path):
            print(f"SKIP chunk {chunk_start:04d}: {tmp_path}")
            continue

        chunk_pdb_list = []
        chunk_pdb_score = []

        for pdb_path in pdb_files[chunk_start:chunk_end]:
            pdb_file = os.path.basename(pdb_path)
            pdb_name = pdb_file.split('.')[0]
            chunk_pdb_list.append(extract_last_digits(pdb_name))

            if score_method == 'esmif':
                chunk_pdb_score.append(esmif_score(pdb_path))
            elif score_method == 'pyrosetta':
                chunk_pdb_score.append(pyrosetta_score(pdb_path))
            elif score_method == 'mpnn':
                chunk_pdb_score.append(mpnn_score(pdb_path))
            elif score_method == 'sablm':
                row_id = extract_last_digits(pdb_name)
                heavy_seq = df_og.loc[row_id, 'heavy']
                light_seq = df_og.loc[row_id, 'light']

                if variant == 'nostr':
                    use_structure = False
                elif variant == 'str':
                    use_structure = True
                else:
                    use_structure = True

                chunk_pdb_score.append(sablm_score(
                    pdb_path=pdb_path,
                    heavy_seq=heavy_seq,
                    light_seq=light_seq,
                    use_structure_info=use_structure,
                    checkpoint=checkpoint,
                    device=device if device else 'cuda',
                    pll_batch_size=pll_batch_size,
                ))

        chunk_df = pd.DataFrame({"pdb_file": chunk_pdb_list, "average_perplexity": chunk_pdb_score})
        chunk_df.to_csv(tmp_path, index=False)
        print(f"SAVED chunk {chunk_start:04d}: {tmp_path} ({len(chunk_pdb_list)} PDBs)")

    # Concat all chunks and produce final output
    all_chunks = [pd.read_csv(p) for p in all_chunk_files]
    df_scores = pd.concat(all_chunks, ignore_index=True)

    df_order = df_scores.sort_values('pdb_file').reset_index(drop=True)
    df = pd.concat([df_og, df_order], axis=1)

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

