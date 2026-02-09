#!/usr/bin/env python3
"""
Unified test script for thermostability scoring across multiple models.

This script serves as a wrapper around score_seq.py and score_struc.py,
allowing batch testing of multiple models with a single command.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Model categorization
SEQUENCE_MODELS = ['antiberty', 'esm2', 'ism', 'ablang2', 'iglm', 'progen']
STRUCTURE_MODELS = ['esmif', 'mpnn', 'pyrosetta', 'sablm_str', 'sablm_nostr']

# Model variants
MODEL_VARIANTS = {
    'esm2': '650M',
    'ism': '650M',
    'progen': 'small',
}

# Map model names to score methods
MODEL_TO_SCORE_METHOD = {
    'sablm_nostr': 'sablm',
    'sablm_str': 'sablm',
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test multiple antibody scoring models on thermostability data'
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Path to CSV file with antibody sequences and fitness values'
    )

    parser.add_argument(
        '--structure-dir',
        type=str,
        default=None,
        help='Directory containing PDB structure files (required for structure-based models)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Output directory for results (default: test_results)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help=f'Models to test. Sequence: {", ".join(SEQUENCE_MODELS)}. Structure: {", ".join(STRUCTURE_MODELS)}'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: cuda or cpu (default: cuda)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for sequence models (default: 32)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='poet2_ab_enc_pretrain',
        help='PoET-2 checkpoint name or path for sablm models (default: poet2_ab_enc_pretrain)'
    )

    return parser.parse_args()


def run_score_seq(model, csv_path, output_dir, device, batch_size):
    """Run score_seq.py for a sequence model."""
    script_path = Path(__file__).parent / 'score_seq.py'

    cmd = [
        'python', str(script_path),
        '--csv-path', csv_path,
        '--score-method', model,
        '--output-dir', output_dir,
        '--device', device,
        '--batch-size', str(batch_size),
        '--no-batch',  # Disable batching due to shape mismatch bug
        '--ppl-only',  # Skip plots for batch testing
    ]

    # Add model variant if needed
    if model in MODEL_VARIANTS:
        cmd.extend(['--model-variant', MODEL_VARIANTS[model]])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_score_struc(model, csv_path, structure_dir, output_dir, device, checkpoint='poet2_ab_enc_pretrain'):
    """Run score_struc.py for a structure-based model."""
    script_path = Path(__file__).parent / 'score_struc.py'

    # Map model name to score method
    score_method = MODEL_TO_SCORE_METHOD.get(model, model)

    cmd = [
        'python', str(script_path),
        '--csv-path', csv_path,
        '--score-method', score_method,
        '--output-dir', output_dir,
        '--device', device,
        '--ppl-only',  # Skip plots for batch testing
    ]

    # Add structure directory if provided
    if structure_dir:
        cmd.extend(['--structure-dir', structure_dir])

    # Add variant for sablm models
    if model == 'sablm_nostr':
        cmd.extend(['--variant', 'nostr'])
    elif model == 'sablm_str':
        cmd.extend(['--variant', 'str'])

    # Add checkpoint for sablm models
    if model in ['sablm_nostr', 'sablm_str']:
        cmd.extend(['--checkpoint', checkpoint])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    args = get_args()

    # Validate models
    unknown_models = []
    for model in args.models:
        if model not in SEQUENCE_MODELS and model not in STRUCTURE_MODELS:
            unknown_models.append(model)

    if unknown_models:
        print(f"Error: Unknown models: {', '.join(unknown_models)}", file=sys.stderr)
        print(f"Available sequence models: {', '.join(SEQUENCE_MODELS)}", file=sys.stderr)
        print(f"Available structure models: {', '.join(STRUCTURE_MODELS)}", file=sys.stderr)
        sys.exit(1)

    # Check structure directory for structure models (except sablm_nostr which doesn't need structure)
    structure_models_to_run = [m for m in args.models if m in STRUCTURE_MODELS and m != 'sablm_nostr']
    if structure_models_to_run and not args.structure_dir:
        print(f"Error: --structure-dir required for structure models: {', '.join(structure_models_to_run)}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Track results
    results = {}
    failed_models = []

    # Run each model
    for model in args.models:
        print(f"\n{'#'*60}")
        print(f"# Testing model: {model}")
        print(f"{'#'*60}")

        try:
            if model in SEQUENCE_MODELS:
                returncode = run_score_seq(
                    model,
                    args.csv_path,
                    args.output_dir,
                    args.device,
                    args.batch_size
                )
            else:
                returncode = run_score_struc(
                    model,
                    args.csv_path,
                    args.structure_dir,
                    args.output_dir,
                    args.device,
                    args.checkpoint
                )

            if returncode == 0:
                results[model] = 'SUCCESS'
                print(f"\n✓ {model} completed successfully")
            else:
                results[model] = 'FAILED'
                failed_models.append(model)
                print(f"\n✗ {model} failed with return code {returncode}", file=sys.stderr)

        except Exception as e:
            results[model] = f'ERROR: {e}'
            failed_models.append(model)
            print(f"\n✗ {model} encountered error: {e}", file=sys.stderr)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, status in results.items():
        status_symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"{status_symbol} {model}: {status}")

    if failed_models:
        print(f"\n{len(failed_models)} model(s) failed: {', '.join(failed_models)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} model(s) completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
