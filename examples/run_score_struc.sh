#!/bin/bash
# Example script for running score_struc.py

# Example with ESM-IF
echo "Running ESM-IF..."
python scripts/score_struc.py \
    --csv-path data/binding/example.csv \
    --score-method esmif \
    --output_dir examples/out_struc/esmif --ppl-only

# Example with PyRosetta
echo "Running PyRosetta..."
python scripts/score_struc.py \
    --csv-path data/binding/example.csv \
    --score-method pyrosetta \
    --output_dir examples/out_struc/pyrosetta --ppl-only

# Example with MPNN
python scripts/score_struc.py \
    --csv-path data/binding/example.csv \
    --score-method mpnn \
    --output_dir examples/out_struc/mpnn --ppl-only


