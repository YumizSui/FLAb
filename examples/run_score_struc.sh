#!/bin/bash
# Example script for running score_struc.py
INPUTFILE=data/thermostability/garbinski2023_tm1.csv
OUTPUTDIR=examples/out_struc
# Example with ESM-IF
echo "Running ESM-IF..."
if [ -f $OUTPUTDIR/esmif/garbinski2023_tm1_ppl.csv ]; then
    echo "ESM-IF already run"
else
    python scripts/score_struc.py \
        --csv-path $INPUTFILE \
        --score-method esmif \
        --output-dir $OUTPUTDIR/esmif --ppl-only
fi

# Example with PyRosetta
echo "Running PyRosetta..."
if [ -f $OUTPUTDIR/pyrosetta/garbinski2023_tm1_ppl.csv ]; then
    echo "PyRosetta already run"
else
    python scripts/score_struc.py \
        --csv-path $INPUTFILE \
        --score-method pyrosetta \
        --output-dir $OUTPUTDIR/pyrosetta --ppl-only
fi

# Example with MPNN
echo "Running MPNN..."
if [ -f $OUTPUTDIR/mpnn/garbinski2023_tm1_ppl.csv ]; then
    echo "MPNN already run"
else
    python scripts/score_struc.py \
        --csv-path $INPUTFILE \
        --score-method mpnn \
        --output-dir $OUTPUTDIR/mpnn --ppl-only
fi

