#!/bin/bash
# Example script for running score_seq.py

# Basic example with IgLM
echo "Running IgLM..."
python scripts/score_seq.py \
    --csv-path data/thermostability/garbinski2023_tm1.csv \
    --score-method iglm --output-dir examples/out_seq/iglm --ppl-only

# Example with ProGen (small variant, CPU)
# echo "Running ProGen..."
# python scripts/score_seq.py \
#     --csv-path data/thermostability/garbinski2023_tm1.csv \
#     --score-method progen \
#     --model-variant small \
#     --output-dir examples/out_seq/progen --ppl-only

# Example with ESM2 (650M variant, GPU)
echo "Running ESM2..."
python scripts/score_seq.py \
    --csv-path data/thermostability/garbinski2023_tm1.csv \
    --score-method esm2 \
    --model-variant 650M \
    --output-dir examples/out_seq/esm2  --ppl-only

# Example with AntiBERTy
echo "Running AntiBERTy..."
python scripts/score_seq.py \
    --csv-path data/thermostability/garbinski2023_tm1.csv \
    --score-method antiberty --output-dir examples/out_seq/antiberty --ppl-only

# Example with AbLang2
echo "Running AbLang2..."
python scripts/score_seq.py \
    --csv-path data/thermostability/garbinski2023_tm1.csv \
    --score-method ablang2 \
    --output-dir examples/out_seq/ablang2 --ppl-only

# Example with ISM
echo "Running ISM..."
python scripts/score_seq.py \
    --csv-path data/thermostability/garbinski2023_tm1.csv \
    --score-method ism \
    --model-variant 650M \
    --output-dir examples/out_seq/ism --ppl-only

