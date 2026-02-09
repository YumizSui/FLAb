#!/bin/bash
# Example script for running score_seq.py

# Basic example with IgLM
echo "Running IgLM..."
if [ -f examples/out_seq/iglm/garbinski2023_tm1_ppl.csv ]; then
    echo "IgLM already run"
else
    python scripts/score_seq.py \
        --csv-path data/thermostability/garbinski2023_tm1.csv \
        --score-method iglm --output-dir examples/out_seq/iglm --ppl-only
fi
# Example with ProGen (small variant, CPU)
# echo "Running ProGen..."
# python scripts/score_seq.py \
#     --csv-path data/thermostability/garbinski2023_tm1.csv \
#     --score-method progen \
#     --model-variant small \
#     --output-dir examples/out_seq/progen --ppl-only

# Example with ESM2 (650M variant, GPU)
echo "Running ESM2..."
if [ -f examples/out_seq/esm2/garbinski2023_tm1_ppl.csv ]; then
    echo "ESM2 already run"
else
    python scripts/score_seq.py \
        --csv-path data/thermostability/garbinski2023_tm1.csv \
        --score-method esm2 \
        --model-variant 650M \
        --output-dir examples/out_seq/esm2  --ppl-only
fi

# Example with AntiBERTy
echo "Running AntiBERTy..."
if [ -f examples/out_seq/antiberty/garbinski2023_tm1_ppl.csv ]; then
    echo "AntiBERTy already run"
else
    python scripts/score_seq.py \
        --csv-path data/thermostability/garbinski2023_tm1.csv \
        --score-method antiberty --output-dir examples/out_seq/antiberty --ppl-only
fi

# Example with AbLang2
echo "Running AbLang2..."
if [ -f examples/out_seq/ablang2/garbinski2023_tm1_ppl.csv ]; then
    echo "AbLang2 already run"
else
    python scripts/score_seq.py \
        --csv-path data/thermostability/garbinski2023_tm1.csv \
        --score-method ablang2 \
        --output-dir examples/out_seq/ablang2 --ppl-only
fi

# Example with ISM
echo "Running ISM..."
if [ -f examples/out_seq/ism/garbinski2023_tm1_ppl.csv ]; then
    echo "ISM already run"
else
    python scripts/score_seq.py \
        --csv-path data/thermostability/garbinski2023_tm1.csv \
        --score-method ism \
        --model-variant 650M \
        --output-dir examples/out_seq/ism --ppl-only
fi
