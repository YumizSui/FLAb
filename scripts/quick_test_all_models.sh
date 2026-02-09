#!/bin/bash
#
# Quick test script to verify all model environments work
# Uses CPU and small sample to quickly validate setup
#

set -e

BASE_DIR="/gs/bs/tga-furui/workspace/antibody/FLAb"
cd "$BASE_DIR"

CSV_PATH="data/thermostability/jain2017biophysical_Tm.csv"
STRUCTURE_DIR="new_structure/thermostability/jain2017biophysical_Tm"
OUTPUT_DIR="test_results_quick"
DEVICE="cpu"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Quick Model Environment Test"
echo "Testing with CPU and --no-batch for speed"
echo "============================================================"
echo ""

# Test sequence models with antiberty environment
echo "### Testing Sequence Models (antiberty env) ###"
source envs/antiberty/.venv/bin/activate

models=("antiberty" "esm2" "ablang2")
for model in "${models[@]}"; do
    echo ""
    echo "Testing $model..."
    if python scripts/score_seq.py \
        --csv-path "$CSV_PATH" \
        --score-method "$model" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --no-batch \
        --ppl-only; then
        echo "✓ $model: SUCCESS"
    else
        echo "✗ $model: FAILED"
    fi
done

# Test structure models with esmif_mpnn environment
echo ""
echo "### Testing Structure Models (esmif_mpnn env) ###"
eval "$(pixi shell-hook -s bash -m envs/esmif_mpnn)"

models=("esmif" "mpnn")
for model in "${models[@]}"; do
    echo ""
    echo "Testing $model..."
    if python scripts/score_struc.py \
        --csv-path "$CSV_PATH" \
        --score-method "$model" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --ppl-only; then
        echo "✓ $model: SUCCESS"
    else
        echo "✗ $model: FAILED"
    fi
done

# Test pyrosetta
echo ""
echo "### Testing PyRosetta (pyrosetta env) ###"
eval "$(pixi shell-hook -s bash -m envs/pyrosetta)"

echo ""
echo "Testing pyrosetta..."
if python scripts/score_struc.py \
    --csv-path "$CSV_PATH" \
    --score-method "pyrosetta" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --ppl-only; then
    echo "✓ pyrosetta: SUCCESS"
else
    echo "✗ pyrosetta: FAILED"
fi

# Test PoET-2 models
echo ""
echo "### Testing PoET-2 Models (PoET-2 env) ###"
source PoET-2/.venv/bin/activate

echo ""
echo "Testing sablm_nostr..."
if python scripts/score_struc.py \
    --csv-path "$CSV_PATH" \
    --score-method "sablm" \
    --variant "nostr" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --ppl-only; then
    echo "✓ sablm_nostr: SUCCESS"
else
    echo "✗ sablm_nostr: FAILED"
fi

echo ""
echo "Testing sablm_str..."
if python scripts/score_struc.py \
    --csv-path "$CSV_PATH" \
    --score-method "sablm" \
    --variant "str" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --ppl-only; then
    echo "✓ sablm_str: SUCCESS"
else
    echo "✗ sablm_str: FAILED"
fi

echo ""
echo "============================================================"
echo "Quick test complete!"
echo "============================================================"
