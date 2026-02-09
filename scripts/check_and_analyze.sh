#!/bin/bash

# Check if all models are complete and run correlation analysis
# Usage: bash scripts/check_and_analyze.sh

cd /home/kfurui/workspace/FLAb

OUTPUT_DIR="/home/kfurui/workspace/FLAb/test_full_pyrosetta"
CSV_PATH="/home/kfurui/workspace/FLAb/data/thermostability/jain2017biophysical_Tm.csv"

# Models to check
MODELS=(
    "antiberty"
    "esm2"
    "ablang2"
    "sablm_nostr"
    "sablm_str"
    "pyrosetta"
    "esmif"
    "mpnn"
)

echo "=========================================="
echo "Checking Model Completion Status"
echo "=========================================="
echo ""

# Check which models are complete
COMPLETE=()
MISSING=()

for model in "${MODELS[@]}"; do
    if [ -f "$OUTPUT_DIR/${model}_ppl.csv" ]; then
        # Check if file has data (more than just header)
        lines=$(wc -l < "$OUTPUT_DIR/${model}_ppl.csv")
        if [ $lines -gt 1 ]; then
            COMPLETE+=("$model")
            echo "✓ $model"
        else
            MISSING+=("$model")
            echo "✗ $model (empty file)"
        fi
    else
        MISSING+=("$model")
        echo "✗ $model (not found)"
    fi
done

echo ""
echo "Complete: ${#COMPLETE[@]}/8"
echo "Missing: ${#MISSING[@]}/8"
echo ""

# If all complete, run correlation analysis
if [ ${#COMPLETE[@]} -eq 8 ]; then
    echo "=========================================="
    echo "All models complete! Running correlation analysis..."
    echo "=========================================="
    echo ""

    source .venv/bin/activate
    python scripts/analyze_model_correlations.py \
        --output-dir "$OUTPUT_DIR" \
        --csv-path "$CSV_PATH" \
        --models "${COMPLETE[@]}"

    echo ""
    echo "=========================================="
    echo "Analysis complete! Check results in:"
    echo "  $OUTPUT_DIR/correlation_*.png"
    echo "  $OUTPUT_DIR/correlation_*.csv"
    echo "  $OUTPUT_DIR/fitness_correlation_summary.csv"
    echo "=========================================="
else
    echo "Waiting for remaining models: ${MISSING[@]}"
    echo ""
    echo "Running jobs:"
    squeue -u $USER
    echo ""
    echo "Re-run this script after jobs complete."
fi
