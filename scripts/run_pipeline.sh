#!/usr/bin/env bash
# run_pipeline.sh — convenience wrapper to run the full xPPM-TDQN pipeline
# for a specific dataset.
#
# Usage:
#   ./scripts/run_pipeline.sh --dataset simbank
#   ./scripts/run_pipeline.sh --dataset mybpic --config configs/config.yaml
#   ./scripts/run_pipeline.sh --dataset simbank --from 03   # resume from step 03
#
# Steps:
#   01  preprocess_log
#   01b validate_and_split
#   02  encode_prefixes
#   03  build_mdp_dataset
#   04  train_tdqn_offline
#   05  run_ope_dr
#   06  explain_policy
#   07  fidelity_tests
#   08  distill_policy
#   09  export_schema
#   10  build_deploy_bundle

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATASET=""
CONFIG="configs/config.yaml"
FROM_STEP="01"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --from)
            FROM_STEP="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^set -/p' "$0" | head -n -1 | sed 's/^# *//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required." >&2
    echo "  Usage: $0 --dataset simbank" >&2
    exit 1
fi

DATASET_ARGS="--config $CONFIG --dataset $DATASET"

echo "=============================================="
echo "  xPPM-TDQN pipeline"
echo "  dataset : $DATASET"
echo "  config  : $CONFIG"
echo "  from    : step $FROM_STEP"
echo "=============================================="

run_step() {
    local step="$1"
    local script="$2"
    shift 2
    if [[ "$step" < "$FROM_STEP" ]]; then
        echo "  [SKIP] Step $step (--from $FROM_STEP)"
        return
    fi
    echo ""
    echo "--- Step $step: $script ---"
    python "scripts/${script}" $DATASET_ARGS "$@"
}

run_step "01"  "01_preprocess_log.py"
run_step "01b" "01b_validate_and_split.py"
run_step "02"  "02_encode_prefixes.py"
run_step "03"  "03_build_mdp_dataset.py"  --overwrite
run_step "04"  "04_train_tdqn_offline.py"
run_step "05"  "05_run_ope_dr.py"
run_step "06"  "06_explain_policy.py"
run_step "07"  "07_fidelity_tests.py"
run_step "08"  "08_distill_policy.py"
run_step "09"  "09_export_schema.py"
run_step "10"  "10_build_deploy_bundle.py"

echo ""
echo "=============================================="
echo "  Pipeline complete for dataset: $DATASET"
echo "=============================================="
