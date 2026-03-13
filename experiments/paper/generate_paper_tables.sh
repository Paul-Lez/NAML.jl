#!/usr/bin/env bash
# ==============================================================================
# generate_paper_tables.sh
#
# Runs all paper experiments and regenerates the corresponding LaTeX tables.
#
# Usage:
#   ./experiments/paper/generate_paper_tables.sh [--quick] [--epochs N] [--samples N]
#
# Flags:
#   --quick      Use reduced epochs/simulations for a fast smoke-test run
#   --epochs N   Override number of epochs (default: 20)
#   --samples N  Override number of samples per config (default: 30)
#
# The script must be run from the repository root, e.g.:
#   bash experiments/paper/generate_paper_tables.sh
# ==============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Parse flags
# ----------------------------------------------------------------------------
QUICK_FLAG=""
EPOCHS_FLAG=""
SAMPLES_FLAG="--samples 30"

for arg in "$@"; do
    case "$arg" in
        --quick)     QUICK_FLAG="--quick" ;;
        --epochs)    shift; EPOCHS_FLAG="--epochs $1" ;;
        --epochs=*)  EPOCHS_FLAG="--epochs ${arg#*=}" ;;
        --samples)   shift; SAMPLES_FLAG="--samples $1" ;;
        --samples=*) SAMPLES_FLAG="--samples ${arg#*=}" ;;
    esac
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

step() { echo; echo "==== $* ===="; }
ok()   { echo "  OK: $*"; }
err()  { echo "  ERROR: $*" >&2; exit 1; }

START_TIME=$(date +%s)

# ----------------------------------------------------------------------------
# Absolute sum minimization
# ----------------------------------------------------------------------------

ABSSUM_DIR="$SCRIPT_DIR/absolute_sum_minimization"
ABSSUM_RESULTS="$ABSSUM_DIR/absolute_sum_results_paper.json"

step "Running absolute_sum_minimization experiments"
julia --project="$REPO_ROOT" \
    "$ABSSUM_DIR/run_experiments.jl" \
    --paper --save \
    --output absolute_sum_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG
ok "Experiments done"

step "Generating absolute_sum_minimization tables"
if [ ! -f "$ABSSUM_RESULTS" ]; then
    err "Expected results file not found: $ABSSUM_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$ABSSUM_DIR/generate_tables.jl" \
    "$ABSSUM_RESULTS" \
    --output absolute_sum_tables.tex
ok "Tables written to $ABSSUM_DIR/absolute_sum_tables.tex"

# ----------------------------------------------------------------------------
# Function learning
# ----------------------------------------------------------------------------

FUNCLEARN_DIR="$SCRIPT_DIR/function_learning"
FUNCLEARN_RESULTS="$FUNCLEARN_DIR/function_learning_results_paper.json"

step "Running function_learning experiments"
julia --project="$REPO_ROOT" \
    "$FUNCLEARN_DIR/run_experiments.jl" \
    --paper --save \
    --output function_learning_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG
ok "Experiments done"

step "Generating function_learning tables"
if [ ! -f "$FUNCLEARN_RESULTS" ]; then
    err "Expected results file not found: $FUNCLEARN_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$FUNCLEARN_DIR/generate_tables.jl" \
    "$FUNCLEARN_RESULTS" \
    --output function_learning_tables.tex
ok "Tables written to $FUNCLEARN_DIR/function_learning_tables.tex"

# ----------------------------------------------------------------------------
# Polynomial learning
# ----------------------------------------------------------------------------

POLYLEARN_DIR="$SCRIPT_DIR/polynomial_learning"
POLYLEARN_RESULTS="$POLYLEARN_DIR/poly_learning_results_paper.json"

step "Running polynomial_learning experiments"
julia --project="$REPO_ROOT" \
    "$POLYLEARN_DIR/run_experiments.jl" \
    --paper --save \
    --output poly_learning_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG
ok "Experiments done"

step "Generating polynomial_learning tables"
if [ ! -f "$POLYLEARN_RESULTS" ]; then
    err "Expected results file not found: $POLYLEARN_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$POLYLEARN_DIR/generate_tables.jl" \
    "$POLYLEARN_RESULTS" \
    --output polynomial_learning_tables.tex
ok "Tables written to $POLYLEARN_DIR/polynomial_learning_tables.tex"

# ----------------------------------------------------------------------------
# Polynomial solving
# ----------------------------------------------------------------------------

POLYSOLVE_DIR="$SCRIPT_DIR/polynomial_solving"
POLYSOLVE_RESULTS="$POLYSOLVE_DIR/polynomial_solving_results_paper.json"

step "Running polynomial_solving experiments"
julia --project="$REPO_ROOT" \
    "$POLYSOLVE_DIR/run_experiments.jl" \
    --paper --save \
    --output polynomial_solving_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG
ok "Experiments done"

step "Generating polynomial_solving tables"
if [ ! -f "$POLYSOLVE_RESULTS" ]; then
    err "Expected results file not found: $POLYSOLVE_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$POLYSOLVE_DIR/generate_tables.jl" \
    "$POLYSOLVE_RESULTS" \
    --output polynomial_solving_tables.tex
ok "Tables written to $POLYSOLVE_DIR/polynomial_solving_tables.tex"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo
echo "======================================================================"
echo "All paper experiments complete and tables regenerated."
echo "  $ABSSUM_DIR/absolute_sum_tables.tex"
echo "  $FUNCLEARN_DIR/function_learning_tables.tex"
echo "  $POLYLEARN_DIR/polynomial_learning_tables.tex"
echo "  $POLYSOLVE_DIR/polynomial_solving_tables.tex"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "======================================================================"
