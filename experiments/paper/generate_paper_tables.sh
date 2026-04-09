#!/usr/bin/env bash
# ==============================================================================
# generate_paper_tables.sh
#
# Three-stage pipeline for paper experiments:
#   1. run_experiments.jl → raw JSON (per-sample results, no aggregation)
#   2. make_stats.jl      → stats JSON (adds rankings, aggregates, global ranking)
#   3. generate_tables.jl → LaTeX tables (reads stats JSON)
#
# Usage:
#   bash experiments/paper/generate_paper_tables.sh [--quick] [--epochs N] [--samples N] [--selection-mode M] [--degree D] [--verbose] [-p N]
#
# Flags:
#   --quick           Reduced epochs/simulations for smoke testing
#   --epochs N        Override epochs (default: 20)
#   --samples N       Override samples per config (default: 30)
#   --selection-mode  MCTS/DAG-MCTS selection mode (default: BestValue)
#   --degree D        Override tree branching degree (default: auto)
#   --verbose         Include per-configuration detailed tables
#   -p N, --procs N   Launch Julia with N additional worker processes (passed as `julia -p N`)
# ==============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Parse flags
# ----------------------------------------------------------------------------
QUICK_FLAG=""
EPOCHS_FLAG=""
SAMPLES_FLAG="--samples 30"
SELECTION_MODE_FLAG=""
DEGREE_FLAG=""
VERBOSE_FLAG=""
PROCS_FLAG=""

# Suite flags
SUITE_FLAGS=""

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --quick)
            QUICK_FLAG="--quick"
            i=$((i+1))
            ;;
        --epochs)
            i=$((i+1))
            EPOCHS_FLAG="--epochs ${!i}"
            i=$((i+1))
            ;;
        --epochs=*)
            EPOCHS_FLAG="--epochs ${arg#*=}"
            i=$((i+1))
            ;;
        --samples)
            i=$((i+1))
            SAMPLES_FLAG="--samples ${!i}"
            i=$((i+1))
            ;;
        --samples=*)
            SAMPLES_FLAG="--samples ${arg#*=}"
            i=$((i+1))
            ;;
        --selection-mode)
            i=$((i+1))
            SELECTION_MODE_FLAG="--selection-mode ${!i}"
            i=$((i+1))
            ;;
        --selection-mode=*)
            SELECTION_MODE_FLAG="--selection-mode ${arg#*=}"
            i=$((i+1))
            ;;
        --degree)
            i=$((i+1))
            DEGREE_FLAG="--degree ${!i}"
            i=$((i+1))
            ;;
        --degree=*)
            DEGREE_FLAG="--degree ${arg#*=}"
            i=$((i+1))
            ;;
        --paper-optimizer-comparison|--paper)
            SUITE_FLAGS="$SUITE_FLAGS --paper-optimizer-comparison"
            i=$((i+1))
            ;;
        --paper-mcts-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-branching"
            i=$((i+1))
            ;;
        --paper-dag-mcts-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-branching"
            i=$((i+1))
            ;;
        --paper-greedy-descent-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-greedy-descent-branching"
            i=$((i+1))
            ;;
        --paper-gradient-descent-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-gradient-descent-branching"
            i=$((i+1))
            ;;
        --paper-mcts-number-of-simulations)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-number-of-simulations"
            i=$((i+1))
            ;;
        --paper-dag-mcts-number-of-simulations)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-number-of-simulations"
            i=$((i+1))
            ;;
        --paper-mcts-exploration-constant)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-exploration-constant"
            i=$((i+1))
            ;;
        --paper-dag-mcts-exploration-constant)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-exploration-constant"
            i=$((i+1))
            ;;
        --verbose)
            VERBOSE_FLAG="--verbose"
            i=$((i+1))
            ;;
        -p)
            i=$((i+1))
            PROCS_FLAG="-p ${!i}"
            i=$((i+1))
            ;;
        -p=*)
            PROCS_FLAG="-p ${arg#*=}"
            i=$((i+1))
            ;;
        --procs)
            i=$((i+1))
            PROCS_FLAG="-p ${!i}"
            i=$((i+1))
            ;;
        --procs=*)
            PROCS_FLAG="-p ${arg#*=}"
            i=$((i+1))
            ;;
        *)
            i=$((i+1))
            ;;
    esac
done

# If no suites specified, default to optimizer comparison
if [ -z "$SUITE_FLAGS" ]; then
    SUITE_FLAGS="--paper-optimizer-comparison"
fi

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
# Per-run output directory: logs/<timestamp>
#
# All artifacts produced by this invocation (raw JSON, stats JSON, LaTeX
# tables, and figures generated by generate_figures.jl) live under
# $RUN_DIR. A `logs/latest` symlink is updated at the end so downstream
# tooling (e.g. run_and_deploy_tables.sh, generate_figures.jl) can find the
# most recent run without having to know the timestamp.
# ----------------------------------------------------------------------------

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_ROOT/logs/$RUN_TS"
mkdir -p "$RUN_DIR"
echo "Run output directory: $RUN_DIR"

# Helper function to run the 3-stage pipeline for one experiment.
# All output paths are absolute and live under $RUN_DIR.
run_pipeline() {
    local DIR="$1"
    local NAME="$2"

    local RAW_PATH="$RUN_DIR/${NAME}_raw.json"
    local STATS_PATH="$RUN_DIR/${NAME}_stats.json"
    local TEX_PATH="$RUN_DIR/${NAME}_tables.tex"

    # Stage 1: Run experiments → raw JSON
    step "[$NAME] Stage 1: Running experiments"
    julia --project="$REPO_ROOT" $PROCS_FLAG \
        "$DIR/run_experiments.jl" \
        --save \
        --output "$RAW_PATH" \
        $SUITE_FLAGS $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
    ok "Raw results: $RAW_PATH"

    if [ ! -f "$RAW_PATH" ]; then
        err "Expected raw results not found: $RAW_PATH"
    fi

    # Stage 2: Compute statistics → stats JSON
    step "[$NAME] Stage 2: Computing statistics"
    julia --project="$REPO_ROOT" \
        "$SCRIPT_DIR/make_stats.jl" \
        "$RAW_PATH" \
        --output "$STATS_PATH"
    ok "Stats: $STATS_PATH"

    if [ ! -f "$STATS_PATH" ]; then
        err "Expected stats file not found: $STATS_PATH"
    fi

    # Stage 3: Generate tables → LaTeX
    step "[$NAME] Stage 3: Generating tables"
    julia --project="$REPO_ROOT" \
        "$DIR/generate_tables.jl" \
        "$STATS_PATH" \
        --output "$TEX_PATH" \
        $VERBOSE_FLAG
    ok "Tables: $TEX_PATH"
}

# ----------------------------------------------------------------------------
# Run all experiments through the pipeline
#
# The NAME positional argument doubles as the canonical "suite key": it is
# used both for the basenames of the artifacts in $RUN_DIR and as the suite
# identifier consumed by generate_figures.jl, so the two stay in sync.
# ----------------------------------------------------------------------------

run_pipeline "$SCRIPT_DIR/absolute_sum_minimization" "absolute_sum_minimization"
run_pipeline "$SCRIPT_DIR/function_learning"        "function_learning"
run_pipeline "$SCRIPT_DIR/polynomial_learning"      "polynomial_learning"
run_pipeline "$SCRIPT_DIR/polynomial_solving"       "polynomial_solving"

# ----------------------------------------------------------------------------
# Update logs/latest symlink to point at this run
# ----------------------------------------------------------------------------

LATEST_LINK="$REPO_ROOT/logs/latest"
ln -sfn "$RUN_TS" "$LATEST_LINK"
ok "Updated symlink: $LATEST_LINK -> $RUN_TS"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo
echo "======================================================================"
echo "All paper experiments complete and tables regenerated."
echo "  Run directory: $RUN_DIR"
echo "  $RUN_DIR/absolute_sum_minimization_tables.tex"
echo "  $RUN_DIR/function_learning_tables.tex"
echo "  $RUN_DIR/polynomial_learning_tables.tex"
echo "  $RUN_DIR/polynomial_solving_tables.tex"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "======================================================================"
