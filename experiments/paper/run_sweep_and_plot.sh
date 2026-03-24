#!/bin/bash
# Run exploration constant sweep then generate plots.
#
# Usage:
#   bash experiments/paper/run_sweep_and_plot.sh [options]
#
# Options:
#   --quick          Fast test: 3 c values, 1 config/suite, 2 samples, 5 epochs
#   --epochs N       Optimization epochs per run (default: 20)
#   --samples N      Samples per config (overrides per-config default)
#   --degree N       Branching degree for MCTS-based optimisers (default: 1)
#   --output FILE    JSON output path (default: experiments/paper/sweep_results_<timestamp>.json)
#   --plots-dir DIR  Directory for plots (default: <json_dir>/plots)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SWEEP_FLAGS=""
OUTPUT_FILE=""
PLOTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)          SWEEP_FLAGS="$SWEEP_FLAGS --quick"; shift ;;
        --epochs)         SWEEP_FLAGS="$SWEEP_FLAGS --epochs $2"; shift 2 ;;
        --samples)        SWEEP_FLAGS="$SWEEP_FLAGS --samples $2"; shift 2 ;;
        --degree)         SWEEP_FLAGS="$SWEEP_FLAGS --degree $2"; shift 2 ;;
        --output)         OUTPUT_FILE="$2"; shift 2 ;;
        --plots-dir)      PLOTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Default output filename
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$SCRIPT_DIR/sweep_results_${TIMESTAMP}.json"
fi

echo "======================================================================"
echo "Step 1: Running exploration constant sweep"
echo "======================================================================"
julia --project="$REPO_ROOT" \
    "$SCRIPT_DIR/sweep_exploration_constant.jl" \
    $SWEEP_FLAGS --output "$OUTPUT_FILE"

echo ""
echo "======================================================================"
echo "Step 2: Generating plots"
echo "======================================================================"

PLOT_ARGS="$OUTPUT_FILE"
if [ -n "$PLOTS_DIR" ]; then
    PLOT_ARGS="$PLOT_ARGS --output-dir $PLOTS_DIR"
fi

julia "$SCRIPT_DIR/plot_sweep_results.jl" $PLOT_ARGS

echo ""
echo "======================================================================"
echo "All done."
echo "  Sweep results : $OUTPUT_FILE"
if [ -n "$PLOTS_DIR" ]; then
    echo "  Plots         : $PLOTS_DIR"
else
    echo "  Plots         : $(dirname "$OUTPUT_FILE")/plots"
fi
echo "======================================================================"
