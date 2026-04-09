"""
Generate figures for the NAML paper.

Reads stats JSON files (one per experiment suite) from a per-run directory
under `logs/` and writes PNG figures into `<run_dir>/figures/`. Each suite is
expected to have a stats file named `<suite>_stats.json`; missing files raise
an error.

Per-suite figures (one of each, four times):
  • <suite>_ranking.png            – mean rank per optimizer (bar)
  • <suite>_final_loss.png         – mean final loss per optimizer (bar, log)
  • <suite>_evals_vs_branching.png – mean evals vs branching factor (lines)
  • <suite>_times_vs_branching.png – mean runtime vs branching factor (lines)

Cross-suite figure:
  • overall_ranking.png            – mean rank across every configuration

Usage:
    # default: read from logs/latest, write to logs/latest/figures/
    julia --project=. experiments/paper/generate_figures.jl

    # explicit run directory
    julia --project=. experiments/paper/generate_figures.jl --run-dir logs/20260408_120000
"""

include(joinpath(@__DIR__, "figures_util.jl"))
include(joinpath(@__DIR__, "table_utils.jl"))   # provides load_stats_json

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

"""Canonical experiment suite identifiers used for filenames and titles."""
const SUITES = [
    "polynomial_learning",
    "function_learning",
    "absolute_sum_minimization",
    "polynomial_solving",
]

"""
    parse_run_dir(args) -> String

Parse a `--run-dir PATH` flag from `args`. Defaults to `<repo>/logs/latest`.
Relative paths are resolved against the repo root.
"""
function parse_run_dir(args)
    run_dir = joinpath(REPO_ROOT, "logs", "latest")
    for (i, arg) in enumerate(args)
        if arg == "--run-dir" && i < length(args)
            run_dir = args[i+1]
        elseif startswith(arg, "--run-dir=")
            run_dir = split(arg, "="; limit=2)[2]
        end
    end
    return isabspath(run_dir) ? run_dir : joinpath(REPO_ROOT, run_dir)
end

const RUN_DIR    = parse_run_dir(ARGS)
const FIGURE_DIR = joinpath(RUN_DIR, "figures")

"""Pretty title for plot headings."""
function suite_title(suite::String)
    replace(suite, "_" => " ") |> titlecase
end

"""Path to a suite's stats JSON inside the run directory."""
suite_stats_path(suite::String) = joinpath(RUN_DIR, "$(suite)_stats.json")

"""Path for an output figure under `<run_dir>/figures/`."""
figure_path(name::String) = joinpath(FIGURE_DIR, name)

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

"""
    load_suite(suite) -> (experiments, optimizer_order)

Load a suite's stats JSON. Errors loudly if the file is missing — the script
assumes all four stats files have already been produced.
"""
function load_suite(suite::String)
    path = suite_stats_path(suite)
    isfile(path) || error("Missing stats file for suite '$(suite)': $(path)")
    experiments, _, optimizer_order = load_stats_json(path)
    return experiments, optimizer_order
end

# ----------------------------------------------------------------------------
# Per-suite figure generation
# ----------------------------------------------------------------------------

function generate_suite_figures(suite::String)
    println("\n=== $(suite) ===")
    experiments, optimizer_order = load_suite(suite)
    title = suite_title(suite)

    save_figure(
        generate_ranking_plot_per_experiment(experiments, optimizer_order;
            title = "$(title): mean rank"),
        figure_path("$(suite)_ranking.png"))

    save_figure(
        generate_average_final_loss(experiments, optimizer_order;
            title = "$(title): mean final loss"),
        figure_path("$(suite)_final_loss.png"))

    save_figure(
        generate_number_of_evals_plot(experiments, optimizer_order;
            title = "$(title): evaluations vs branching factor"),
        figure_path("$(suite)_evals_vs_branching.png"))

    save_figure(
        generate_times_plot(experiments, optimizer_order;
            title = "$(title): runtime vs branching factor"),
        figure_path("$(suite)_times_vs_branching.png"))

    return experiments, optimizer_order
end

# ----------------------------------------------------------------------------
# Cross-suite figure generation
# ----------------------------------------------------------------------------

"""
    merge_optimizer_orders(orders) -> Vector{String}

Combine the optimizer-name lists from several suites into a single ordering
that respects `DISPLAY_ORDER` and appends any unknown optimizers afterwards.
"""
function merge_optimizer_orders(orders)
    seen = Set{String}()
    for order in orders, name in order
        push!(seen, name)
    end
    ordered = String[name for name in DISPLAY_ORDER if name in seen]
    extras = sort([name for name in seen if !(name in DISPLAY_ORDER)])
    return vcat(ordered, extras)
end

function generate_cross_suite_figures(per_suite_data)
    println("\n=== overall ===")
    all_experiments = vcat([exps for (exps, _) in per_suite_data]...)
    optimizer_order = merge_optimizer_orders([order for (_, order) in per_suite_data])

    save_figure(
        generate_overall_ranking_plot(all_experiments, optimizer_order;
            title = "Overall mean rank across all suites"),
        figure_path("overall_ranking.png"))
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

function main()
    isdir(RUN_DIR) || error("Run directory does not exist: $(RUN_DIR)")
    println("Reading stats from: $(RUN_DIR)")
    println("Writing figures to: $(FIGURE_DIR)")

    per_suite_data = [generate_suite_figures(suite) for suite in SUITES]
    generate_cross_suite_figures(per_suite_data)

    println("\nDone.")
end

main()
