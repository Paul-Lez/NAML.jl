"""
LaTeX Table Generator for Absolute Sum Minimization Results

Reads stats JSON (produced by make_stats.jl) and generates LaTeX tables.

Usage:
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json>
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --output tables.tex
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --stdout
    julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl <stats.json> --verbose
"""

include("../table_utils.jl")

# Parse arguments
targs = parse_table_args(ARGS, "absolute_sum_tables.tex")
experiments, metadata, optimizer_order = load_stats_json(targs.json_file)

# ============================================================================
# Experiment-specific: Configuration table
# ============================================================================

function abssum_config_table(experiments)
    generate_config_table(
        experiments,
        "tab:abssum-config",
        "Absolute sum minimization experiment configurations. Each row describes one experimental setup.",
        "Experiment & Prime (\$p\$) & Precision & \\#Polys & \\#Vars & Poly Deg. & \\#Samples",
        config -> "$(config["prime"]) & $(config["prec"]) & $(config["num_polys"]) & $(config["num_vars"]) & $(config["degree"]) & $(config["num_samples"])"
    )
end

# ============================================================================
# Generate unified document
# ============================================================================

function generate_document(experiments, optimizer_order; verbose=false)
    lines = String[]

    push!(lines, "% ============================================================================")
    push!(lines, "% LaTeX Tables for Absolute Sum Minimization Experiment")
    push!(lines, "% Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% ============================================================================")
    push!(lines, "")

    # Config table
    push!(lines, abssum_config_table(experiments))

    # Summary (mean final loss)
    push!(lines, as_landscape(generate_summary_table(experiments, optimizer_order,
        "tab:abssum-summary",
        "Absolute sum minimization: mean final loss across optimizers. Lower is better. Values are averaged over multiple random problem instances.")))

    # Timing
    push!(lines, as_landscape(generate_timing_table(experiments, optimizer_order,
        "tab:abssum-timing",
        "Mean wall-clock time (seconds) per optimizer for absolute sum minimization.")))

    # Overall optimizer comparison
    push!(lines, generate_optimizer_aggregate_table(experiments, optimizer_order,
        "tab:abssum-optimizer-aggregate",
        "Overall optimizer comparison aggregated across all absolute sum configurations. Shows mean performance metrics."))

    # Eval counts
    push!(lines, as_landscape(generate_eval_count_table(experiments, optimizer_order,
        "tab:abssum-evals",
        "Mean number of function evaluations per optimizer for absolute sum minimization (excluding monitoring calls). Lower is more efficient.")))

    # Detailed (verbose only)
    if verbose
        push!(lines, generate_detailed_tables(experiments, optimizer_order,
            "tab:abssum-detail",
            "Detailed results for configuration"))
    end

    # Ranking
    push!(lines, as_landscape(generate_ranking_table(experiments, optimizer_order,
        "tab:abssum-ranking",
        "Absolute sum minimization: optimizer ranking by mean final loss per configuration (rank 1 = best, lower is better). Averaged over random samples. Bold marks the best-ranked optimizer per row (excluding Random). The \\textit{Average} row shows the mean rank across all configurations.")))

    return join(lines, "\n")
end

# ============================================================================
# Main
# ============================================================================

document = generate_document(experiments, optimizer_order; verbose=targs.verbose)
write_or_print(document, targs.json_file, targs.output_file, targs.print_stdout)
