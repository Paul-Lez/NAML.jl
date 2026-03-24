"""
Plot Exploration Constant Sweep Results

Reads a JSON file produced by sweep_exploration_constant.jl and generates:
  - Per-config figures: loss and rank vs c, with ±std ribbon (one per config)
  - Per-suite summary figures: averaged across all configs in the suite
  - A combined all_plots.pdf (via ghostscript if available)

All PDFs are saved to a plots/ subdirectory next to the JSON file.

Usage:
    julia experiments/paper/plot_sweep_results.jl <path_to_sweep_results.json>
    julia experiments/paper/plot_sweep_results.jl <path_to_sweep_results.json> --output-dir /path/to/dir
"""

using JSON
using Plots
using Printf

gr()  # GR backend produces clean PDFs

# ============================================================================
# Arguments
# ============================================================================

if length(ARGS) < 1
    println("Usage: julia plot_sweep_results.jl <sweep_results.json> [--output-dir DIR]")
    exit(1)
end

json_path = ARGS[1]

output_dir = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--output-dir" && i < length(ARGS)
        global output_dir = ARGS[i+1]
    end
end

if isnothing(output_dir)
    output_dir = joinpath(dirname(abspath(json_path)), "plots")
end

mkpath(output_dir)
println("Loading: $json_path")
println("Output : $output_dir")

# ============================================================================
# Load data
# ============================================================================

data       = JSON.parsefile(json_path)
aggregates = data["aggregates"]
metadata   = data["metadata"]

c_values   = Float64.(metadata["c_values"])
suites     = String.(metadata["suites"])

# Only plot optimizers that actually vary with c
const TREE_SEARCH_OPTS = ["MCTS-50", "MCTS-100", "MCTS-200",
                          "DAG-MCTS-50", "DAG-MCTS-100", "DAG-MCTS-200"]

# ============================================================================
# Aesthetics
# ============================================================================

# Color + style per optimizer
const OPT_STYLE = Dict(
    "MCTS-50"     => (color=:steelblue,      ls=:solid,  lw=1.5, label="MCTS-50"),
    "MCTS-100"    => (color=:royalblue,      ls=:dash,   lw=1.5, label="MCTS-100"),
    "MCTS-200"    => (color=:navy,           ls=:dot,    lw=2.0, label="MCTS-200"),
    "DAG-MCTS-50" => (color=:darkorange,     ls=:solid,  lw=1.5, label="DAG-MCTS-50"),
    "DAG-MCTS-100"=> (color=:orangered,      ls=:dash,   lw=1.5, label="DAG-MCTS-100"),
    "DAG-MCTS-200"=> (color=:darkred,        ls=:dot,    lw=2.0, label="DAG-MCTS-200"),
)

const RIBBON_ALPHA = 0.15
const FONT_SIZE    = 10
const TITLE_SIZE   = 11
const FIG_SIZE     = (950, 420)

default(
    framestyle  = :box,
    grid        = true,
    gridalpha   = 0.3,
    tickfontsize= FONT_SIZE - 2,
    legendfontsize = FONT_SIZE - 2,
    guidefontsize  = FONT_SIZE,
    titlefontsize  = TITLE_SIZE,
    margin      = 4Plots.mm,
)

# ============================================================================
# Helper: extract (c, mean, std) vectors for one (suite, config, optimizer)
# ============================================================================

function get_series(agg_list, suite, config_name, optimizer)
    rows = filter(r -> r["suite"] == suite &&
                       get(r, "config_name", "") == config_name &&
                       r["optimizer"] == optimizer, agg_list)
    isempty(rows) && return Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    sort!(rows, by=r -> r["c"])

    cs       = Float64[r["c"] for r in rows]
    losses   = Float64[r["mean_final_loss"] for r in rows]
    loss_std = Float64[get(r, "std_final_loss", 0.0) for r in rows]
    ranks    = Float64[get(r, "mean_rank", NaN) for r in rows]
    rank_std = Float64[get(r, "std_rank", 0.0) for r in rows]
    return cs, losses, loss_std, ranks, rank_std
end

# ============================================================================
# Helper: build a 2-panel (loss | rank) figure for a set of (c, series) data
# ============================================================================

function make_figure(title_str::String,
                     series_data::Vector;   # [(opt_name, cs, losses, loss_std, ranks, rank_std)]
                     loss_ylog::Bool=true)

    p_loss = plot(
        title  = "Mean Final Loss",
        xlabel = "Exploration constant c",
        ylabel = loss_ylog ? "Loss (log scale)" : "Loss",
        yscale = loss_ylog ? :log10 : :identity,
        legend = :topright,
    )

    p_rank = plot(
        title  = "Mean Rank",
        xlabel = "Exploration constant c",
        ylabel = "Rank (1 = best)",
        legend = :topright,
    )

    for (opt_name, cs, losses, loss_std, ranks, rank_std) in series_data
        isempty(cs) && continue
        sty = OPT_STYLE[opt_name]

        # Loss panel
        if loss_ylog
            # Ribbon on log scale: use asymmetric lower/upper clipped to positives
            lo = max.(losses .- loss_std, losses .* 1e-3)
            hi = losses .+ loss_std
            plot!(p_loss, cs, losses,
                ribbon    = (losses .- lo, hi .- losses),
                fillalpha = RIBBON_ALPHA,
                color     = sty.color, linestyle = sty.ls, linewidth = sty.lw,
                label     = sty.label)
        else
            plot!(p_loss, cs, losses,
                ribbon    = loss_std,
                fillalpha = RIBBON_ALPHA,
                color     = sty.color, linestyle = sty.ls, linewidth = sty.lw,
                label     = sty.label)
        end

        # Rank panel (linear)
        valid = .!isnan.(ranks)
        if any(valid)
            plot!(p_rank, cs[valid], ranks[valid],
                ribbon    = rank_std[valid],
                fillalpha = RIBBON_ALPHA,
                color     = sty.color, linestyle = sty.ls, linewidth = sty.lw,
                label     = sty.label)
        end
    end

    fig = plot(p_loss, p_rank,
               layout     = (1, 2),
               size       = FIG_SIZE,
               plot_title = title_str,
               plot_titlefontsize = TITLE_SIZE,
               left_margin  = 6Plots.mm,
               bottom_margin= 6Plots.mm)
    return fig
end

# ============================================================================
# Build all figures
# ============================================================================

all_pdfs = String[]

function save_fig(fig, path)
    mkpath(dirname(path))
    savefig(fig, path)
    push!(all_pdfs, path)
    println("  ✓ $path")
end

for suite in suites
    println("\n── Suite: $suite ──")
    suite_dir = joinpath(output_dir, suite)
    mkpath(suite_dir)

    suite_agg    = filter(r -> r["suite"] == suite, aggregates)
    config_names = sort(unique(String[r["config_name"] for r in suite_agg]))

    # ------------------------------------------------------------------
    # Per-config figures
    # ------------------------------------------------------------------
    for config_name in config_names
        series = []
        for opt in TREE_SEARCH_OPTS
            cs, losses, loss_std, ranks, rank_std =
                get_series(suite_agg, suite, config_name, opt)
            isempty(cs) || push!(series, (opt, cs, losses, loss_std, ranks, rank_std))
        end
        isempty(series) && continue

        title = "$(replace(suite, '_' => ' ')) — $(replace(config_name, '_' => ' '))"
        fig   = make_figure(title, series)
        save_fig(fig, joinpath(suite_dir, "$(config_name).pdf"))
    end

    # ------------------------------------------------------------------
    # Per-suite summary: average mean_final_loss and mean_rank across configs
    # ------------------------------------------------------------------
    summary_series = []
    for opt in TREE_SEARCH_OPTS
        # For each c value, average across all configs
        c_rows = Dict{Float64, Vector{Dict}}()
        for r in suite_agg
            r["optimizer"] != opt && continue
            !get(r, "uses_exploration_c", false) && continue
            c = Float64(r["c"])
            push!(get!(c_rows, c, Dict[]), r)
        end
        isempty(c_rows) && continue

        cs_sorted = sort(collect(keys(c_rows)))

        _mean(v) = isempty(v) ? NaN : sum(v) / length(v)

        cs       = cs_sorted
        losses   = [_mean([r["mean_final_loss"] for r in c_rows[c]]) for c in cs_sorted]
        loss_std = [_mean([get(r, "std_final_loss", 0.0) for r in c_rows[c]]) for c in cs_sorted]
        ranks    = [_mean([get(r, "mean_rank", NaN) for r in c_rows[c]
                           if !isnan(get(r, "mean_rank", NaN))]) for c in cs_sorted]
        rank_std = [_mean([get(r, "std_rank", 0.0) for r in c_rows[c]]) for c in cs_sorted]

        push!(summary_series, (opt, cs, losses, loss_std, ranks, rank_std))
    end

    if !isempty(summary_series)
        title = "$(replace(suite, '_' => ' ')) — summary (avg across $(length(config_names)) configs)"
        fig   = make_figure(title, summary_series)
        save_fig(fig, joinpath(output_dir, "$(suite)_summary.pdf"))
    end
end

# ============================================================================
# Combine all PDFs into one file
# ============================================================================

combined_pdf = joinpath(output_dir, "all_plots.pdf")
println("\nAttempting to combine $(length(all_pdfs)) PDFs → all_plots.pdf ...")

function try_combine_pdfs(pdfs, outpath)
    # Try ghostscript
    gs_bin = Sys.which("gs")
    if !isnothing(gs_bin)
        cmd = Cmd([gs_bin, "-dBATCH", "-dNOPAUSE", "-q",
                   "-sDEVICE=pdfwrite", "-sOutputFile=$outpath", pdfs...])
        try
            run(cmd); return true
        catch; end
    end
    # Try pdfunite (poppler-utils)
    pu_bin = Sys.which("pdfunite")
    if !isnothing(pu_bin)
        cmd = Cmd([pu_bin, pdfs..., outpath])
        try
            run(cmd); return true
        catch; end
    end
    return false
end

if !isempty(all_pdfs)
    if try_combine_pdfs(all_pdfs, combined_pdf)
        println("✓ Combined PDF: $combined_pdf")
    else
        println("⚠  Could not combine PDFs (install ghostscript or poppler-utils).")
        println("   Individual PDFs are in: $output_dir")
        println("   gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=all_plots.pdf \\")
        println("      $(join(all_pdfs, " \\\n      "))")
    end
end

println("\nDone! $(length(all_pdfs)) figures saved to $output_dir")
