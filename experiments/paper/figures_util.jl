"""
Figure-generation utilities for the NAML paper.

All plots use Plots.jl with the GR backend. Pure aggregation logic lives in
`stats_utils.jl`; this file is responsible only for turning aggregated numbers
into figures.

Each `generate_*` function returns a `Plots.Plot` so callers can either save it
(via `save_figure`) or further customise it.
"""

using Plots
gr()

# Display names, branching-factor helpers, and aggregation primitives.
include(joinpath(@__DIR__, "stats_utils.jl"))

# ============================================================================
# Theme — change these constants in one place to restyle every figure
# ============================================================================

"""Half an A4 page (210mm wide) at ≈100 dpi."""
const FIGURE_SIZE             = (420, 300)

"""Pastel categorical palette for optimizers. Swap to taste."""
const FIGURE_PALETTE          = palette(:Pastel1)

const FIGURE_BACKGROUND       = :white
const FIGURE_GRID_COLOR       = :gray85
const FIGURE_FONT_FAMILY      = "Helvetica"
const FIGURE_TITLE_FONT_SIZE  = 10
const FIGURE_GUIDE_FONT_SIZE  = 9
const FIGURE_TICK_FONT_SIZE   = 7
const FIGURE_LEGEND_FONT_SIZE = 7

"""Common keyword arguments applied to every figure."""
function _base_attrs(; title::String="")
    return (
        size              = FIGURE_SIZE,
        title             = title,
        background_color  = FIGURE_BACKGROUND,
        gridcolor         = FIGURE_GRID_COLOR,
        fontfamily        = FIGURE_FONT_FAMILY,
        titlefontsize     = FIGURE_TITLE_FONT_SIZE,
        guidefontsize     = FIGURE_GUIDE_FONT_SIZE,
        tickfontsize      = FIGURE_TICK_FONT_SIZE,
        legendfontsize    = FIGURE_LEGEND_FONT_SIZE,
        framestyle        = :box,
    )
end

"""One palette colour per optimizer (cycles if there are more optimizers than
palette entries)."""
function _optimizer_color(idx::Int)
    n = length(FIGURE_PALETTE)
    return FIGURE_PALETTE[mod1(idx, n)]
end

"""Plain-text version of `display_name` for plot labels — GR cannot render the
LaTeX dollar-math used by the table generators."""
figure_label(name) = replace(display_name(name), "\$" => "")

# ============================================================================
# Shared rendering helpers
# ============================================================================

"""
    _bar_per_optimizer(values::Dict, optimizer_order; ylabel, title, yscale)

Render a bar chart with one bar per optimizer that has data. Optimizers are
plotted in `optimizer_order`; those whose value is `NaN` are skipped.
"""
function _bar_per_optimizer(values::AbstractDict,
                            optimizer_order::AbstractVector;
                            ylabel::String,
                            title::String="",
                            yscale::Symbol=:identity)
    present = [opt for opt in optimizer_order if !isnan(values[opt])]
    isempty(present) && return plot(; title="No data", _base_attrs(title=title)...)

    heights = [values[opt] for opt in present]
    labels  = [figure_label(opt) for opt in present]
    colors  = [_optimizer_color(i) for i in eachindex(present)]

    return bar(
        labels, heights;
        ylabel    = ylabel,
        xrotation = 45,
        legend    = false,
        color     = colors,
        yscale    = yscale,
        _base_attrs(title=title)...
    )
end

"""
    _lines_by_branching_factor(series::Dict, optimizer_order; ylabel, title, yscale)

One line per optimizer; x-axis is branching factor, y-axis is the supplied
metric. Each entry in `series` is a sorted vector of `(branching_factor, value)`
pairs as produced by `mean_metric_by_branching_factor`.
"""
function _lines_by_branching_factor(series::AbstractDict,
                                    optimizer_order::AbstractVector;
                                    ylabel::String,
                                    title::String="",
                                    yscale::Symbol=:identity)
    plt = plot(;
        xlabel = "Branching factor",
        ylabel = ylabel,
        yscale = yscale,
        legend = :outerright,
        _base_attrs(title=title)...
    )

    for (i, opt) in enumerate(optimizer_order)
        pts = get(series, opt, Tuple{Int,Float64}[])
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        plot!(plt, xs, ys;
              label  = figure_label(opt),
              marker = :circle,
              color  = _optimizer_color(i),
              linewidth = 1.5,
              markersize = 4)
    end
    return plt
end

# ============================================================================
# Per-experiment-suite plots
# ============================================================================

"""
    generate_ranking_plot_per_experiment(experiments, optimizer_order; title)

Bar plot of the average rank of each optimizer across one experiment suite.
Lower is better.
"""
function generate_ranking_plot_per_experiment(experiments,
                                              optimizer_order::AbstractVector;
                                              title::String="")
    means = mean_metric_across_experiments(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_rank"))
    return _bar_per_optimizer(means, optimizer_order;
        ylabel = "Mean rank",
        title  = title)
end

"""
    generate_average_final_loss(experiments, optimizer_order; title)

Bar plot of the mean final loss of each optimizer across one experiment suite.
Plotted on a log scale because losses span many orders of magnitude.
"""
function generate_average_final_loss(experiments,
                                     optimizer_order::AbstractVector;
                                     title::String="")
    means = mean_metric_across_experiments(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_final_loss"))
    return _bar_per_optimizer(means, optimizer_order;
        ylabel = "Mean final loss",
        title  = title,
        yscale = :log10)
end

"""
    generate_number_of_evals_plot(experiments, optimizer_order; title)

Line plot — one line per optimizer — of mean function-evaluation count vs
branching factor. Y-axis is log-scaled because eval counts span many orders.
"""
function generate_number_of_evals_plot(experiments,
                                       optimizer_order::AbstractVector;
                                       title::String="")
    series = mean_metric_by_branching_factor(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_total_evals"))
    return _lines_by_branching_factor(series, optimizer_order;
        ylabel = "Mean function evaluations",
        title  = title,
        yscale = :log10)
end

"""
    generate_times_plot(experiments, optimizer_order; title)

Line plot — one line per optimizer — of mean wall-clock runtime vs branching
factor.
"""
function generate_times_plot(experiments,
                             optimizer_order::AbstractVector;
                             title::String="")
    series = mean_metric_by_branching_factor(experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_time"))
    return _lines_by_branching_factor(series, optimizer_order;
        ylabel = "Mean runtime (s)",
        title  = title,
        yscale = :log10)
end

# ============================================================================
# Cross-suite plot
# ============================================================================

"""
    generate_overall_ranking_plot(all_experiments, optimizer_order; title)

Bar plot of the mean rank of each optimizer across the *concatenation* of all
experiment suites' configurations.
"""
function generate_overall_ranking_plot(all_experiments,
                                       optimizer_order::AbstractVector;
                                       title::String="Overall mean rank")
    means = mean_metric_across_experiments(all_experiments, optimizer_order,
                s -> optimizer_metric(s, "mean_rank"))
    return _bar_per_optimizer(means, optimizer_order;
        ylabel = "Mean rank",
        title  = title)
end

# ============================================================================
# I/O
# ============================================================================

"""
    save_figure(plt, path)

Create the parent directory if needed and save `plt` to `path`. Format is
inferred from the extension by Plots.jl.
"""
function save_figure(plt, path::String)
    mkpath(dirname(path))
    savefig(plt, path)
    println("✓ Wrote $path")
    return path
end
