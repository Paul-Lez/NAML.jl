"""
Shared statistics utilities for the NAML paper experiment infrastructure.

Provides:
1. Basic statistics (mean, std)
2. Per-sample optimizer ranking
3. Per-experiment aggregate statistics
4. Cross-experiment global ranking
5. Display-name conventions for optimizers (shared by tables and figures)
6. Branching-factor helpers for grouping experiments by tree shape
7. Generic per-optimizer aggregation across experiments (used by both
   `table_utils.jl` and `figures_util.jl` to avoid duplication)

Used by `make_stats.jl`, `table_utils.jl`, and `figures_util.jl`.
"""

# ============================================================================
# Basic statistics
# ============================================================================

_mean(x) = sum(x) / length(x)

function _std(x)
    m = _mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end


# ============================================================================
# Display names and ordering (shared by table and figure generators)
# ============================================================================

const DISPLAY_NAMES = Dict(
    "Random"              => "Random",
    "Best-First"          => "Best First Value",
    "Best-First-branch2"  => "Best First Branch 2",
    "Best-First-Gradient" => "Best First Gradient",
    "DOO"                 => "DOO",
    "MCTS-k"              => "MCTS-\$k\$",
    "MCTS-5k"             => "MCTS-\$5k\$",
    "MCTS-10k"            => "MCTS-\$10k\$",
    "DAG-MCTS-k"          => "DAG-MCTS-\$k\$",
    "DAG-MCTS-5k"         => "DAG-MCTS-\$5k\$",
    "DAG-MCTS-10k"        => "DAG-MCTS-\$10k\$",

    # New suite-based names
    "MCTS-10k-deg1"       => "MCTS-\$10k\$ (deg 1)",
    "MCTS-10k-deg2"       => "MCTS-\$10k\$ (deg 2)",
    "DAG-MCTS-10k-deg1"   => "DAG-MCTS-\$10k\$ (deg 1)",
    "DAG-MCTS-10k-deg2"   => "DAG-MCTS-\$10k\$ (deg 2)",
    "Greedy-deg1"         => "Greedy (deg 1)",
    "Greedy-deg2"         => "Greedy (deg 2)",
    "Gradient-deg1"       => "Gradient (deg 1)",
    "Gradient-deg2"       => "Gradient (deg 2)",
)
display_name(n) = get(DISPLAY_NAMES, n, n)

const DISPLAY_ORDER = [
    "Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
    "MCTS-k", "MCTS-5k", "MCTS-10k",
    "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k",
    "MCTS-10k-deg1", "MCTS-10k-deg2", "DAG-MCTS-10k-deg1", "DAG-MCTS-10k-deg2",
    "Greedy-deg1", "Greedy-deg2", "Gradient-deg1", "Gradient-deg2",
    "DOO"
]


# ============================================================================
# Branching factor
# ============================================================================

"""
    get_branching_factor(prime, dimension, degree) -> Int

Number of children of a polydisc node in the search tree.

`prime` is the residue characteristic, `dimension` is the number of polydisc
coordinates (one per learnable coefficient), and `degree` is the search-tree
refinement degree (how many coordinate refinements happen per child step).

TODO(user): replace this default with the exact formula you want.
"""
function get_branching_factor(prime::Int, dimension::Int, degree::Int)
    return prime^dimension
end

"""
    config_dimension(config) -> Int

Polydisc dimension implied by an experiment config.

- For `polynomial_learning` and `function_learning`, the model is a degree-`n`
  polynomial with `n + 1` learnable coefficients, so `dimension = degree + 1`.
- For `absolute_sum_minimization` and `polynomial_solving`, the dimension is
  given directly by `num_vars`.
"""
function config_dimension(config::AbstractDict)
    if haskey(config, "num_vars")
        return Int(config["num_vars"])
    elseif haskey(config, "degree")
        return Int(config["degree"]) + 1
    else
        error("Cannot infer dimension from config: $(config)")
    end
end

"""
    config_branching_factor(config; refinement_degree=1) -> Int

Branching factor for an experiment config, computed via `get_branching_factor`.
The `refinement_degree` defaults to 1 (one coordinate refined per child step).
"""
function config_branching_factor(config::AbstractDict; refinement_degree::Int=1)
    return get_branching_factor(Int(config["prime"]), config_dimension(config),
                                refinement_degree)
end


# ============================================================================
# Generic per-optimizer aggregation across experiments
# ============================================================================

"""
    experiment_aggregate(exp; suite_name=nothing) -> Union{AbstractDict,Nothing}

Return the aggregate-stats block of an experiment, or `nothing` if the
experiment is errored / has no aggregate for the requested suite.

- If `suite_name === nothing`, look up the legacy flat `exp["aggregate"]`.
- Otherwise, look up `exp["suites_aggregate"][suite_name]`.
"""
function experiment_aggregate(exp; suite_name=nothing)
    haskey(exp, "error") && return nothing
    if suite_name === nothing
        haskey(exp, "aggregate") || return nothing
        agg = exp["aggregate"]
    else
        haskey(exp, "suites_aggregate") || return nothing
        haskey(exp["suites_aggregate"], suite_name) || return nothing
        agg = exp["suites_aggregate"][suite_name]
    end
    haskey(agg, "error") && return nothing
    return agg
end

"""
    valid_aggregates(experiments; suite_name=nothing) -> Vector

Filter experiments to those that contain a non-error aggregate block for the
requested suite (or for the legacy flat `aggregate` when `suite_name` is not
provided).
"""
valid_aggregates(experiments; suite_name=nothing) =
    filter(e -> experiment_aggregate(e; suite_name=suite_name) !== nothing,
           experiments)

"""
    optimizer_metric(opt_stats, key) -> Union{Float64,Nothing}

Look up `key` in an optimizer's aggregate-stats dict, returning `nothing` if
absent or in error state. Convenience wrapper used by extractor closures.
"""
function optimizer_metric(opt_stats, key::String)
    (opt_stats === nothing || haskey(opt_stats, "error")) && return nothing
    haskey(opt_stats, key) || return nothing
    return Float64(opt_stats[key])
end

"""
    mean_metric_across_experiments(experiments, optimizer_order, extractor;
                                    suite_name=nothing) -> Dict

For each optimizer, average the value returned by `extractor(opt_stats)` over
all valid experiments. Returns a `Dict{String,Float64}`; optimizers with no
data map to `NaN`.

When `suite_name` is provided, only the aggregate stats for that suite
(`exp["suites_aggregate"][suite_name]`) are inspected.
"""
function mean_metric_across_experiments(experiments, optimizer_order::AbstractVector,
                                         extractor; suite_name=nothing)
    sums = Dict{Any,Float64}(opt => 0.0 for opt in optimizer_order)
    counts = Dict{Any,Int}(opt => 0 for opt in optimizer_order)

    for exp in experiments
        agg = experiment_aggregate(exp; suite_name=suite_name)
        agg === nothing && continue
        for opt in optimizer_order
            opt_stats = get(agg, opt, nothing)
            v = extractor(opt_stats)
            v === nothing && continue
            sums[opt] += v
            counts[opt] += 1
        end
    end

    return Dict{Any,Float64}(
        opt => counts[opt] > 0 ? sums[opt] / counts[opt] : NaN
        for opt in optimizer_order
    )
end

"""
    mean_metric_by_branching_factor(experiments, optimizer_order, extractor;
                                    refinement_degree=1, suite_name=nothing)
        -> Dict{String, Vector{Tuple{Int,Float64}}}

For each optimizer, group experiments by branching factor and average
`extractor(opt_stats)` within each group. Returned vectors are sorted by
branching factor. As with `mean_metric_across_experiments`, `suite_name`
selects which per-suite aggregate to read.
"""
function mean_metric_by_branching_factor(experiments,
                                          optimizer_order::AbstractVector,
                                          extractor;
                                          refinement_degree::Int=1,
                                          suite_name=nothing)
    buckets = Dict{Any, Dict{Int, Vector{Float64}}}(
        opt => Dict{Int, Vector{Float64}}() for opt in optimizer_order
    )

    for exp in experiments
        agg = experiment_aggregate(exp; suite_name=suite_name)
        agg === nothing && continue
        haskey(exp, "config") || continue
        bf = config_branching_factor(exp["config"]; refinement_degree=refinement_degree)
        for opt in optimizer_order
            v = extractor(get(agg, opt, nothing))
            v === nothing && continue
            push!(get!(buckets[opt], bf, Float64[]), v)
        end
    end

    return Dict{Any, Vector{Tuple{Int,Float64}}}(
        opt => sort([(bf, _mean(vs)) for (bf, vs) in buckets[opt]], by=first)
        for opt in optimizer_order
    )
end

# ============================================================================
# Per-sample rankings
# ============================================================================

"""
    compute_sample_rankings!(optimizer_results::Dict)

Rank optimizers within a dictionary by final_loss (lower = rank 1).
Adds a "rank" field to each valid optimizer result.
Ties share the minimum rank (competition ranking: 1,1,3).
"""
function compute_sample_rankings!(optimizer_results::AbstractDict)
    valid_opts = [(name, res["final_loss"]) for (name, res) in optimizer_results
                  if !haskey(res, "error")]
    isempty(valid_opts) && return
    sort!(valid_opts, by=x -> x[2])
    n = length(valid_opts)
    i = 1
    while i <= n
        j = i
        while j <= n && valid_opts[j][2] == valid_opts[i][2]
            j += 1
        end
        for k in i:j-1
            optimizer_results[valid_opts[k][1]]["rank"] = Float64(i)
        end
        i = j
    end
end


# ============================================================================
# Per-experiment aggregate statistics
# ============================================================================

"""
    compute_aggregate_stats(samples_in_suite::Vector{Dict}, suite_name::String;
                            extra_fields::Vector{String}=String[]) -> Dict

Compute aggregate statistics across samples for a specific suite.
`samples_in_suite` is a Vector of optimizer results for this suite (one per sample).
"""
function compute_aggregate_stats(samples_in_suite::AbstractVector, suite_name::String;
                                  extra_fields::AbstractVector=String[])
    if isempty(samples_in_suite)
        return Dict("error" => "No samples in suite")
    end

    # Identify all optimizers in this suite
    all_opt_names = Set{String}()
    for sample in samples_in_suite
        for opt_name in keys(sample)
            push!(all_opt_names, opt_name)
        end
    end

    aggregate = Dict{String, Any}()

    for opt_name in sort(collect(all_opt_names))
        opt_data = []
        for sample in samples_in_suite
            if haskey(sample, opt_name)
                opt_result = sample[opt_name]
                if !haskey(opt_result, "error")
                    push!(opt_data, opt_result)
                end
            end
        end

        if !isempty(opt_data)
            final_losses = [d["final_loss"] for d in opt_data]

            agg = Dict{String, Any}(
                "mean_final_loss" => _mean(final_losses),
                "std_final_loss" => length(opt_data) > 1 ? _std(final_losses) : 0.0,
                "min_final_loss" => minimum(final_losses),
                "max_final_loss" => maximum(final_losses),
                "mean_improvement" => _mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => _mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => _mean([d["time"] for d in opt_data]),
                "std_time" => length(opt_data) > 1 ? _std([d["time"] for d in opt_data]) : 0.0,
                "n_valid" => length(opt_data),
            )

            # Eval counts
            if haskey(opt_data[1], "total_evals")
                agg["mean_total_evals"] = _mean([d["total_evals"] for d in opt_data])
            end

            # Rankings
            ranks = [d["rank"] for d in opt_data if haskey(d, "rank")]
            if !isempty(ranks)
                agg["mean_rank"] = _mean(ranks)
                agg["std_rank"] = length(ranks) > 1 ? _std(ranks) : 0.0
            end

            # Extra fields
            for field in extra_fields
                mean_key = "mean_$field"
                std_key = "std_$field"
                vals = [d[field] for d in opt_data if haskey(d, field)]
                if !isempty(vals)
                    agg[mean_key] = _mean(vals)
                    agg[std_key] = length(vals) > 1 ? _std(vals) : 0.0
                    agg["min_$field"] = minimum(vals)
                    agg["max_$field"] = maximum(vals)
                end
            end

            aggregate[opt_name] = agg
        end
    end

    return aggregate
end


# ============================================================================
# Global ranking across experiments
# ============================================================================

"""
    compute_global_ranking(experiments) -> Dict{String, Dict}

Compute average rank across all experiment configurations, grouped by suite.
Returns Dict{SuiteName => Dict{OptName => {avg_rank, n_configs}}}.
"""
function compute_global_ranking(experiments::AbstractVector)
    # SuiteName => OptName => Vector{Ranks}
    suite_global_ranks = Dict{String, Dict{String, Vector{Float64}}}()

    for result in experiments
        if haskey(result, "error") || !haskey(result, "suites_aggregate")
            continue
        end
        
        suites_agg = result["suites_aggregate"]
        for (suite_name, agg) in suites_agg
            if haskey(agg, "error")
                continue
            end
            
            if !haskey(suite_global_ranks, suite_name)
                suite_global_ranks[suite_name] = Dict{String, Vector{Float64}}()
            end
            
            opt_ranks = suite_global_ranks[suite_name]
            for (opt_name, stats) in agg
                if haskey(stats, "mean_rank")
                    if !haskey(opt_ranks, opt_name)
                        opt_ranks[opt_name] = Float64[]
                    end
                    push!(opt_ranks[opt_name], stats["mean_rank"])
                end
            end
        end
    end

    # Convert to results
    output = Dict{String, Any}()
    for (suite_name, opt_ranks) in suite_global_ranks
        suite_res = Dict{String, Any}()
        for (opt_name, ranks) in opt_ranks
            suite_res[opt_name] = Dict(
                "avg_rank" => _mean(ranks),
                "n_configs" => length(ranks)
            )
        end
        output[suite_name] = suite_res
    end

    return output
end
