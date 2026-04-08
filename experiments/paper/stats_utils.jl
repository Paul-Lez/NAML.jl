"""
Shared statistics utilities for the NAML paper experiment infrastructure.

Provides:
1. Basic statistics (mean, std)
2. Per-sample optimizer ranking
3. Per-experiment aggregate statistics
4. Cross-experiment global ranking

Used by make_stats.jl to compute all stats from raw experiment JSON.
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
# Per-sample rankings
# ============================================================================

"""
    compute_sample_rankings!(sample_results::Dict)

Rank optimizers within a sample by final_loss (lower = rank 1).
Adds a "rank" field to each valid optimizer result.
Ties share the minimum rank (competition ranking: 1,1,3 not 1.5,1.5,3).
"""
function compute_sample_rankings!(sample_results::Dict)
    optimizers = sample_results["optimizers"]
    valid_opts = [(name, res["final_loss"]) for (name, res) in optimizers
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
            optimizers[valid_opts[k][1]]["rank"] = Float64(i)
        end
        i = j
    end
end


# ============================================================================
# Per-experiment aggregate statistics
# ============================================================================

"""
    compute_aggregate_stats(samples, optimizer_order; extra_fields=[]) -> Dict

Compute aggregate statistics across samples for each optimizer.

Returns a Dict mapping optimizer name => Dict of aggregate stats:
- mean_final_loss, std_final_loss, min_final_loss, max_final_loss
- mean_improvement, mean_improvement_ratio
- mean_time, std_time
- mean_total_evals (if available)
- mean_rank, std_rank (if ranking data available)
- Any fields listed in extra_fields (e.g., "final_accuracy", "accuracy_improvement")
"""
function compute_aggregate_stats(samples::Vector, optimizer_order::Vector{String};
                                  extra_fields::Vector{String}=String[])
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    if isempty(valid_samples)
        return Dict("error" => "No valid samples")
    end

    aggregate = Dict{String, Any}()

    for opt_name in optimizer_order
        opt_data = []
        for sample in valid_samples
            if haskey(sample["optimizers"], opt_name)
                opt_result = sample["optimizers"][opt_name]
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

            # Extra fields (experiment-specific, e.g., accuracy)
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
    compute_global_ranking(experiments, optimizer_order) -> Dict

Compute average rank across all experiment configurations.
Returns a Dict mapping optimizer name => Dict("avg_rank" => ..., "n_configs" => ...).
"""
function compute_global_ranking(experiments::Vector, optimizer_order::Vector{String})
    global_ranks = Dict{String, Vector{Float64}}()

    for result in experiments
        if !haskey(result, "error") && haskey(result, "aggregate") && !haskey(result["aggregate"], "error")
            agg = result["aggregate"]
            for opt_name in optimizer_order
                if haskey(agg, opt_name) && haskey(agg[opt_name], "mean_rank")
                    if !haskey(global_ranks, opt_name)
                        global_ranks[opt_name] = Float64[]
                    end
                    push!(global_ranks[opt_name], agg[opt_name]["mean_rank"])
                end
            end
        end
    end

    return Dict{String, Any}(
        opt => Dict("avg_rank" => _mean(ranks), "n_configs" => length(ranks))
        for (opt, ranks) in global_ranks if !isempty(ranks)
    )
end
