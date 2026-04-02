"""
Sanity Check Runner: MCTS vs DAG-MCTS Comparison

Systematically compares MCTS and DAG-MCTS across:
- Simulation counts: 50, 100, 200
- Degree parameters: 1, 2, and dimension-matched

Usage:
    julia --project=. experiments/paper/polynomial_learning/sanity_run.jl
"""

include("../../../src/NAML.jl")
include("../util.jl")

using Oscar
using .NAML
using Printf
using Dates

# ============================================================================
# Configuration
# ============================================================================

include("sanity_config.jl")
configs = experiment_configs

n_epochs = 20
println("Loaded sanity check configurations")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")

# ============================================================================
# Optimizer configurations: MCTS vs DAG-MCTS
# ============================================================================

function get_sanity_optimizer_configs(dimension::Int)
    opts = Dict{String, Any}()

    # Degrees to test: 1, 2 (if valid), and dimension
    test_degrees = unique(filter(d -> d <= dimension, [1, 2, dimension]))

    # For each simulation count
    for n_sims in [50, 100, 200]
        # For each degree
        for deg in test_degrees
            # MCTS
            name_mcts = "MCTS-$(n_sims)-deg$(deg)"
            opts[name_mcts] = Dict(
                "type" => "MCTS",
                "params" => Dict("num_simulations" => n_sims,
                                 "exploration_constant" => 1.41,
                                 "degree" => deg),
                "init" => (param, loss) -> begin
                    config = NAML.MCTSConfig(
                        num_simulations=n_sims,
                        exploration_constant=1.41,
                        selection_mode=NAML.BestValue,
                        degree=deg
                    )
                    NAML.mcts_descent_init(param, loss, config)
                end
            )

            # DAG-MCTS
            name_dag = "DAG-MCTS-$(n_sims)-deg$(deg)"
            opts[name_dag] = Dict(
                "type" => "DAG-MCTS",
                "params" => Dict("num_simulations" => n_sims,
                                 "exploration_constant" => 1.41,
                                 "degree" => deg,
                                 "persist_table" => true),
                "init" => (param, loss) -> begin
                    config = NAML.DAGMCTSConfig(
                        num_simulations=n_sims,
                        exploration_constant=1.41,
                        degree=deg,
                        persist_table=true,
                        selection_mode=NAML.BestValue
                    )
                    NAML.dag_mcts_descent_init(param, loss, config)
                end
            )
        end
    end

    return opts
end

# Canonical ordering for display
function get_optimizer_order(dimension::Int)
    test_degrees = unique(filter(d -> d <= dimension, [1, 2, dimension]))
    order = String[]
    for n_sims in [50, 100, 200]
        for deg in test_degrees
            push!(order, "MCTS-$(n_sims)-deg$(deg)")
            push!(order, "DAG-MCTS-$(n_sims)-deg$(deg)")
        end
    end
    return order
end

# ============================================================================
# Helper functions
# ============================================================================

mean(x) = sum(x) / length(x)
function std_dev(x)
    m = mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

# ============================================================================
# Run single sample
# ============================================================================

function run_single_sample(config::Dict, sample_num::Int, opt_configs::Dict)
    p = config["prime"]
    prec = config["prec"]
    degree = config["degree"]
    n_points = config["n_points"]

    K = PadicField(p, prec)

    # Generate random polynomial interpolation problem
    # Need: 1 variable for x, and degree+1 variables for coefficients
    R, vars = polynomial_ring(K, vcat(["x"], ["a$i" for i in 0:degree]))
    x = vars[1]
    coeffs = vars[2:end]  # a0, a1, ..., a_degree
    f_poly = sum(coeffs[i+1] * x^i for i in 0:degree)
    f = AbsolutePolynomialSum([f_poly])

    # Create model (x is data, coefficients are parameters)
    param_info = [i == 1 for i in 1:(degree+2)]  # x is data, rest are params
    model = AbstractModel(f, param_info)

    # Generate training data
    data = [(generate_random_padic(p, prec, 0, 8), generate_random_padic(p, prec, 0, 8))
            for _ in 1:n_points]

    # Create loss
    loss = MSE_loss_init(model, data)

    # Initialize starting point
    initial_param = ValuationPolydisc([generate_random_padic(p, prec, 0, 8) for _ in 1:(degree+1)],
                                      [5 for _ in 1:(degree+1)])
    initial_loss = loss.eval([initial_param])[1]

    # Results storage
    sample_results = Dict{String, Any}()
    sample_results["sample_num"] = sample_num
    sample_results["initial_loss"] = initial_loss
    sample_results["optimizers"] = Dict{String, Any}()

    # Run each optimizer
    for (opt_name, opt_setup) in opt_configs
        try
            optim = opt_setup["init"](initial_param, loss)

            losses = Float64[]
            t_start = time()

            for epoch in 1:n_epochs
                current_loss = NAML.eval_loss(optim)
                push!(losses, current_loss)
                NAML.step!(optim)
                NAML.has_converged(optim) && break
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            sample_results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "losses" => losses,
                "improvement" => initial_loss - final_loss,
                "improvement_ratio" => (initial_loss > 0) ? (initial_loss - final_loss) / initial_loss : 0.0
            )

        catch e
            println("    ✗ Error in $opt_name: $e")
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    return sample_results
end

# ============================================================================
# Run single experiment
# ============================================================================

function run_single_experiment(config::Dict)
    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("="^70)
    println("Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"]), Samples: $(config["num_samples"])")

    dimension = config["degree"] + 1  # Number of coefficients
    println("Dimension (num coefficients): $dimension")
    println("-"^70)

    # Get optimizer configs for this dimension
    opt_configs = get_sanity_optimizer_configs(dimension)
    opt_order = get_optimizer_order(dimension)

    results = Dict{String, Any}()
    results["config"] = config
    results["dimension"] = dimension
    results["samples"] = []

    # Run multiple samples
    for sample in 1:config["num_samples"]
        println("\n  [Sample $sample/$(config["num_samples"])]")

        try
            sample_result = run_single_sample(config, sample, opt_configs)
            push!(results["samples"], sample_result)

            # Print brief summary
            println(@sprintf("    Initial: %.6e", sample_result["initial_loss"]))

        catch e
            println("    ✗ Sample $sample failed: $e")
            push!(results["samples"], Dict("sample_num" => sample, "error" => string(e)))
        end
    end

    # Compute aggregate statistics
    compute_aggregate_stats!(results, opt_order)

    return results
end

# ============================================================================
# Compute aggregate statistics
# ============================================================================

function compute_aggregate_stats!(results::Dict, opt_order::Vector{String})
    samples = results["samples"]
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    if isempty(valid_samples)
        results["aggregate"] = Dict("error" => "No valid samples")
        return
    end

    results["aggregate"] = Dict{String, Any}()

    for opt_name in opt_order
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
            results["aggregate"][opt_name] = Dict(
                "mean_final_loss" => mean([d["final_loss"] for d in opt_data]),
                "mean_improvement" => mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => mean([d["time"] for d in opt_data]),
                "std_final_loss" => length(opt_data) > 1 ? std_dev([d["final_loss"] for d in opt_data]) : 0.0,
            )
        end
    end
end

# ============================================================================
# Run all experiments
# ============================================================================

println("\n" * "="^70)
println("SANITY CHECK: MCTS vs DAG-MCTS COMPARISON")
println("="^70)
println("Start time: $(Dates.now())")
println("="^70)

all_results = []

for (i, config) in enumerate(configs)
    println("\n\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs))")
    println("#"^70)

    try
        result = run_single_experiment(config)
        push!(all_results, result)
    catch e
        println("\n✗ Experiment $(config["name"]) failed with error:")
        println("  $e")
        push!(all_results, Dict("config" => config, "error" => string(e)))
    end
end

# ============================================================================
# Summary Analysis
# ============================================================================

println("\n\n" * "="^70)
println("SUMMARY: MCTS vs DAG-MCTS COMPARISON")
println("="^70)

for (i, result) in enumerate(all_results)
    if haskey(result, "error")
        println("\nExperiment $(i): $(result["config"]["name"]) - FAILED")
        continue
    end

    config = result["config"]
    dimension = result["dimension"]

    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("Dimension: $dimension, Degree: $(config["degree"]), Prime: $(config["prime"])")
    println("="^70)

    if !haskey(result, "aggregate") || haskey(result["aggregate"], "error")
        println("No valid aggregate statistics")
        continue
    end

    agg = result["aggregate"]

    # Compare MCTS vs DAG-MCTS for each configuration
    println("\nComparison by simulation count and degree:")
    println("-"^70)

    test_degrees = unique(filter(d -> d <= dimension, [1, 2, dimension]))
    for n_sims in [50, 100, 200]
        for deg in test_degrees
            mcts_name = "MCTS-$(n_sims)-deg$(deg)"
            dag_name = "DAG-MCTS-$(n_sims)-deg$(deg)"

            if haskey(agg, mcts_name) && haskey(agg, dag_name)
                mcts_stats = agg[mcts_name]
                dag_stats = agg[dag_name]

                println("\n  Sims=$n_sims, Degree=$deg:")
                println(@sprintf("    MCTS:     Final=%.6e  Improv=%.6e (%.1f%%)  Time=%.2fs",
                    mcts_stats["mean_final_loss"],
                    mcts_stats["mean_improvement"],
                    mcts_stats["mean_improvement_ratio"] * 100,
                    mcts_stats["mean_time"]))
                println(@sprintf("    DAG-MCTS: Final=%.6e  Improv=%.6e (%.1f%%)  Time=%.2fs",
                    dag_stats["mean_final_loss"],
                    dag_stats["mean_improvement"],
                    dag_stats["mean_improvement_ratio"] * 100,
                    dag_stats["mean_time"]))

                # Compute relative performance
                if mcts_stats["mean_final_loss"] > 0
                    loss_ratio = dag_stats["mean_final_loss"] / mcts_stats["mean_final_loss"]
                    time_ratio = dag_stats["mean_time"] / mcts_stats["mean_time"]

                    println(@sprintf("    Ratio:    Loss=%.3fx  Time=%.3fx", loss_ratio, time_ratio))

                    if loss_ratio < 1.0
                        println("    → DAG-MCTS achieves better loss (lower is better)")
                    elseif loss_ratio > 1.0
                        println("    → MCTS achieves better loss")
                    else
                        println("    → Similar performance")
                    end
                end
            end
        end
    end
end

println("\n" * "="^70)
println("End time: $(Dates.now())")
println("="^70)

# ============================================================================
# Save results
# ============================================================================

try
    using JSON

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    filename = "sanity_check_results_$(timestamp).json"
    filepath = joinpath(@__DIR__, filename)

    open(filepath, "w") do f
        JSON.print(f, all_results, 2)
    end

    println("\n✓ Results saved to: $filepath")
catch e
    if e isa ArgumentError && occursin("Package JSON not found", string(e))
        println("\n⚠ Warning: JSON package not installed. Cannot save results.")
    else
        println("\n✗ Error saving results: $e")
    end
end

println("\n✓ Sanity check complete!")
