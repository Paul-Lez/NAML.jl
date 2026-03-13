"""
Polynomial Learning Experiment Runner

Learn polynomial coefficients (a₀, ..., aₙ) via p-adic optimization, comparing
multiple optimizers across varying hyperparameters. Results are saved to JSON.

Usage:
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl [flags]

Flags:
    --quick      Reduced epochs (5) and simulations (10) for quick testing
    --save       Save results to JSON file (default filename with timestamp)
    --config     Load experiment configurations from config.jl
    --paper      Use paper-ready configurations from paper_config.jl
    --epochs N   Set number of epochs (default: 20)
    --samples N  Override number of samples per config
    --output F   Specify output JSON filename

Examples:
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --quick --save
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --config --save
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --paper --save
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl --epochs 50 --save --output results.json
"""

include("../../../src/NAML.jl")
include("../util.jl")

using Oscar
using .NAML
using Printf
using Dates
using Random

# ============================================================================
# Parse command line arguments
# ============================================================================

quick_mode = "--quick" in ARGS
save_results = "--save" in ARGS
use_config_file = "--config" in ARGS
use_paper_config = "--paper" in ARGS

# Parse --epochs N
global n_epochs = quick_mode ? 5 : 20
for (i, arg) in enumerate(ARGS)
    if arg == "--epochs" && i < length(ARGS)
        global n_epochs = parse(Int, ARGS[i+1])
    end
end

# Parse --output filename
global output_filename = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--output" && i < length(ARGS)
        global output_filename = ARGS[i+1]
    end
end

# Parse --samples N
global n_samples_override = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--samples" && i < length(ARGS)
        global n_samples_override = parse(Int, ARGS[i+1])
    end
end

# ============================================================================
# Experiment Configuration
# ============================================================================

if use_paper_config
    include("paper_config.jl")
    configs = experiment_configs
    println("Loaded PAPER-READY experiment configurations from paper_config.jl")
elseif use_config_file
    include("config.jl")
    configs = experiment_configs
    println("Loaded experiment configurations from config.jl")
else
    # Default configurations
    configs = [
        Dict("name" => "p2_deg2_3pts", "prime" => 2, "prec" => 20,
             "degree" => 2, "n_points" => 3, "num_samples" => 3),
        Dict("name" => "p2_deg3_4pts", "prime" => 2, "prec" => 20,
             "degree" => 3, "n_points" => 4, "num_samples" => 3),
        Dict("name" => "p3_deg3_4pts", "prime" => 3, "prec" => 15,
             "degree" => 3, "n_points" => 4, "num_samples" => 3),
    ]
end

# Apply samples override if specified
if !isnothing(n_samples_override)
    for config in configs
        config["num_samples"] = n_samples_override
    end
    println("Overriding num_samples to $n_samples_override for all configs")
end

# ============================================================================
# Optimizer definitions
# ============================================================================

"""
Return a dict of optimizer name => initializer function.

Each initializer takes (param, loss) and returns an OptimSetup.
The optimizer names encode the hyperparameters used.
"""
function get_optimizer_configs(; quick::Bool=false)
    return Dict(
        "Random" => Dict(
            "type" => "Random",
            "params" => Dict("degree" => 1),
            "init" => (param, loss) -> begin
                NAML.random_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First" => Dict(
            "type" => "Best-First",
            "params" => Dict("strict" => false, "degree" => 1),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First-branch2" => Dict(
            "type" => "Best-First",
            "params" => Dict("strict" => false, "degree" => 2),
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 2))
            end
        ),
        "MCTS-50" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => quick ? 10 : 50,
                             "exploration_constant" => 1.41, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick ? 10 : 50,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=1
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-100" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => quick ? 20 : 100,
                             "exploration_constant" => 2, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick ? 20 : 100,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=1
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-200" => Dict(
            "type" => "MCTS",
            "params" => Dict("num_simulations" => quick ? 40 : 200,
                             "exploration_constant" => 1.41, "degree" => 1),
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick ? 40 : 200,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=1
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-50" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => quick ? 10 : 50,
                             "exploration_constant" => 1.41, "degree" => 1,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick ? 10 : 50,
                    exploration_constant=1.41,
                    degree=1,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-100" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => quick ? 20 : 100,
                             "exploration_constant" => 1.41, "degree" => 1,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick ? 20 : 100,
                    exploration_constant=1.41,
                    degree=1,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-200" => Dict(
            "type" => "DAG-MCTS",
            "params" => Dict("num_simulations" => quick ? 40 : 200,
                             "exploration_constant" => 1.41, "degree" => 1,
                             "persist_table" => true),
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick ? 40 : 200,
                    exploration_constant=1.41,
                    degree=1,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DOO" => Dict(
            "type" => "DOO",
            "params" => Dict("max_depth" => quick ? 10 : 15, "degree" => 1),
            "init" => (param, loss) -> begin
                # Get prime from parameter polydisc
                p = Float64(NAML.prime(param))
                # Delta function: p^(-h) for depth h
                delta = h -> p^(-h)
                config = NAML.DOOConfig(
                    delta=delta,
                    max_depth=quick ? 10 : 15,
                    degree=1,
                    strict=false
                )
                NAML.doo_descent_init(param, loss, 1, config)
            end
        ),
        "Best-First-Gradient" => Dict(
            "type" => "Best-First-Gradient",
            "params" => Dict("degree" => 1),
            "init" => (param, loss) -> begin
                NAML.gradient_descent_init(param, loss, 1, 1)
            end
        ),
    )
end

# Canonical ordering for display (shared across all experiments)
const OPTIMIZER_ORDER = ["Random", "Best-First", "Best-First-branch2", "MCTS-50", "MCTS-100", "MCTS-200", "DAG-MCTS-50", "DAG-MCTS-100", "DAG-MCTS-200", "DOO", "Best-First-Gradient"]

# ============================================================================
# Run a single sample (one random problem instance)
# ============================================================================

function run_single_sample(config::Dict, sample_num::Int)
    p = config["prime"]
    prec = config["prec"]
    degree = config["degree"]
    n_points = config["n_points"]

    K = PadicField(p, prec)

    # Generate distinct random x values
    x_values = Vector{PadicFieldElem}()
    max_attempts = n_points * 100
    attempts = 0
    while length(x_values) < n_points && attempts < max_attempts
        x = generate_random_padic(p, prec, 0, 8)
        if !any(existing_x -> existing_x == x, x_values)
            push!(x_values, x)
        end
        attempts += 1
    end
    if length(x_values) < n_points
        error("Could not generate $n_points distinct points")
    end

    # Generate random y values (p-adic)
    y_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]
    data = collect(zip(x_values, y_values))

    # Create loss (p-adic output, no cutoff)
    loss = polynomial_to_linear_loss(data, degree, nothing)

    # Initial parameters at Gauss point
    initial_param = generate_gauss_point(degree + 1, K)
    initial_loss = loss.eval([initial_param])[1]

    # Get optimizer configs
    opt_configs = get_optimizer_configs(quick=quick_mode)

    # Results for this sample
    sample_results = Dict{String, Any}()
    sample_results["sample_num"] = sample_num
    sample_results["initial_loss"] = initial_loss

    # Store data info (as floats for JSON serializability)
    sample_results["data"] = Dict(
        "x_abs_values" => [Float64(abs(x)) for x in x_values],
        "y_abs_values" => [Float64(abs(y)) for y in y_values],
    )

    sample_results["optimizers"] = Dict{String, Any}()

    # Run each optimizer
    for opt_name in OPTIMIZER_ORDER
        opt_setup = opt_configs[opt_name]
        try
            # Wrap loss with evaluation counting
            counted_loss, eval_counter = wrap_loss_with_counting(loss)

            optim = opt_setup["init"](initial_param, counted_loss)

            losses = Float64[]
            t_start = time()

            for epoch in 1:n_epochs
                current_loss = NAML.eval_loss(optim)
                push!(losses, current_loss)
                NAML.step!(optim)
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            # Subtract monitoring eval_loss calls: n_epochs in-loop + 1 final, each length 1
            monitoring_evals = n_epochs + 1
            total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

            sample_results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "losses" => losses,
                "improvement" => initial_loss - final_loss,
                "improvement_ratio" => (initial_loss > 0) ?
                    (initial_loss - final_loss) / initial_loss : 0.0,
                "hyperparameters" => opt_setup["params"],
                "total_evals" => total_optimizer_evals,
            )

        catch e
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    return sample_results
end

# ============================================================================
# Run a single experiment (multiple samples for one config)
# ============================================================================

function run_single_experiment(config::Dict)
    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("="^70)
    println("  Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"]), Samples: $(config["num_samples"])")
    println("-"^70)

    results = Dict{String, Any}()
    results["config"] = config
    results["samples"] = []

    for sample in 1:config["num_samples"]
        println("\n  [Sample $sample/$(config["num_samples"])]")

        try
            sample_result = run_single_sample(config, sample)
            push!(results["samples"], sample_result)

            # Brief summary
            println(@sprintf("    Initial loss: %.6e", sample_result["initial_loss"]))
            for opt_name in OPTIMIZER_ORDER
                if haskey(sample_result["optimizers"], opt_name)
                    opt_result = sample_result["optimizers"][opt_name]
                    if !haskey(opt_result, "error")
                        println(@sprintf("    %-15s Final: %.6e  (%.1f%% improvement, %.2fs)",
                            opt_name, opt_result["final_loss"],
                            opt_result["improvement_ratio"] * 100,
                            opt_result["time"]))
                    else
                        println("    $opt_name: ERROR - $(opt_result["error"])")
                    end
                end
            end
        catch e
            println("    Sample $sample failed: $e")
            push!(results["samples"], Dict("sample_num" => sample, "error" => string(e)))
        end
    end

    # Compute aggregate statistics
    compute_aggregate_stats!(results)

    return results
end

# ============================================================================
# Aggregate statistics across samples
# ============================================================================

_mean(x) = sum(x) / length(x)
function _std(x)
    m = _mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

function compute_aggregate_stats!(results::Dict)
    samples = results["samples"]
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    if isempty(valid_samples)
        results["aggregate"] = Dict("error" => "No valid samples")
        return
    end

    results["aggregate"] = Dict{String, Any}()

    for opt_name in OPTIMIZER_ORDER
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
            agg = Dict(
                "mean_final_loss" => _mean(final_losses),
                "std_final_loss" => length(opt_data) > 1 ? _std(final_losses) : 0.0,
                "min_final_loss" => minimum(final_losses),
                "max_final_loss" => maximum(final_losses),
                "mean_improvement" => _mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => _mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => _mean([d["time"] for d in opt_data]),
                "n_valid" => length(opt_data),
                "hyperparameters" => opt_data[1]["hyperparameters"],
            )
            if haskey(opt_data[1], "total_evals")
                agg["mean_total_evals"] = _mean([d["total_evals"] for d in opt_data])
            end
            results["aggregate"][opt_name] = agg
        end
    end
end

# ============================================================================
# Main execution
# ============================================================================

# Set random seed for reproducibility
Random.seed!(42)

println("\n" * "="^70)
println("POLYNOMIAL LEARNING EXPERIMENT RUNNER")
println("="^70)
println("Start time: $(Dates.now())")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")
println("Quick mode: $quick_mode")
println("Random seed: 42 (for reproducibility)")
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
        println("\nExperiment $(config["name"]) failed: $e")
        push!(all_results, Dict("config" => config, "error" => string(e)))
    end
end

# ============================================================================
# Summary Table
# ============================================================================

println("\n\n" * "="^70)
println("SUMMARY TABLE")
println("="^70)

for (i, result) in enumerate(all_results)
    if haskey(result, "error")
        println("\nExperiment $(i): $(result["config"]["name"]) - FAILED")
        continue
    end

    config = result["config"]
    println("\nExperiment $(i): $(config["name"])")
    println("  p=$(config["prime"]), degree=$(config["degree"]), " *
            "points=$(config["n_points"]), samples=$(config["num_samples"])")

    if haskey(result, "aggregate") && !haskey(result["aggregate"], "error")
        println()
        println(@sprintf("  %-15s %15s %12s %12s %12s",
            "Optimizer", "Mean Final", "Std", "Improv %", "Time (s)"))
        println("  " * "-"^65)

        for opt_name in OPTIMIZER_ORDER
            if haskey(result["aggregate"], opt_name)
                agg = result["aggregate"][opt_name]
                println(@sprintf("  %-15s %15.6e %12.2e %11.1f%% %12.2f",
                    opt_name, agg["mean_final_loss"], agg["std_final_loss"],
                    agg["mean_improvement_ratio"] * 100, agg["mean_time"]))
            end
        end
    end
end

println("\n" * "="^70)
println("End time: $(Dates.now())")
println("="^70)

# ============================================================================
# Save results to JSON
# ============================================================================

if save_results
    try
        using JSON

        # Build output
        json_output = Dict{String, Any}()
        json_output["metadata"] = Dict(
            "timestamp" => string(Dates.now()),
            "n_epochs" => n_epochs,
            "quick_mode" => quick_mode,
            "optimizer_order" => OPTIMIZER_ORDER,
        )

        json_experiments = []
        for result in all_results
            json_result = Dict{String, Any}()
            json_result["config"] = result["config"]

            if haskey(result, "error")
                json_result["error"] = result["error"]
            else
                json_result["samples"] = result["samples"]
                if haskey(result, "aggregate")
                    json_result["aggregate"] = result["aggregate"]
                end
            end

            push!(json_experiments, json_result)
        end
        json_output["experiments"] = json_experiments

        # Determine filename
        if isnothing(output_filename)
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            global output_filename = "poly_learning_results_$(timestamp).json"
        end
        filepath = joinpath(@__DIR__, output_filename)

        open(filepath, "w") do f
            JSON.print(f, json_output, 2)
        end

        println("\nResults saved to: $filepath")
    catch e
        if e isa ArgumentError && occursin("Package JSON not found", string(e))
            println("\nWarning: JSON package not installed. Cannot save results.")
            println("  Install with: julia -e 'using Pkg; Pkg.add(\"JSON\")'")
        else
            println("\nError saving results: $e")
        end
    end
end

println("\nAll experiments complete!")
