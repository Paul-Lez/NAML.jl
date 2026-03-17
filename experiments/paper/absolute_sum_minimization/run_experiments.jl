"""
Absolute Sum Minimization Experiment Runner

Minimize sums of absolute polynomials: |f₁(x)| + |f₂(x)| + ... + |fₙ(x)|
where f₁, ..., fₙ are polynomials with random fixed coefficients.

Run experiments with varying:
- Number of polynomials in the sum
- Number of variables (dimension)
- Polynomial degrees
- Base fields (different primes)

Usage:
    julia --project=. run_experiments.jl [--quick] [--save] [--config] [--paper]

Flags:
    --quick: Use reduced epochs (5) and simulations (10) for quick testing
    --save: Save results to JSON file
    --config: Use experiment configurations from config.jl
    --paper: Use comprehensive paper-ready configurations from paper_config.jl
    --samples N: Override number of samples per config

Examples:
    julia --project=. run_experiments.jl --quick
    julia --project=. run_experiments.jl --config --save
    julia --project=. run_experiments.jl --paper --save
    julia --project=. run_experiments.jl --paper --quick --save
"""

include("../../../src/NAML.jl")
include("../util.jl")  # Use parent directory's util for generate_random_padic
include("util.jl")     # Local util for problem generation

using Oscar
using .NAML
using Printf
using Dates
using Random

# Parse command line arguments
quick_mode = "--quick" in ARGS
save_results = "--save" in ARGS
use_config_file = "--config" in ARGS

# Parse --epochs N
global n_epochs_arg = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--epochs" && i < length(ARGS)
        global n_epochs_arg = parse(Int, ARGS[i+1])
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

# Load configurations
use_paper_config = "--paper" in ARGS

if use_paper_config
    include("paper_config.jl")
    configs = experiment_configs
    println("Loaded PAPER-READY experiment configurations from paper_config.jl")
elseif use_config_file
    include("config.jl")
    configs = experiment_configs
    println("Loaded experiment configurations from config.jl")
else
    # Default configurations (small, fast experiments)
    configs = [
        Dict("name" => "p2_1var_2poly_lin_opt1", "prime" => 2, "prec" => 20,
             "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 2, "opt_degree" => 1),
        Dict("name" => "p2_1var_2poly_quad_opt1", "prime" => 2, "prec" => 20,
             "num_polys" => 2, "num_vars" => 1, "degree" => 2, "num_samples" => 2, "opt_degree" => 1),
    ]
end

# Set epochs based on mode
global n_epochs = quick_mode ? 5 : 20
if !isnothing(n_epochs_arg)
    global n_epochs = n_epochs_arg
end
if quick_mode
    println("="^70)
    println("QUICK MODE: Running with only $n_epochs epochs per optimizer")
    println("="^70)
end

# Apply samples override if specified
if !isnothing(n_samples_override)
    for config in configs
        config["num_samples"] = n_samples_override
    end
    println("Overriding num_samples to $n_samples_override for all configs")
end

# ============================================================================
# Optimizer configurations
# ============================================================================

function get_optimizer_configs(K, opt_degree)
    return Dict(
        "Random" => Dict(
            "init" => (param, loss) -> begin
                state = 1
                config = (false, opt_degree)
                NAML.random_descent_init(param, loss, state, config)
            end
        ),
        "Best-First" => Dict(
            "init" => (param, loss) -> begin
                state = 1
                config = (false, opt_degree)
                NAML.greedy_descent_init(param, loss, state, config)
            end
        ),
        "MCTS-50" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick_mode ? 10 : 50,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=opt_degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-100" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick_mode ? 20 : 100,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=opt_degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-50" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick_mode ? 10 : 50,
                    exploration_constant=1.41,
                    degree=opt_degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-100" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick_mode ? 20 : 100,
                    exploration_constant=1.41,
                    degree=opt_degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-200" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=quick_mode ? 40 : 200,
                    exploration_constant=1.41,
                    degree=opt_degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-200" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=quick_mode ? 40 : 200,
                    exploration_constant=1.41,
                    selection_mode=NAML.BestValue,
                    degree=opt_degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "Best-First-branch2" => Dict(
            "init" => (param, loss) -> begin
                state = 1
                config = (false, 2)
                NAML.greedy_descent_init(param, loss, state, config)
            end
        ),
        "DOO" => Dict(
            "init" => (param, loss) -> begin
                # Get prime from field K
                p = Float64(Oscar.prime(K))
                # Delta function: p^(-h) for depth h
                delta = h -> p^(-h)
                config = NAML.DOOConfig(
                    delta=delta,
                    max_depth=quick_mode ? 10 : 15,
                    degree=opt_degree,
                    strict=false
                )
                NAML.doo_descent_init(param, loss, 1, config)
            end
        ),
        "Best-First-Gradient" => Dict(
            "init" => (param, loss) -> begin
                NAML.gradient_descent_init(param, loss, 1, (false, opt_degree))
            end
        ),
    )
end

# Canonical ordering for display (shared across all experiments)
const OPTIMIZER_ORDER = ["Random", "Best-First", "Best-First-branch2", "MCTS-50", "MCTS-100", "MCTS-200", "DAG-MCTS-50", "DAG-MCTS-100", "DAG-MCTS-200", "DOO", "Best-First-Gradient"]

# ============================================================================
# Run single sample (one random problem instance)
# ============================================================================

function run_single_sample(config::Dict, sample_num::Int)
    p = config["prime"]
    prec = config["prec"]
    num_polys = config["num_polys"]
    num_vars = config["num_vars"]
    degree = config["degree"]
    opt_degree = config["opt_degree"]

    K = PadicField(p, prec)

    # Generate random problem
    loss = generate_random_absolute_sum_problem(p, prec, num_polys, num_vars, degree)

    # Initialize starting point (Gauss point)
    initial_param = generate_initial_point(num_vars, K)
    initial_loss = loss.eval([initial_param])[1]

    # Get optimizer configs
    opt_configs = get_optimizer_configs(K, opt_degree)

    # Results storage for this sample
    sample_results = Dict{String, Any}()
    sample_results["sample_num"] = sample_num
    sample_results["initial_loss"] = initial_loss
    sample_results["optimizers"] = Dict{String, Any}()

    # Run each optimizer
    for (opt_name, opt_setup) in opt_configs
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
                NAML.has_converged(optim) && break
            end

            t_end = time()
            elapsed = t_end - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses, final_loss)

            # Subtract monitoring eval_loss calls: length(losses) in-loop + 1 final
            monitoring_evals = length(losses)
            total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

            sample_results["optimizers"][opt_name] = Dict(
                "time" => elapsed,
                "final_loss" => final_loss,
                "losses" => losses,
                "improvement" => initial_loss - final_loss,
                "improvement_ratio" => (initial_loss > 0) ? (initial_loss - final_loss) / initial_loss : 0.0,
                "total_evals" => total_optimizer_evals,
            )

        catch e
            println("    ✗ Error in $opt_name: $e")
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    return sample_results
end

# ============================================================================
# Run single experiment (multiple samples)
# ============================================================================

function run_single_experiment(config::Dict)
    println("\n" * "="^70)
    println("Experiment: $(config["name"])")
    println("="^70)
    println("Prime: $(config["prime"]), Polynomials: $(config["num_polys"]), " *
            "Variables: $(config["num_vars"]), Degree: $(config["degree"])")
    println("Samples: $(config["num_samples"]), Optimization degree: $(config["opt_degree"])")
    println("-"^70)

    results = Dict{String, Any}()
    results["config"] = config
    results["samples"] = []

    # Run multiple samples
    for sample in 1:config["num_samples"]
        println("\n  [Sample $sample/$(config["num_samples"])]")

        try
            sample_result = run_single_sample(config, sample)
            push!(results["samples"], sample_result)

            # Print brief summary
            println(@sprintf("    Initial: %.6e", sample_result["initial_loss"]))
            for opt_name in OPTIMIZER_ORDER
                if haskey(sample_result["optimizers"], opt_name)
                    opt_result = sample_result["optimizers"][opt_name]
                    if !haskey(opt_result, "error")
                        println(@sprintf("    %-10s Final: %.6e (Δ: %.6e, %.1f%%)",
                            opt_name, opt_result["final_loss"], opt_result["improvement"],
                            opt_result["improvement_ratio"] * 100))
                    end
                end
            end

        catch e
            println("    ✗ Sample $sample failed: $e")
            push!(results["samples"], Dict("sample_num" => sample, "error" => string(e)))
        end
    end

    # Compute aggregate statistics
    compute_aggregate_stats!(results)

    return results
end

# ============================================================================
# Compute aggregate statistics across samples
# ============================================================================

function compute_aggregate_stats!(results::Dict)
    samples = results["samples"]
    valid_samples = filter(s -> !haskey(s, "error"), samples)

    if isempty(valid_samples)
        results["aggregate"] = Dict("error" => "No valid samples")
        return
    end

    results["aggregate"] = Dict{String, Any}()

    # Get optimizer names from first valid sample
    opt_names = keys(valid_samples[1]["optimizers"])

    for opt_name in opt_names
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
            agg = Dict(
                "mean_final_loss" => mean([d["final_loss"] for d in opt_data]),
                "mean_improvement" => mean([d["improvement"] for d in opt_data]),
                "mean_improvement_ratio" => mean([d["improvement_ratio"] for d in opt_data]),
                "mean_time" => mean([d["time"] for d in opt_data]),
                "std_final_loss" => length(opt_data) > 1 ? std([d["final_loss"] for d in opt_data]) : 0.0,
                "min_final_loss" => minimum([d["final_loss"] for d in opt_data]),
                "max_final_loss" => maximum([d["final_loss"] for d in opt_data]),
            )
            if haskey(opt_data[1], "total_evals")
                agg["mean_total_evals"] = mean([d["total_evals"] for d in opt_data])
            end
            results["aggregate"][opt_name] = agg
        end
    end
end

# Helper functions for statistics
mean(x) = sum(x) / length(x)
function std(x)
    m = mean(x)
    sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

# ============================================================================
# Run all experiments
# ============================================================================

# Set random seed for reproducibility
Random.seed!(42)

println("\n" * "="^70)
println("ABSOLUTE SUM MINIMIZATION EXPERIMENT RUNNER")
println("="^70)
println("Start time: $(Dates.now())")
println("Number of experiments: $(length(configs))")
println("Epochs per optimizer: $n_epochs")
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
        println("\n✗ Experiment $(config["name"]) failed with error:")
        println("  $e")
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
    println("  Polynomials: $(config["num_polys"]), Variables: $(config["num_vars"]), " *
            "Degree: $(config["degree"]), Samples: $(config["num_samples"])")

    if haskey(result, "aggregate") && !haskey(result["aggregate"], "error")
        println()
        println(@sprintf("  %-15s %15s %15s %15s %12s",
            "Optimizer", "Mean Final", "Mean Improv.", "Improv. %", "Time (s)"))
        println("  " * "-"^75)

        for opt_name in OPTIMIZER_ORDER
            if haskey(result["aggregate"], opt_name)
                agg = result["aggregate"][opt_name]
                println(@sprintf("  %-15s %15.6e %15.6e %14.1f%% %12.2f",
                    opt_name, agg["mean_final_loss"], agg["mean_improvement"],
                    agg["mean_improvement_ratio"] * 100, agg["mean_time"]))
            end
        end
    else
        println("  No valid aggregate statistics")
    end
end

println("\n" * "="^70)
println("End time: $(Dates.now())")
println("="^70)

# ============================================================================
# Save results if requested
# ============================================================================

if save_results
    try
        using JSON

        # Convert results to JSON-serializable format
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

        json_output = Dict{String, Any}()
        json_output["metadata"] = Dict(
            "timestamp" => string(Dates.now()),
            "n_epochs" => n_epochs,
            "quick_mode" => quick_mode,
            "optimizer_order" => OPTIMIZER_ORDER,
        )
        json_output["experiments"] = json_experiments

        if isnothing(output_filename)
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            global output_filename = "absolute_sum_results_$(timestamp).json"
        end
        filepath = joinpath(@__DIR__, output_filename)

        open(filepath, "w") do f
            JSON.print(f, json_output, 2)
        end

        println("\n✓ Results saved to: $filepath")
    catch e
        if e isa ArgumentError && occursin("Package JSON not found", string(e))
            println("\n⚠ Warning: JSON package not installed. Cannot save results.")
            println("  To enable JSON saving, run: julia -e 'using Pkg; Pkg.add(\"JSON\")'")
        else
            println("\n✗ Error saving results: $e")
        end
    end
end

println("\n✓ All experiments complete!")
