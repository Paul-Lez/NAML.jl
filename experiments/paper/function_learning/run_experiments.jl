"""
Function Learning Experiment Runner

Learn binary classification functions using polynomial approximation with
cross-entropy loss. Tests learning various target functions (zero, one, random)
over p-adic inputs.

Usage:
    julia --project=. experiments/paper/function_learning/run_experiments.jl [FLAGS]

Flags:
    --quick       Reduced epochs and simulations for smoke testing
    --save        Save results to JSON file
    --config      Use configurations from config.jl
    --paper       Use paper-ready configurations from paper_config.jl
    --epochs N    Override number of epochs
    --output FILE Override output filename
    --samples N   Override number of samples per config
    --selection-mode M   MCTS selection mode: BestValue, VisitCount, or BestLoss
    --degree D    Override tree branching degree
    --description TEXT   Experiment description
    --git-commit HASH   Git commit hash
"""

# ============================================================================
# Setup
# ============================================================================

include("../../../src/NAML.jl")
include("../util.jl")
include("../experiment_utils.jl")

using Oscar
using .NAML

args = parse_experiment_args(ARGS)

# ============================================================================
# Default configurations
# ============================================================================

default_configs = [
    Dict("name" => "p2_deg2_zero", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 10, "target_fn" => "zero",
         "threshold" => 0.5, "scale" => 0.1, "num_samples" => 2),
    Dict("name" => "p2_deg2_one", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 10, "target_fn" => "one",
         "threshold" => 0.5, "scale" => 0.1, "num_samples" => 2),
]

load_config_file(@__DIR__, args)
configs = load_configs(args, default_configs)

# ============================================================================
# Classification helpers
# ============================================================================

"""
Create a loss function for learning a target function using polynomial
approximation with cross-entropy loss and sigmoid activation.
"""
function create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)
    p = Int(Oscar.prime(K))
    prec = Oscar.precision(K)
    x_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]

    if target_fn == "zero"
        y_values = [0.0 for _ in 1:n_points]
    elseif target_fn == "one"
        y_values = [1.0 for _ in 1:n_points]
    elseif target_fn == "random"
        y_values = [Float64(rand(0:1)) for _ in 1:n_points]
    else
        error("Unknown target function: $target_fn")
    end

    data = collect(zip(x_values, y_values))
    loss = polynomial_to_crossentropy_loss(data, degree, threshold, scale)
    return loss, data
end

"""
Compute classification accuracy for a polynomial classifier.
"""
function compute_accuracy(coeffs, data, threshold, scale)
    function eval_polynomial(coeffs, x)
        result = coeffs[1]
        x_power = x
        for i in 2:length(coeffs)
            result += coeffs[i] * x_power
            x_power *= x
        end
        return result
    end

    correct = 0
    for (x, y) in data
        poly_val = eval_polynomial(coeffs, x)
        val_float = Float64(abs(poly_val))
        z = (val_float - threshold) / scale
        prob = 1.0 / (1.0 + exp(-z))
        prediction = prob > 0.5 ? 1.0 : 0.0
        if prediction == y
            correct += 1
        end
    end
    return correct / length(data)
end

# ============================================================================
# Run single sample
# ============================================================================

function run_single_sample(config::Dict, sample_num::Int)
    p = config["prime"]
    prec = config["prec"]
    degree = config["degree"]
    n_points = config["n_points"]
    target_fn = config["target_fn"]
    threshold = config["threshold"]
    scale = config["scale"]

    K = PadicField(p, prec)

    # Create loss function
    loss, data = create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)

    # Initialize parameters at origin with radius 0
    num_params = degree + 1
    param_center = [K(0) for _ in 1:num_params]
    initial_param = NAML.ValuationPolydisc(param_center, [0 for _ in 1:num_params])
    initial_loss = loss.eval([initial_param])[1]

    # Compute initial accuracy
    initial_coeffs = [NAML.unwrap(c) for c in NAML.center(initial_param)]
    initial_accuracy = compute_accuracy(initial_coeffs, data, threshold, scale)

    # Get optimizer configs
    eff_degree = effective_degree(num_params, args.mcts_degree_override)
    opt_configs = get_optimizer_configs(
        quick=args.quick_mode, selection_mode=args.selection_mode,
        degree=eff_degree, prime=p, dim=num_params
    )

    # Post-run callback to compute accuracy
    post_run_fn = (optim) -> begin
        final_coeffs = [NAML.unwrap(c) for c in NAML.center(optim.param)]
        final_accuracy = compute_accuracy(final_coeffs, data, threshold, scale)
        Dict{String, Any}(
            "final_accuracy" => final_accuracy,
            "accuracy_improvement" => final_accuracy - initial_accuracy,
        )
    end

    # Run all optimizers serially inside this task. Parallelism lives at the
    # (config, sample) level above; running optimizers serially here avoids
    # nested threading and contention on the shared loss evaluator.
    optimizer_results = run_all_optimizers_serial(
        opt_configs, initial_param, loss, args.n_epochs;
        post_run_fn=post_run_fn
    )

    return Dict{String, Any}(
        "sample_num" => sample_num,
        "initial_loss" => initial_loss,
        "initial_accuracy" => initial_accuracy,
        "data" => Dict(
            "x_abs_values" => [Float64(abs(x)) for (x, _) in data],
            "y_values" => [y for (_, y) in data],
        ),
        "optimizers" => optimizer_results,
    )
end

# ============================================================================
# Print helpers
# ============================================================================

function print_sample_result(config::Dict, sample_num::Int, sample_result::Dict)
    # Build the whole block in an IOBuffer, then write in a single atomic call.
    # Multiple println calls on stdout can get chunked across libuv writes, so
    # assemble the full output first and then emit it all at once.
    io = IOBuffer()
    println(io, "\n  [$(config["name"]) sample $sample_num/$(config["num_samples"])]")
    println(io, @sprintf("    Initial loss: %.6e, accuracy: %.2f%%",
        sample_result["initial_loss"], sample_result["initial_accuracy"] * 100))
    for opt_name in OPTIMIZER_ORDER
        if haskey(sample_result["optimizers"], opt_name)
            r = sample_result["optimizers"][opt_name]
            if !haskey(r, "error")
                acc_imp = r["accuracy_improvement"] * 100
                acc_imp_str = acc_imp >= 0 ? "+$(@sprintf("%.2f", acc_imp))" : @sprintf("%.2f", acc_imp)
                println(io, Printf.format(
                    Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e, acc: %.2f%%  (loss: %.1f%%, acc: %s%%, %.2fs)"),
                    opt_name, r["final_loss"],
                    r["final_accuracy"] * 100,
                    r["improvement_ratio"] * 100,
                    acc_imp_str,
                    r["time"]))
            end
        end
    end
    write(stdout, take!(io))
    flush(stdout)
end

# ============================================================================
# Main execution
# ============================================================================

Random.seed!(42)

println("="^70)
println("Function Learning Experiments")
println("Start time: $(Dates.now())")
println("Epochs: $(args.n_epochs), Quick: $(args.quick_mode)")
println("="^70)

# Announce each experiment up front
for (i, config) in enumerate(configs)
    println("\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs)): $(config["name"])")
    println("#"^70)
    println("  Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"]), Target: $(config["target_fn"])")
    println("  Samples: $(config["num_samples"]), Threshold: $(config["threshold"]), Scale: $(config["scale"])")
end

# Per-config result holders (preserves config order in output)
results_by_config = [Dict{String, Any}("config" => config, "samples" => Any[])
                     for config in configs]

# Flatten (config_idx, sample) into a single task list.
tasks = [(ci, s) for ci in 1:length(configs) for s in 1:configs[ci]["num_samples"]]

println("\nRunning $(length(tasks)) (config, sample) tasks serially...\n")

for (ci, sample) in tasks
    config = configs[ci]
    sample_result = try
        run_single_sample(config, sample)
    catch e
        println("    ✗ $(config["name"]) sample $sample failed: $e")
        flush(stdout)
        Dict{String, Any}("sample_num" => sample, "error" => string(e))
    end

    push!(results_by_config[ci]["samples"], sample_result)

    if !haskey(sample_result, "error")
        print_sample_result(config, sample, sample_result)
    end
end

all_results = results_by_config

# ============================================================================
# Save results
# ============================================================================

if args.save_results
    metadata = build_metadata(
        experiment_type="function_learning",
        n_epochs=args.n_epochs,
        quick_mode=args.quick_mode,
        optimizer_order=OPTIMIZER_ORDER,
        description=args.description,
        git_commit=args.git_commit,
    )

    output_fn = if !isnothing(args.output_filename)
        args.output_filename
    else
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        "function_learning_results_$(timestamp)_raw.json"
    end
    filepath = joinpath(@__DIR__, output_fn)

    save_raw_results(all_results, metadata, filepath)
    save_to_logs(filepath)
end

println("\n✓ All experiments complete!")
