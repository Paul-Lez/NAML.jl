"""
Polynomial Learning Experiment Runner

Learn polynomial coefficients from (x, y) data where y = f(x) for a random
unknown polynomial f. The optimizer searches for coefficients that minimize
the reconstruction error.

Usage:
    julia --project=. experiments/paper/polynomial_learning/run_experiments.jl [FLAGS]

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
    Dict("name" => "p2_deg2_5pts", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 5, "num_samples" => 2),
]

load_config_file(@__DIR__, args)
configs = load_configs(args, default_configs)

# ============================================================================
# Run single sample
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

    # Generate random target polynomial and evaluate
    polynomial = generate_random_polynomial(K, 1, degree, ["x"])
    y_values = [evaluate(polynomial, [x]) for x in x_values]
    data = collect(zip(x_values, y_values))

    # Create loss (p-adic output, no cutoff)
    loss = polynomial_to_linear_loss(data, degree, nothing)

    # Initial parameters at Gauss point
    num_params = degree + 1
    initial_param = generate_gauss_point(num_params, K)
    initial_loss = loss.eval([initial_param])[1]

    # Get optimizer configs
    eff_degree = effective_degree(num_params, args.mcts_degree_override)
    opt_configs = get_optimizer_configs(
        quick=args.quick_mode, selection_mode=args.selection_mode,
        degree=eff_degree, prime=p, dim=num_params
    )

    # Run all optimizers serially inside this task. Parallelism lives at the
    # (config, sample) level above; running optimizers serially here avoids
    # nested threading and contention on the shared loss evaluator.
    optimizer_results = run_all_optimizers_serial(
        opt_configs, initial_param, loss, args.n_epochs
    )

    return Dict{String, Any}(
        "sample_num" => sample_num,
        "initial_loss" => initial_loss,
        "data" => Dict(
            "x_abs_values" => [Float64(abs(x)) for x in x_values],
            "y_abs_values" => [Float64(abs(y)) for y in y_values],
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
    println(io, @sprintf("    Initial: %.6e", sample_result["initial_loss"]))
    for opt_name in OPTIMIZER_ORDER
        if haskey(sample_result["optimizers"], opt_name)
            r = sample_result["optimizers"][opt_name]
            if !haskey(r, "error")
                println(io, Printf.format(
                    Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e (Δ: %.6e, %.1f%%)"),
                    opt_name, r["final_loss"], r["improvement"],
                    r["improvement_ratio"] * 100))
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
println("Polynomial Learning Experiments")
println("Start time: $(Dates.now())")
println("Epochs: $(args.n_epochs), Quick: $(args.quick_mode)")
println("="^70)

# Announce each experiment up front
for (i, config) in enumerate(configs)
    println("\n" * "#"^70)
    println("# EXPERIMENT $i/$(length(configs)): $(config["name"])")
    println("#"^70)
    println("Prime: $(config["prime"]), Degree: $(config["degree"]), " *
            "Points: $(config["n_points"])")
    println("Samples: $(config["num_samples"]), Epochs: $(args.n_epochs)")
end

# Initialize per-config result holders (preserves config order in output)
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
        experiment_type="polynomial_learning",
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
        "poly_learning_results_$(timestamp)_raw.json"
    end
    filepath = joinpath(@__DIR__, output_fn)

    save_raw_results(all_results, metadata, filepath)
    save_to_logs(filepath)
end

println("\n✓ All experiments complete!")
