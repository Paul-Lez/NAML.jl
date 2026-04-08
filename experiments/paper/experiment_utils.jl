"""
Shared experiment utilities for the NAML paper experiment infrastructure.

Provides:
1. Unified CLI argument parsing
2. Canonical optimizer factory (all optimizers in one place)
3. Threaded experiment runner (parallelizes across optimizers within a sample)
4. Raw JSON serialization (no statistics — that's make_stats.jl's job)

Usage from any run_experiments.jl:
    include("../experiment_utils.jl")
    args = parse_experiment_args(ARGS)
    ...
"""

using JSON
using Printf
using Dates
using Random

# ============================================================================
# CLI Argument Parsing
# ============================================================================

"""
    parse_experiment_args(ARGS) -> NamedTuple

Unified CLI argument parser for all run_experiments.jl scripts.

Returns a NamedTuple with fields:
- quick_mode::Bool
- save_results::Bool
- use_config_file::Bool
- use_paper_config::Bool
- n_epochs::Int
- output_filename::Union{String, Nothing}
- n_samples_override::Union{Int, Nothing}
- selection_mode  (NAML.BestValue, NAML.VisitCount, or NAML.BestLoss)
- mcts_degree_override::Union{Int, Nothing}
- description::String
- git_commit::String
"""
function parse_experiment_args(args)
    quick_mode = "--quick" in args
    save_results = "--save" in args
    use_config_file = "--config" in args
    use_paper_config = "--paper" in args

    # Default epochs
    n_epochs = quick_mode ? 5 : 20

    output_filename = nothing
    n_samples_override = nothing
    selection_mode = NAML.BestValue
    mcts_degree_override = nothing
    description = ""
    git_commit = ""

    for (i, arg) in enumerate(args)
        if arg == "--epochs" && i < length(args)
            n_epochs = parse(Int, args[i+1])
        elseif arg == "--output" && i < length(args)
            output_filename = args[i+1]
        elseif arg == "--samples" && i < length(args)
            n_samples_override = parse(Int, args[i+1])
        elseif arg == "--selection-mode" && i < length(args)
            mode_str = args[i+1]
            if mode_str == "BestValue"
                selection_mode = NAML.BestValue
            elseif mode_str == "VisitCount"
                selection_mode = NAML.VisitCount
            elseif mode_str == "BestLoss"
                selection_mode = NAML.BestLoss
            else
                error("Invalid selection mode: $mode_str. Must be BestValue, VisitCount, or BestLoss")
            end
        elseif arg == "--degree" && i < length(args)
            mcts_degree_override = parse(Int, args[i+1])
        elseif startswith(arg, "--degree=")
            mcts_degree_override = parse(Int, arg[10:end])
        elseif arg == "--description" && i < length(args)
            description = args[i+1]
        elseif arg == "--git-commit" && i < length(args)
            git_commit = args[i+1]
        end
    end

    return (
        quick_mode = quick_mode,
        save_results = save_results,
        use_config_file = use_config_file,
        use_paper_config = use_paper_config,
        n_epochs = n_epochs,
        output_filename = output_filename,
        n_samples_override = n_samples_override,
        selection_mode = selection_mode,
        mcts_degree_override = mcts_degree_override,
        description = description,
        git_commit = git_commit,
    )
end


# ============================================================================
# Load configurations
# ============================================================================

function load_config_file(experiment_dir::String, args)
    if args.use_paper_config
        include(joinpath(experiment_dir, "paper_config.jl"))
    elseif args.use_config_file
        include(joinpath(experiment_dir, "config.jl"))
    end
end

"""
    load_configs(experiment_dir, args, default_configs) -> Vector{Dict}

Load experiment configurations based on CLI flags.
"""
function load_configs(args, default_configs::Vector)
    configs = if args.use_paper_config
        println("Loaded PAPER-READY experiment configurations")
        paper_experiments
    elseif args.use_config_file
        println("Loaded experiment configurations from config.jl")
        experiment_configs
    else
        println("Using default configurations")
        default_configs
    end

    # Apply samples override
    if !isnothing(args.n_samples_override)
        for config in configs
            config["num_samples"] = args.n_samples_override
        end
        println("Overriding num_samples to $(args.n_samples_override) for all configs")
    end

    return configs
end


# ============================================================================
# Optimizer Factory
# ============================================================================

"""Canonical display ordering for all experiments."""
const OPTIMIZER_ORDER = [
    "Random", "Best-First", "Best-First-branch2", "Best-First-Gradient",
    "MCTS-k", "MCTS-5k", "MCTS-10k",
    "DAG-MCTS-k", "DAG-MCTS-5k", "DAG-MCTS-10k",
    "DOO"
]

const NAME_WIDTH = maximum(length(n) for n in OPTIMIZER_ORDER)

"""
    get_optimizer_configs(; quick, selection_mode, degree, prime, dim) -> Dict

Return a Dict of optimizer name => Dict("init" => (param, loss) -> OptimSetup).

All experiments use the same set of optimizers. Prime is used only to compute
k (number of polydisc children) and DOO's delta function.
"""
function get_optimizer_configs(; quick::Bool=false,
                                 selection_mode=NAML.BestValue,
                                 degree::Int=1,
                                 prime::Int=2,
                                 dim::Int=1)
    # k = number of children of a polydisc
    k = binomial(dim, degree) * prime^degree
    sims_k   = quick ? 50 : k
    sims_5k  = quick ? 100 : 5 * k
    sims_10k = quick ? 200 : 10 * k

    p_float = Float64(prime)

    return Dict(
        "Random" => Dict(
            "init" => (param, loss) -> begin
                NAML.random_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First" => Dict(
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "Best-First-branch2" => Dict(
            "init" => (param, loss) -> begin
                NAML.greedy_descent_init(param, loss, 1, (false, 2))
            end
        ),
        "Best-First-Gradient" => Dict(
            "init" => (param, loss) -> begin
                NAML.gradient_descent_init(param, loss, 1, (false, 1))
            end
        ),
        "MCTS-k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-5k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_5k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "MCTS-10k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.MCTSConfig(
                    num_simulations=sims_10k,
                    exploration_constant=1.41,
                    selection_mode=selection_mode,
                    degree=degree
                )
                NAML.mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-5k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_5k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DAG-MCTS-10k" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.DAGMCTSConfig(
                    num_simulations=sims_10k,
                    exploration_constant=1.41,
                    degree=degree,
                    persist_table=true,
                    selection_mode=NAML.BestValue
                )
                NAML.dag_mcts_descent_init(param, loss, config)
            end
        ),
        "DOO" => Dict(
            "init" => (param, loss) -> begin
                delta = h -> p_float^(-h)
                config = NAML.DOOConfig(
                    delta=delta,
                    max_depth=quick ? 10 : 15,
                    degree=degree,
                    strict=false
                )
                NAML.doo_descent_init(param, loss, 1, config)
            end
        ),
    )
end

"""
    effective_degree(num_dims, mcts_degree_override) -> Int

Compute the effective MCTS/tree branching degree.
Default: 1 for 1-dimensional, 2 for ≥2 dimensions.
"""
function effective_degree(num_dims::Int, mcts_degree_override)
    auto_degree = num_dims >= 2 ? 2 : 1
    return isnothing(mcts_degree_override) ? auto_degree : mcts_degree_override
end


# ============================================================================
# Single optimizer run
# ============================================================================

"""
    run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                         post_run_fn=nothing) -> Dict

Run a single optimizer on a single problem instance.
Returns a Dict with raw results (no ranking or aggregate stats).

Deep-copies `initial_param` to avoid mutation issues when running in parallel.

If `post_run_fn` is provided, it is called as `post_run_fn(optim)` and
the returned Dict is merged into the result. Use this for experiment-specific
fields like classification accuracy.
"""
function run_single_optimizer(opt_name::String, opt_setup::Dict,
                               initial_param, loss, n_epochs::Int;
                               post_run_fn::Union{Function,Nothing}=nothing)
    # Deep copy starting parameter to avoid cross-thread mutation
    param_copy = deepcopy(initial_param)
    initial_loss_val = loss.eval([param_copy])[1]

    try
        # Wrap loss with evaluation counting
        counted_loss, eval_counter = wrap_loss_with_counting(loss)

        optim = opt_setup["init"](param_copy, counted_loss)

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

        # Subtract monitoring eval_loss calls
        monitoring_evals = length(losses)
        total_optimizer_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

        result = Dict{String, Any}(
            "time" => elapsed,
            "final_loss" => final_loss,
            "losses" => losses,
            "improvement" => initial_loss_val - final_loss,
            "improvement_ratio" => (initial_loss_val > 0) ?
                (initial_loss_val - final_loss) / initial_loss_val : 0.0,
            "total_evals" => total_optimizer_evals,
        )

        # Run experiment-specific post-processing (e.g., accuracy computation)
        if !isnothing(post_run_fn)
            extra = post_run_fn(optim)
            merge!(result, extra)
        end

        return result
    catch e
        return Dict{String, Any}("error" => string(e))
    end
end


# ============================================================================
# Threaded run across all optimizers for one sample
# ============================================================================

"""
    run_all_optimizers_serial(opt_configs, initial_param, loss, n_epochs;
                              post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, serially (no threading).

Use this when the caller is already parallelizing at a coarser level (e.g. over
`(config, sample)` pairs) and each task owns its own `loss`. Running optimizers
serially here avoids nested threading and contention on shared evaluator state.
"""
function run_all_optimizers_serial(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                    post_run_fn::Union{Function,Nothing}=nothing)
    results = Dict{String, Any}()
    for opt_name in keys(opt_configs)
        opt_setup = opt_configs[opt_name]
        results[opt_name] = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                                  post_run_fn=post_run_fn)
    end
    return results
end


"""
    run_all_optimizers_threaded(opt_configs, initial_param, loss, n_epochs;
                                post_run_fn=nothing) -> Dict{String,Any}

Run all optimizers on a single problem instance, using threads.
Returns a Dict mapping optimizer name => result Dict.
"""
function run_all_optimizers_threaded(opt_configs::Dict, initial_param, loss, n_epochs::Int;
                                     post_run_fn::Union{Function,Nothing}=nothing)
    opt_names = collect(keys(opt_configs))

    results = Dict{String, Any}()
    result_lock = ReentrantLock()

    Threads.@threads for i in 1:length(opt_names)
        opt_name = opt_names[i]
        opt_setup = opt_configs[opt_name]
        result = run_single_optimizer(opt_name, opt_setup, initial_param, loss, n_epochs;
                                       post_run_fn=post_run_fn)
        lock(result_lock) do
            results[opt_name] = result
        end
    end

    return results
end


# ============================================================================
# JSON serialization
# ============================================================================

"""
    build_metadata(; experiment_type, n_epochs, quick_mode, optimizer_order,
                     description, git_commit, extra...) -> Dict

Build metadata dict for JSON output.
"""
function build_metadata(; experiment_type::String,
                          n_epochs::Int,
                          quick_mode::Bool,
                          optimizer_order::Vector{String},
                          description::String="",
                          git_commit::String="",
                          extra::Dict{String,Any}=Dict{String,Any}())
    metadata = Dict{String, Any}(
        "experiment_type" => experiment_type,
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        "n_epochs" => n_epochs,
        "quick_mode" => quick_mode,
        "optimizer_order" => optimizer_order,
        "description" => description,
        "git_commit" => git_commit,
    )
    merge!(metadata, extra)
    return metadata
end

"""
    save_raw_results(all_results, metadata, filepath)

Save raw experiment results to JSON. No aggregate stats — just raw per-sample data.
"""
function save_raw_results(all_results::Vector, metadata::Dict, filepath::String)
    json_experiments = []
    for result in all_results
        json_result = Dict{String, Any}()
        json_result["config"] = result["config"]

        if haskey(result, "error")
            json_result["error"] = result["error"]
        else
            json_result["samples"] = result["samples"]
        end

        push!(json_experiments, json_result)
    end

    json_output = Dict{String, Any}(
        "metadata" => metadata,
        "experiments" => json_experiments,
    )

    open(filepath, "w") do f
        JSON.print(f, json_output, 2)
    end

    println("\n✓ Raw results saved to: $filepath")
end


# ============================================================================
# Progress printing
# ============================================================================

"""Print a brief per-sample summary to stdout."""
function print_sample_summary(sample_result::Dict, initial_loss::Float64)
    println(@sprintf("    Initial: %.6e", initial_loss))
    for opt_name in OPTIMIZER_ORDER
        if haskey(sample_result, opt_name)
            opt_result = sample_result[opt_name]
            if !haskey(opt_result, "error")
                println(Printf.format(Printf.Format("    %-$(NAME_WIDTH)s Final: %.6e (Δ: %.6e, %.1f%%)"),
                    opt_name, opt_result["final_loss"], opt_result["improvement"],
                    opt_result["improvement_ratio"] * 100))
            end
        end
    end
end
