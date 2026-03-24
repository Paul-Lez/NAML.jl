"""
Exploration Constant Sweep

Sweeps the UCT/MCTS exploration constant c from 1.4 to 2.5 in steps of 0.1
across all four paper experiment suites (polynomial_learning,
absolute_sum_minimization, function_learning, polynomial_solving).

Results are saved as a JSON file with:
  - "records"    : flat per-sample per-optimizer rows (good for plotting)
  - "aggregates" : pre-computed mean/std per (c, suite, config, optimizer)
  - "metadata"   : sweep parameters

Usage:
    julia --project=. experiments/paper/sweep_exploration_constant.jl [flags]

Flags:
    --quick          Fast smoke-test: 3 c values, 1 config/suite, 2 samples, 5 epochs
    --epochs N       Optimization epochs per run (default: 20)
    --samples N      Samples per config (overrides per-config default)
    --degree N       Branching degree for MCTS-based optimisers (default: 1)
    --output FILE    Output JSON filename (default: sweep_results_<timestamp>.json)
"""

include(joinpath(@__DIR__, "../../src/NAML.jl"))
include(joinpath(@__DIR__, "util.jl"))
include(joinpath(@__DIR__, "absolute_sum_minimization/util.jl"))
include(joinpath(@__DIR__, "polynomial_solving/util.jl"))

using Oscar
using .NAML
using Printf
using Dates
using Random

# ============================================================================
# Argument parsing
# ============================================================================

quick_mode = "--quick" in ARGS

global n_epochs = quick_mode ? 5 : 20
for (i, arg) in enumerate(ARGS)
    if arg == "--epochs" && i < length(ARGS)
        global n_epochs = parse(Int, ARGS[i+1])
    end
end

global n_samples_override = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--samples" && i < length(ARGS)
        global n_samples_override = parse(Int, ARGS[i+1])
    end
end

global mcts_degree = 1
for (i, arg) in enumerate(ARGS)
    if arg == "--degree" && i < length(ARGS)
        global mcts_degree = parse(Int, ARGS[i+1])
    elseif startswith(arg, "--degree=")
        global mcts_degree = parse(Int, arg[length("--degree=")+1:end])
    end
end

global output_filename = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--output" && i < length(ARGS)
        global output_filename = ARGS[i+1]
    end
end

const C_VALUES_FULL  = round.(collect(1.4:0.1:2.5), digits=1)
const C_VALUES_QUICK = [1.4, 1.8, 2.5]
const C_VALUES       = quick_mode ? C_VALUES_QUICK : C_VALUES_FULL

const SUITES = ["polynomial_learning", "absolute_sum_minimization",
                "function_learning", "polynomial_solving"]

println("=" ^ 70)
println("Exploration Constant Sweep")
println("=" ^ 70)
println("  c values  : $(C_VALUES)")
println("  Epochs    : $n_epochs")
println("  Degree    : $mcts_degree")
println("  Quick mode: $quick_mode")
println()

# ============================================================================
# Load paper configs for each suite
# ============================================================================

# Each paper_config.jl sets the module-level variable `experiment_configs`.
# Capture each before the next include overwrites it.

include(joinpath(@__DIR__, "polynomial_learning/paper_config.jl"))
poly_learning_configs = deepcopy(experiment_configs)

include(joinpath(@__DIR__, "absolute_sum_minimization/paper_config.jl"))
abssum_configs = deepcopy(experiment_configs)

include(joinpath(@__DIR__, "function_learning/paper_config.jl"))
func_learning_configs = deepcopy(experiment_configs)

include(joinpath(@__DIR__, "polynomial_solving/paper_config.jl"))
poly_solving_configs = deepcopy(experiment_configs)

# Apply --samples override or quick trimming
function trim_configs(configs, quick, samples_override)
    cs = quick ? configs[1:1] : configs
    for c in cs
        if !isnothing(samples_override)
            c["num_samples"] = samples_override
        elseif quick
            c["num_samples"] = 2
        end
    end
    return cs
end

poly_learning_configs  = trim_configs(poly_learning_configs,  quick_mode, n_samples_override)
abssum_configs         = trim_configs(abssum_configs,         quick_mode, n_samples_override)
func_learning_configs  = trim_configs(func_learning_configs,  quick_mode, n_samples_override)
poly_solving_configs   = trim_configs(poly_solving_configs,   quick_mode, n_samples_override)

suite_configs = Dict(
    "polynomial_learning"        => poly_learning_configs,
    "absolute_sum_minimization"  => abssum_configs,
    "function_learning"          => func_learning_configs,
    "polynomial_solving"         => poly_solving_configs,
)

# ============================================================================
# Stats helpers
# ============================================================================

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)
_std(v)  = length(v) < 2 ? 0.0 :
    sqrt(sum((x - _mean(v))^2 for x in v) / (length(v) - 1))

# ============================================================================
# Optimizer factory with parameterized exploration constant
# ============================================================================

const OPTIMIZER_ORDER = ["Random", "Best-First", "Best-First-branch2",
                         "MCTS-50", "MCTS-100", "MCTS-200",
                         "DAG-MCTS-50", "DAG-MCTS-100", "DAG-MCTS-200",
                         "DOO", "Best-First-Gradient"]

# Tree-search optimizers that actually use the exploration constant
const USES_EXPLORATION_CONSTANT = Set(["MCTS-50", "MCTS-100", "MCTS-200",
                                        "DAG-MCTS-50", "DAG-MCTS-100", "DAG-MCTS-200"])

function get_sweep_optimizer_configs(c::Float64; quick::Bool=false, degree::Int=1)
    sim_50  = quick ? 10 : 50
    sim_100 = quick ? 20 : 100
    sim_200 = quick ? 40 : 200
    doo_depth = quick ? 10 : 15

    return Dict(
        "Random" => Dict(
            "params" => Dict("degree" => 1),
            "init"   => (param, loss) -> NAML.random_descent_init(param, loss, 1, (false, 1)),
        ),
        "Best-First" => Dict(
            "params" => Dict("strict" => false, "degree" => 1),
            "init"   => (param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 1)),
        ),
        "Best-First-branch2" => Dict(
            "params" => Dict("strict" => false, "degree" => 2),
            "init"   => (param, loss) -> NAML.greedy_descent_init(param, loss, 1, (false, 2)),
        ),
        "MCTS-50" => Dict(
            "params" => Dict("num_simulations" => sim_50, "exploration_constant" => c),
            "init"   => (param, loss) -> NAML.mcts_descent_init(param, loss,
                NAML.MCTSConfig(num_simulations=sim_50, exploration_constant=c,
                                selection_mode=NAML.BestValue, degree=degree)),
        ),
        "MCTS-100" => Dict(
            "params" => Dict("num_simulations" => sim_100, "exploration_constant" => c),
            "init"   => (param, loss) -> NAML.mcts_descent_init(param, loss,
                NAML.MCTSConfig(num_simulations=sim_100, exploration_constant=c,
                                selection_mode=NAML.BestValue, degree=degree)),
        ),
        "MCTS-200" => Dict(
            "params" => Dict("num_simulations" => sim_200, "exploration_constant" => c),
            "init"   => (param, loss) -> NAML.mcts_descent_init(param, loss,
                NAML.MCTSConfig(num_simulations=sim_200, exploration_constant=c,
                                selection_mode=NAML.BestValue, degree=degree)),
        ),
        "DAG-MCTS-50" => Dict(
            "params" => Dict("num_simulations" => sim_50, "exploration_constant" => c,
                             "persist_table" => true),
            "init"   => (param, loss) -> NAML.dag_mcts_descent_init(param, loss,
                NAML.DAGMCTSConfig(num_simulations=sim_50, exploration_constant=c,
                                   degree=degree, persist_table=true,
                                   selection_mode=NAML.BestValue)),
        ),
        "DAG-MCTS-100" => Dict(
            "params" => Dict("num_simulations" => sim_100, "exploration_constant" => c,
                             "persist_table" => true),
            "init"   => (param, loss) -> NAML.dag_mcts_descent_init(param, loss,
                NAML.DAGMCTSConfig(num_simulations=sim_100, exploration_constant=c,
                                   degree=degree, persist_table=true,
                                   selection_mode=NAML.BestValue)),
        ),
        "DAG-MCTS-200" => Dict(
            "params" => Dict("num_simulations" => sim_200, "exploration_constant" => c,
                             "persist_table" => true),
            "init"   => (param, loss) -> NAML.dag_mcts_descent_init(param, loss,
                NAML.DAGMCTSConfig(num_simulations=sim_200, exploration_constant=c,
                                   degree=degree, persist_table=true,
                                   selection_mode=NAML.BestValue)),
        ),
        "DOO" => Dict(
            "params" => Dict("max_depth" => doo_depth),
            "init"   => (param, loss) -> begin
                p = Float64(NAML.prime(param))
                NAML.doo_descent_init(param, loss, 1,
                    NAML.DOOConfig(delta=h->p^(-h), max_depth=doo_depth,
                                   degree=degree, strict=false))
            end,
        ),
        "Best-First-Gradient" => Dict(
            "params" => Dict("degree" => 1),
            "init"   => (param, loss) -> NAML.gradient_descent_init(param, loss, 1, (false, 1)),
        ),
    )
end

# ============================================================================
# Common optimizer run loop
# ============================================================================

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
        avg_rank = (i + j - 1) / 2.0
        for k in i:j-1
            optimizers[valid_opts[k][1]]["rank"] = avg_rank
        end
        i = j
    end
end

"""
Run all optimizers from `opt_configs` starting from `initial_param`.
`post_process` is an optional function `(opt_name, optim, result_dict) -> nothing`
for suite-specific additions (e.g. accuracy).
"""
function run_all_optimizers(initial_param, initial_loss, loss, opt_configs,
                             n_epochs_local; post_process=nothing)
    sample_results = Dict{String,Any}("optimizers" => Dict{String,Any}())

    for opt_name in OPTIMIZER_ORDER
        !haskey(opt_configs, opt_name) && continue
        opt_setup = opt_configs[opt_name]
        try
            counted_loss, eval_counter = wrap_loss_with_counting(loss)
            optim = opt_setup["init"](initial_param, counted_loss)

            losses_arr = Float64[]
            t_start = time()
            for _ in 1:n_epochs_local
                push!(losses_arr, NAML.eval_loss(optim))
                NAML.step!(optim)
                NAML.has_converged(optim) && break
            end
            elapsed = time() - t_start

            final_loss = NAML.eval_loss(optim)
            push!(losses_arr, final_loss)
            monitoring_evals = length(losses_arr)
            total_evals = eval_counter.eval_count - monitoring_evals + eval_counter.grad_count

            opt_result = Dict{String,Any}(
                "final_loss"        => final_loss,
                "time"              => elapsed,
                "improvement"       => initial_loss - final_loss,
                "improvement_ratio" => initial_loss > 0 ?
                                       (initial_loss - final_loss) / initial_loss : 0.0,
                "total_evals"       => total_evals,
            )

            !isnothing(post_process) && post_process(opt_name, optim, opt_result)

            sample_results["optimizers"][opt_name] = opt_result
        catch e
            println("    ✗ Error in $opt_name: $e")
            sample_results["optimizers"][opt_name] = Dict("error" => string(e))
        end
    end

    compute_sample_rankings!(sample_results)
    return sample_results
end

# ============================================================================
# Function-learning helpers (from function_learning/run_experiments.jl)
# ============================================================================

function _create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)
    p    = Int(Oscar.prime(K))
    prec = Oscar.precision(K)
    xs   = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]
    ys   = target_fn == "zero"   ? [0.0 for _ in xs] :
           target_fn == "one"    ? [1.0 for _ in xs] :
           target_fn == "random" ? [Float64(rand(0:1)) for _ in xs] :
           error("Unknown target_fn: $target_fn")
    data = collect(zip(xs, ys))
    loss = polynomial_to_crossentropy_loss(data, degree, threshold, scale)
    return loss, data
end

function _compute_fl_accuracy(param, data, threshold, scale)
    coeffs = [NAML.unwrap(c) for c in NAML.center(param)]
    function eval_poly(x)
        r = coeffs[1]
        xp = x
        for i in 2:length(coeffs)
            r += coeffs[i] * xp; xp *= x
        end
        return r
    end
    correct = 0
    for (x, y) in data
        val = Float64(abs(eval_poly(x)))
        prob = 1.0 / (1.0 + exp(-((val - threshold) / scale)))
        prediction = prob > 0.5 ? 1.0 : 0.0
        prediction == y && (correct += 1)
    end
    return correct / length(data)
end

# ============================================================================
# Per-suite sample runners
# ============================================================================

function run_sample_polynomial_learning(config::Dict, sample_num::Int, c::Float64)
    p = config["prime"]; prec = config["prec"]; degree = config["degree"]
    n_points = config["n_points"]
    K = PadicField(p, prec)

    # Distinct random x values
    x_values = PadicFieldElem[]
    attempts = 0
    while length(x_values) < n_points && attempts < n_points * 100
        x = generate_random_padic(p, prec, 0, 8)
        any(e -> e == x, x_values) || push!(x_values, x)
        attempts += 1
    end
    length(x_values) < n_points && error("Could not generate $n_points distinct points")

    y_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]
    data = collect(zip(x_values, y_values))

    loss = polynomial_to_linear_loss(data, degree, nothing)
    initial_param = generate_gauss_point(degree + 1, K)
    initial_loss = loss.eval([initial_param])[1]

    opt_configs = get_sweep_optimizer_configs(c; quick=quick_mode, degree=mcts_degree)
    result = run_all_optimizers(initial_param, initial_loss, loss, opt_configs, n_epochs)
    result["initial_loss"] = initial_loss
    result["sample_num"] = sample_num
    return result
end

function run_sample_abssum(config::Dict, sample_num::Int, c::Float64)
    p = config["prime"]; prec = config["prec"]
    num_polys = config["num_polys"]; num_vars = config["num_vars"]
    degree = config["degree"]
    K = PadicField(p, prec)

    loss = generate_random_absolute_sum_problem(p, prec, num_polys, num_vars, degree)
    initial_param = generate_initial_point(num_vars, K)
    initial_loss = loss.eval([initial_param])[1]

    opt_configs = get_sweep_optimizer_configs(c; quick=quick_mode, degree=mcts_degree)
    result = run_all_optimizers(initial_param, initial_loss, loss, opt_configs, n_epochs)
    result["initial_loss"] = initial_loss
    result["sample_num"] = sample_num
    return result
end

function run_sample_function_learning(config::Dict, sample_num::Int, c::Float64)
    p = config["prime"]; prec = config["prec"]; degree = config["degree"]
    n_points = config["n_points"]; target_fn = config["target_fn"]
    threshold = config["threshold"]; scale = config["scale"]
    K = PadicField(p, prec)

    loss, data = _create_function_learning_loss(K, degree, n_points, target_fn, threshold, scale)

    param_center = [K(0) for _ in 1:degree+1]
    initial_param = NAML.ValuationPolydisc(param_center, [0 for _ in 1:degree+1])
    initial_loss = loss.eval([initial_param])[1]
    initial_accuracy = _compute_fl_accuracy(initial_param, data, threshold, scale)

    # Post-processor: attach final_accuracy to each optimizer result
    function add_accuracy(opt_name, optim, opt_result)
        opt_result["final_accuracy"] = _compute_fl_accuracy(optim.param, data, threshold, scale)
        opt_result["accuracy_improvement"] = opt_result["final_accuracy"] - initial_accuracy
    end

    opt_configs = get_sweep_optimizer_configs(c; quick=quick_mode, degree=mcts_degree)
    result = run_all_optimizers(initial_param, initial_loss, loss, opt_configs, n_epochs;
                                post_process=add_accuracy)
    result["initial_loss"] = initial_loss
    result["initial_accuracy"] = initial_accuracy
    result["sample_num"] = sample_num
    return result
end

function run_sample_polynomial_solving(config::Dict, sample_num::Int, c::Float64)
    p = config["prime"]; prec = config["prec"]
    num_vars = config["num_vars"]; degree = config["degree"]
    opt_degree = config["opt_degree"]
    K = PadicField(p, prec)

    loss, _root = generate_polynomial_solving_problem(p, prec, num_vars, degree)
    initial_param = generate_initial_point(num_vars, K)
    initial_loss = loss.eval([initial_param])[1]

    effective_degree = mcts_degree != 1 ? mcts_degree : opt_degree
    opt_configs = get_sweep_optimizer_configs(c; quick=quick_mode, degree=effective_degree)
    result = run_all_optimizers(initial_param, initial_loss, loss, opt_configs, n_epochs)
    result["initial_loss"] = initial_loss
    result["sample_num"] = sample_num
    return result
end

const SUITE_RUNNERS = Dict(
    "polynomial_learning"       => run_sample_polynomial_learning,
    "absolute_sum_minimization" => run_sample_abssum,
    "function_learning"         => run_sample_function_learning,
    "polynomial_solving"        => run_sample_polynomial_solving,
)

# ============================================================================
# Aggregate statistics over samples
# ============================================================================

function compute_aggregates(samples::Vector, optimizer_order)
    agg = Dict{String,Any}()
    for opt_name in optimizer_order
        opt_data = [s["optimizers"][opt_name] for s in samples
                    if haskey(s, "optimizers") && haskey(s["optimizers"], opt_name) &&
                       !haskey(s["optimizers"][opt_name], "error")]
        isempty(opt_data) && continue

        losses  = [d["final_loss"] for d in opt_data]
        times   = [d["time"] for d in opt_data]
        improvs = [d["improvement_ratio"] for d in opt_data]
        ranks   = [d["rank"] for d in opt_data if haskey(d, "rank")]

        entry = Dict{String,Any}(
            "mean_final_loss"        => _mean(losses),
            "std_final_loss"         => _std(losses),
            "mean_time"              => _mean(times),
            "std_time"               => _std(times),
            "mean_improvement_ratio" => _mean(improvs),
            "n_samples"              => length(opt_data),
        )
        if !isempty(ranks)
            entry["mean_rank"] = _mean(ranks)
            entry["std_rank"]  = _std(ranks)
        end
        # function_learning accuracy
        accs = [d["final_accuracy"] for d in opt_data if haskey(d, "final_accuracy")]
        if !isempty(accs)
            entry["mean_final_accuracy"] = _mean(accs)
            entry["std_final_accuracy"]  = _std(accs)
        end

        agg[opt_name] = entry
    end
    return agg
end

# ============================================================================
# Main sweep
# ============================================================================

Random.seed!(42)

flat_records  = Dict{String,Any}[]  # one row per (c, suite, config, sample, optimizer)
agg_records   = Dict{String,Any}[]  # one row per (c, suite, config, optimizer)

total_configs = sum(length(v) for v in values(suite_configs))
total_iters   = length(C_VALUES) * total_configs
n_done = 0

t_sweep_start = time()

for c in C_VALUES
    println("\n" * "=" ^ 70)
    println(@sprintf("c = %.1f", c))
    println("=" ^ 70)

    for suite in SUITES
        configs  = suite_configs[suite]
        runner   = SUITE_RUNNERS[suite]

        for config in configs
            n_samples = config["num_samples"]
            println(@sprintf("  [%s] %s  (%d samples)", suite, config["name"], n_samples))

            samples_collected = Dict{String,Any}[]

            for s in 1:n_samples
                try
                    sample_result = runner(config, s, c)
                    push!(samples_collected, sample_result)

                    # Emit flat records
                    for opt_name in OPTIMIZER_ORDER
                        !haskey(sample_result["optimizers"], opt_name) && continue
                        opt_res = sample_result["optimizers"][opt_name]
                        haskey(opt_res, "error") && continue

                        rec = Dict{String,Any}(
                            "c"                  => c,
                            "suite"              => suite,
                            "config_name"        => config["name"],
                            "sample_num"         => s,
                            "optimizer"          => opt_name,
                            "uses_exploration_c" => opt_name in USES_EXPLORATION_CONSTANT,
                            "final_loss"         => opt_res["final_loss"],
                            "improvement_ratio"  => opt_res["improvement_ratio"],
                            "rank"               => get(opt_res, "rank", nothing),
                            "time"               => opt_res["time"],
                        )
                        if haskey(opt_res, "final_accuracy")
                            rec["final_accuracy"]       = opt_res["final_accuracy"]
                            rec["accuracy_improvement"] = opt_res["accuracy_improvement"]
                        end
                        push!(flat_records, rec)
                    end
                catch e
                    println("    ✗ Sample $s failed: $e")
                end
            end

            # Aggregate over collected samples
            if !isempty(samples_collected)
                agg = compute_aggregates(samples_collected, OPTIMIZER_ORDER)
                for opt_name in OPTIMIZER_ORDER
                    !haskey(agg, opt_name) && continue
                    a = agg[opt_name]
                    arec = Dict{String,Any}(
                        "c"                      => c,
                        "suite"                  => suite,
                        "config_name"            => config["name"],
                        "optimizer"              => opt_name,
                        "uses_exploration_c"     => opt_name in USES_EXPLORATION_CONSTANT,
                        "mean_final_loss"        => a["mean_final_loss"],
                        "std_final_loss"         => a["std_final_loss"],
                        "mean_time"              => a["mean_time"],
                        "std_time"               => a["std_time"],
                        "mean_improvement_ratio" => a["mean_improvement_ratio"],
                        "n_samples"              => a["n_samples"],
                    )
                    if haskey(a, "mean_rank")
                        arec["mean_rank"] = a["mean_rank"]
                        arec["std_rank"]  = a["std_rank"]
                    end
                    if haskey(a, "mean_final_accuracy")
                        arec["mean_final_accuracy"] = a["mean_final_accuracy"]
                        arec["std_final_accuracy"]  = a["std_final_accuracy"]
                    end
                    push!(agg_records, arec)
                end
            end

            global n_done += 1
            elapsed_so_far = time() - t_sweep_start
            pct = round(100 * n_done / total_iters, digits=1)
            println(@sprintf("    → done  [%d/%d configs, %.1f%%,  %.0fs elapsed]",
                             n_done, total_iters, pct, elapsed_so_far))
        end
    end
end

t_sweep_end = time()
println("\n✓ Sweep complete in $(round(t_sweep_end - t_sweep_start, digits=1))s")
println("  Records    : $(length(flat_records))")
println("  Aggregates : $(length(agg_records))")

# ============================================================================
# Save to JSON
# ============================================================================

try
    using JSON

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    filepath  = isnothing(output_filename) ?
        joinpath(@__DIR__, "sweep_results_$(timestamp).json") :
        output_filename

    output = Dict{String,Any}(
        "metadata" => Dict{String,Any}(
            "c_values"       => C_VALUES,
            "suites"         => SUITES,
            "optimizer_order"=> OPTIMIZER_ORDER,
            "n_epochs"       => n_epochs,
            "mcts_degree"    => mcts_degree,
            "quick_mode"     => quick_mode,
            "timestamp"      => string(Dates.now()),
            "n_records"      => length(flat_records),
            "n_aggregates"   => length(agg_records),
        ),
        "records"    => flat_records,
        "aggregates" => agg_records,
    )

    open(filepath, "w") do f
        JSON.print(f, output, 2)
    end
    println("✓ Results saved to: $filepath")
catch e
    println("✗ Could not save JSON: $e")
end
