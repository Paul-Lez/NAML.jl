"""
Polynomial Interpolation with Optimizer Benchmarking

This script generalizes polynomial learning to polynomial interpolation,
comparing multiple optimization algorithms.

Problem: Given data points (x_i, y_i) where x_i, y_i are p-adic numbers,
learn polynomial coefficients (a₀, ..., aₙ) such that:
    f(x_i) = a₀ + a₁x_i + ... + aₙx_iⁿ ≈ y_i

Optimizers benchmarked:
1. Greedy Descent (classical)
2. MCTS (Monte Carlo Tree Search)
3. UCT (Upper Confidence Trees)
4. Modified UCT (variant)
5. Flat UCB (flat variant)
6. HOO (Hierarchical Optimistic Optimization)
"""

include("../../src/NAML.jl")
include("util.jl")

using Oscar
using .NAML
using Printf

println("="^70)
println("Polynomial Interpolation with Optimizer Benchmarking")
println("="^70)

# ============================================================================
# Configuration
# ============================================================================
println("\n[1] Configuration")
println("-"^70)

prec = 20                # p-adic precision
p = 2                    # prime
K = PadicField(p, prec)  # p-adic field

n_points = 5             # number of data points
poly_degree = 4          # polynomial degree (we have degree+1 parameters)
n_epochs = 15            # number of optimization epochs

println("Prime:             $p")
println("Precision:         $prec")
println("Data points:       $n_points")
println("Polynomial degree: $poly_degree")
println("Parameters:        $(poly_degree + 1) coefficients")
println("Epochs:            $n_epochs")

# ============================================================================
# Generate Interpolation Data
# ============================================================================
println("\n[2] Generating interpolation data")
println("-"^70)

# Generate random x values (distinct)
x_values = Vector{PadicFieldElem}()
max_attempts = n_points * 100
attempts = 0

while length(x_values) < n_points && attempts < max_attempts
    x = generate_random_padic(p, prec, 0, 8)  # integers with 8 terms
    if !any(existing_x -> existing_x == x, x_values)
        push!(x_values, x)
    end
    global attempts += 1
end

# Generate random y values (p-adic, not necessarily zero!)
y_values = [generate_random_padic(p, prec, 0, 8) for _ in 1:n_points]

# Create data pairs
data = [(x, y) for (x, y) in zip(x_values, y_values)]

println("Data points (x_i, y_i):")
for (i, (x, y)) in enumerate(data)
    println("  Point $i: x = $x")
    println("            y = $y")
end

# Verify distinctness
all_distinct = length(x_values) == length(unique(x_values))
println("\nAll x values distinct: $all_distinct")

# ============================================================================
# Create Loss Function
# ============================================================================
println("\n[3] Creating loss function")
println("-"^70)

# Use polynomial_to_linear_loss with p-adic outputs (no cutoff needed)
loss = polynomial_to_linear_loss(data, poly_degree, nothing)

println("Loss function created successfully")
println("Loss type: Mean squared error (p-adic)")
println("Parameters: $(poly_degree + 1) coefficients (a₀, ..., a$poly_degree)")

# ============================================================================
# Initialize Starting Point
# ============================================================================
println("\n[4] Initializing starting point")
println("-"^70)

# Start at Gauss point (standard starting point)
initial_param = generate_gauss_point(poly_degree + 1, K)

println("Starting point: Gauss point")
println("Center: $(NAML.center(initial_param))")
println("Radius: $(NAML.radius(initial_param))")

# Compute initial loss
initial_loss = loss.eval([initial_param])[1]
println("\nInitial loss: $initial_loss")

# ============================================================================
# Benchmark All Optimizers
# ============================================================================
println("\n[5] Benchmarking optimizers")
println("="^70)

# Storage for results
results = Dict{String, Any}()

# ----------------------------------------------------------------------------
# 5.1 Greedy Descent
# ----------------------------------------------------------------------------
println("\n[5.1] Greedy Descent")
println("-"^70)

state = 1
config = (true, 0)
optim_greedy = NAML.greedy_descent_init(initial_param, loss, state, config)

greedy_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_greedy)
    push!(greedy_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_greedy)
end
t_end = time()
greedy_time = t_end - t_start

greedy_final_loss = NAML.eval_loss(optim_greedy)
push!(greedy_losses, greedy_final_loss)

results["Greedy"] = Dict(
    "time" => greedy_time,
    "final_loss" => greedy_final_loss,
    "losses" => greedy_losses
)

println(@sprintf("\nGreedy: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                greedy_time, greedy_final_loss, initial_loss - greedy_final_loss))

# ----------------------------------------------------------------------------
# 5.2 MCTS
# ----------------------------------------------------------------------------
println("\n[5.2] MCTS (Monte Carlo Tree Search)")
println("-"^70)

mcts_config = NAML.MCTSConfig(
    num_simulations=100,
    exploration_constant=1.41,
    selection_mode=NAML.BestValue,
    degree=3
)
optim_mcts = NAML.mcts_descent_init(initial_param, loss, mcts_config)

mcts_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_mcts)
    push!(mcts_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_mcts)
end
t_end = time()
mcts_time = t_end - t_start

mcts_final_loss = NAML.eval_loss(optim_mcts)
push!(mcts_losses, mcts_final_loss)

results["MCTS"] = Dict(
    "time" => mcts_time,
    "final_loss" => mcts_final_loss,
    "losses" => mcts_losses
)

println(@sprintf("\nMCTS: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                mcts_time, mcts_final_loss, initial_loss - mcts_final_loss))

# ----------------------------------------------------------------------------
# 5.3 UCT
# ----------------------------------------------------------------------------
println("\n[5.3] UCT (Upper Confidence Trees)")
println("-"^70)

uct_config = NAML.UCTConfig(
    max_depth=10,
    num_simulations=100,
    exploration_constant=1.41,
    degree=3
)
optim_uct = NAML.uct_descent_init(initial_param, loss, uct_config)

uct_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_uct)
    push!(uct_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_uct)
end
t_end = time()
uct_time = t_end - t_start

uct_final_loss = NAML.eval_loss(optim_uct)
push!(uct_losses, uct_final_loss)

results["UCT"] = Dict(
    "time" => uct_time,
    "final_loss" => uct_final_loss,
    "losses" => uct_losses
)

println(@sprintf("\nUCT: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                uct_time, uct_final_loss, initial_loss - uct_final_loss))

# ----------------------------------------------------------------------------
# 5.4 Modified UCT
# ----------------------------------------------------------------------------
println("\n[5.4] Modified UCT")
println("-"^70)

mod_uct_config = NAML.ModifiedUCTConfig(
    max_depth=10,
    num_simulations=100,
    beta=0.05,
    degree=3
)
optim_mod_uct = NAML.modified_uct_descent_init(initial_param, loss, mod_uct_config)

mod_uct_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_mod_uct)
    push!(mod_uct_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_mod_uct)
end
t_end = time()
mod_uct_time = t_end - t_start

mod_uct_final_loss = NAML.eval_loss(optim_mod_uct)
push!(mod_uct_losses, mod_uct_final_loss)

results["Modified UCT"] = Dict(
    "time" => mod_uct_time,
    "final_loss" => mod_uct_final_loss,
    "losses" => mod_uct_losses
)

println(@sprintf("\nModified UCT: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                mod_uct_time, mod_uct_final_loss, initial_loss - mod_uct_final_loss))

# ----------------------------------------------------------------------------
# 5.5 Flat UCB
# ----------------------------------------------------------------------------
println("\n[5.5] Flat UCB")
println("-"^70)

flat_ucb_config = NAML.FlatUCBConfig(
    max_depth=10,
    num_simulations=100,
    beta=0.05,
    degree=3
)
optim_flat_ucb = NAML.flat_ucb_descent_init(initial_param, loss, flat_ucb_config)

flat_ucb_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_flat_ucb)
    push!(flat_ucb_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_flat_ucb)
end
t_end = time()
flat_ucb_time = t_end - t_start

flat_ucb_final_loss = NAML.eval_loss(optim_flat_ucb)
push!(flat_ucb_losses, flat_ucb_final_loss)

results["Flat UCB"] = Dict(
    "time" => flat_ucb_time,
    "final_loss" => flat_ucb_final_loss,
    "losses" => flat_ucb_losses
)

println(@sprintf("\nFlat UCB: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                flat_ucb_time, flat_ucb_final_loss, initial_loss - flat_ucb_final_loss))

# ----------------------------------------------------------------------------
# 5.6 HOO
# ----------------------------------------------------------------------------
println("\n[5.6] HOO (Hierarchical Optimistic Optimization)")
println("-"^70)

hoo_config = NAML.HOOConfig(
    rho=0.5,
    nu1=0.1,
    max_depth=15
)
optim_hoo = NAML.hoo_descent_init(initial_param, loss, hoo_config)

hoo_losses = Float64[]
t_start = time()
for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim_hoo)
    push!(hoo_losses, current_loss)
    println(@sprintf("  Epoch %2d: loss = %.6e", epoch, current_loss))
    NAML.step!(optim_hoo)
end
t_end = time()
hoo_time = t_end - t_start

hoo_final_loss = NAML.eval_loss(optim_hoo)
push!(hoo_losses, hoo_final_loss)

# Get tree statistics
tree_size = NAML.get_tree_size(optim_hoo.state.root)
visited = NAML.get_visited_nodes(optim_hoo.state.root)
leaves = NAML.get_leaf_nodes(optim_hoo.state.root)

results["HOO"] = Dict(
    "time" => hoo_time,
    "final_loss" => hoo_final_loss,
    "losses" => hoo_losses,
    "tree_size" => tree_size,
    "visited" => visited,
    "leaves" => leaves
)

println(@sprintf("\nHOO: Time = %.3f s, Final Loss = %.6e, Improvement = %.6e",
                hoo_time, hoo_final_loss, initial_loss - hoo_final_loss))
println("  Tree size: $tree_size nodes, Visited: $visited, Leaves: $leaves")

# ============================================================================
# Comparison Table
# ============================================================================
println("\n" * "="^70)
println("COMPARISON TABLE")
println("="^70)
println()
println(@sprintf("%-20s %15s %15s %15s", "Optimizer", "Time (s)", "Final Loss", "Improvement"))
println("-"^70)

optimizers = ["Greedy", "MCTS", "UCT", "Modified UCT", "Flat UCB", "HOO"]
for opt in optimizers
    r = results[opt]
    improvement = initial_loss - r["final_loss"]
    println(@sprintf("%-20s %15.3f %15.6e %15.6e", opt, r["time"], r["final_loss"], improvement))
end

println("="^70)

# ============================================================================
# Loss Trajectories
# ============================================================================
println("\n" * "="^70)
println("LOSS TRAJECTORIES")
println("="^70)
println()
println(@sprintf("%-6s %15s %15s %15s %15s %15s %15s",
                "Epoch", "Greedy", "MCTS", "UCT", "Mod-UCT", "Flat-UCB", "HOO"))
println("-"^100)

max_epochs = length(results["Greedy"]["losses"])
for epoch in 1:max_epochs
    println(@sprintf("%-6d %15.6e %15.6e %15.6e %15.6e %15.6e %15.6e",
                    epoch - 1,
                    results["Greedy"]["losses"][epoch],
                    results["MCTS"]["losses"][epoch],
                    results["UCT"]["losses"][epoch],
                    results["Modified UCT"]["losses"][epoch],
                    results["Flat UCB"]["losses"][epoch],
                    results["HOO"]["losses"][epoch]))
end

println("="^100)

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println()
println("Problem: Learn polynomial coefficients to interpolate p-adic data")
println("  - Input data: $(n_points) p-adic points (x_i, y_i)")
println("  - Model: polynomial of degree $poly_degree")
println("  - Parameters: $(poly_degree + 1) coefficients to optimize")
println("  - Initial loss: $initial_loss")
println()

# Find best optimizer
best_opt = ""
best_loss = Inf
for opt in optimizers
    if results[opt]["final_loss"] < best_loss
        best_loss = results[opt]["final_loss"]
        best_opt = opt
    end
end

println("Best optimizer: $best_opt")
println("  - Final loss: $best_loss")
println("  - Time: $(results[best_opt]["time"]) seconds")
println("  - Improvement: $(initial_loss - best_loss)")
println()

println("Key insights:")
println("  - Greedy descent provides a fast baseline")
println("  - Tree search methods explore the parameter space systematically")
println("  - HOO provides detailed tree statistics for analysis")
println("  - Trade-off between computation time and solution quality")
println("="^70)
