# Test file for polynomial learning via HOO algorithm.
#
# This file demonstrates learning the roots of a cubic polynomial
# (x - a)(x - b)(x - c) using HOO optimization in 2-adic space.
# The task is to find parameters a, b, c that minimize the loss function
# given several data points (roots).
#
# This is a comparison test to polynomial_learning.jl which uses greedy descent.

using Printf

include("../src/naml.jl")
include("../src/optim/mcts/hoo.jl")

println("="^70)
println("Polynomial Learning with HOO Algorithm")
println("="^70)

# Initialize 2-adic field with precision
p, prec = 2, 20
K = PadicField(p, prec)
println("\n[1] Setup: 2-adic field with precision $prec")

# Create polynomial ring with variables: x (data), a, b, c (parameters)
R, (x, a, b, c) = K["x", "a", "b", "c"]
g = AbsolutePolynomialSum([(x - a) * (x - b) * (x - c)])
println("Function: f(x, a, b, c) = (x - a)(x - b)(x - c)")

# Create model: x is a data variable (true), a,b,c are parameters (false)
f = AbstractModel(g, [true, false, false, false])

# Setting up training data points
# Each data point is paired with the desired output value
# We want to find a, b, c such that f(x, a, b, c) = 0 at these points
p1 = ValuationPolydisc([K(p^0)], Vector{Int}([prec]))  # x = 1
p2 = ValuationPolydisc([K(p^1)], Vector{Int}([prec]))  # x = 2
p3 = ValuationPolydisc([K(p^2)], Vector{Int}([prec]))  # x = 4
p4 = ValuationPolydisc([K(3)], Vector{Int}([prec]))    # x = 3
data = [(p1, 0), (p2, 0), (p3, 0), (p4, -2)]

println("\n[2] Training data: 4 points")
println("  x=1, target=0")
println("  x=2, target=0")
println("  x=4, target=0")
println("  x=3, target=-2")

# Set initial parameter values
initial_param = ValuationPolydisc([K(11), K(22), K(33)], [0, 0, 0])
model = Model(f, initial_param)
println("\n[3] Initial parameters: a=11, b=22, c=33")

# Create Mean p-Power Error (MPE) loss function with p=2 (MSE-like)
ell = MPE_loss_init(f, data, 2)
println("\n[4] Loss function: Mean p-Power Error (p=2)")

# HOO Configuration
# For multi-parameter (3D) problems, adjust parameters appropriately
config = HOOConfig(
    rho = 0.5,        # Shrinkage rate
    nu1 = 1.0,        # Smoothness parameter (higher for more dimensions)
    max_depth = 12,   # Maximum tree depth
    degree = 1,       # Use degree 1 for children generation
    strict = false    # Use full children
)

println("\n[5] HOO Configuration:")
println("  ρ = $(config.rho)")
println("  ν₁ = $(config.nu1)")
println("  max_depth = $(config.max_depth)")
println("  degree = $(config.degree)")

# Initialize HOO optimizer
hoo_optim = hoo_descent_init(initial_param, ell, config)

println("\n[6] Running HOO optimization...")
println("-"^70)

# Run HOO optimization
N_epochs = 50
initial_loss = eval_loss(hoo_optim)
println(@sprintf("Initial loss: %.10f", initial_loss))
println()

t1 = time()
for i in 1:N_epochs
    step!(hoo_optim)

    if i % 10 == 0 || i == 1
        current_loss = eval_loss(hoo_optim)
        state = hoo_optim.state
        tree_size = get_tree_size(state.root)
        visited_nodes = get_visited_nodes(state.root)

        @printf("Epoch %3d: Loss = %.10f\n", i, current_loss)
        println("          Tree: $(tree_size) nodes, $(length(visited_nodes)) visited, " *
                "$(state.total_samples) samples")
    end
end
t2 = time()

println("-"^70)
println(@sprintf("\n[7] HOO Results (%.2f seconds):", t2 - t1))
final_loss = eval_loss(hoo_optim)
@printf("  Final loss: %.10f\n", final_loss)
@printf("  Improvement: %.2fx\n", initial_loss / final_loss)

println("\n  Final parameters:")
param_centers = center(hoo_optim.param)
param_radii = radius(hoo_optim.param)
println("    a = $(param_centers[1]) ± 2^($(param_radii[1]))")
println("    b = $(param_centers[2]) ± 2^($(param_radii[2]))")
println("    c = $(param_centers[3]) ± 2^($(param_radii[3]))")

state = hoo_optim.state
println("\n  Tree statistics:")
println("    Total nodes: $(get_tree_size(state.root))")
println("    Visited nodes: $(length(get_visited_nodes(state.root)))")
println("    Total samples: $(state.total_samples)")

##################################################
# Comparison with Greedy Descent
##################################################

println("\n" * "="^70)
println("Comparison with Greedy Descent")
println("="^70)

# Reset to same initial point
greedy_param = ValuationPolydisc([K(11), K(22), K(33)], [0, 0, 0])
greedy_optim = greedy_descent_init(greedy_param, ell, 1, (false, 1))

println("\n[8] Running greedy descent for $N_epochs epochs...")
t3 = time()
for i in 1:N_epochs
    step!(greedy_optim)
end
t4 = time()

greedy_loss = eval_loss(greedy_optim)
println(@sprintf("\n[9] Greedy Descent Results (%.2f seconds):", t4 - t3))
@printf("  Final loss: %.10f\n", greedy_loss)
@printf("  Improvement: %.2fx\n", initial_loss / greedy_loss)

println("\n  Final parameters:")
greedy_centers = center(greedy_optim.param)
greedy_radii = radius(greedy_optim.param)
println("    a = $(greedy_centers[1]) ± 2^($(greedy_radii[1]))")
println("    b = $(greedy_centers[2]) ± 2^($(greedy_radii[2]))")
println("    c = $(greedy_centers[3]) ± 2^($(greedy_radii[3]))")

##################################################
# Summary
##################################################

println("\n" * "="^70)
println("Summary")
println("="^70)

@printf("\nLoss comparison:\n")
@printf("  HOO:    %.10f\n", final_loss)
@printf("  Greedy: %.10f\n", greedy_loss)

if final_loss < greedy_loss
    @printf("  → HOO is %.2fx better!\n", greedy_loss / final_loss)
elseif final_loss > greedy_loss
    @printf("  → Greedy is %.2fx better!\n", final_loss / greedy_loss)
else
    println("  → Both methods achieved the same loss")
end

@printf("\nTime comparison:\n")
@printf("  HOO:    %.2f seconds\n", t2 - t1)
@printf("  Greedy: %.2f seconds\n", t4 - t3)

println("\n" * "="^70)
println("Experiment complete!")
println("="^70)
