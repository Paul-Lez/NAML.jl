"""
Function Learning Experiment (Clean Implementation)

This script reimplements the function_learning.ipynb notebook using the
clean utility API from util.jl. It demonstrates polynomial function learning
with p-adic inputs and binary outputs.

The goal: Learn a polynomial f(x) = a₀ + a₁x + ... + aₙx^n such that
cutoff(|f(x)|) approximates the binary labels y ∈ {0, 1}.
"""

include("../../src/NAML.jl")
include("util.jl")

using Oscar
using .NAML

println("="^70)
println("Function Learning Experiment - Clean Implementation")
println("="^70)

# ============================================================================
# Configuration
# ============================================================================
println("\n[1] Setting up configuration...")

prec = 20                # p-adic precision
p = 2                    # prime
K = PadicField(p, prec)  # p-adic field

n_points = 3             # number of training data points
poly_degree = 6          # degree of polynomial to learn
cutoff_val = 0.25        # cutoff threshold for binary classification
n_epochs = 20            # number of optimization epochs

println("  - Prime: $p")
println("  - Precision: $prec")
println("  - Training points: $n_points")
println("  - Polynomial degree: $poly_degree")
println("  - Cutoff value: $cutoff_val")
println("  - Epochs: $n_epochs")

# ============================================================================
# Data Generation
# ============================================================================
println("\n[2] Generating training data...")

# Generate random p-adic x values and binary y values
# Using default parameters (min_exp=0, num_terms=10) ensures distinct x values
data = generate_polynomial_learning_data(p, prec, n_points)

println("  Generated $n_points data points:")
for (i, (x, y)) in enumerate(data)
    println("    Point $i: x = $x, y = $y")
end

# Verify all x values are distinct
x_vals = [x for (x, y) in data]
all_distinct = length(x_vals) == length(unique(x_vals))
println("  All x values distinct: $all_distinct")

# ============================================================================
# Loss Function Creation
# ============================================================================
println("\n[3] Creating loss function...")

# Use the polynomial_to_linear_loss utility to create the loss
# This transforms the polynomial learning problem into a linear one
loss = polynomial_to_linear_loss(data, poly_degree, cutoff_val)

println("  Loss function created successfully")
println("  Loss type: Polynomial learning with cutoff")
println("  Parameters to learn: $(poly_degree + 1) coefficients (a₀, ..., a$poly_degree)")

# ============================================================================
# Optimizer Initialization
# ============================================================================
println("\n[4] Initializing optimizer...")

# Start at the Gauss point (standard starting point in p-adic optimization)
initial_param = generate_gauss_point(poly_degree + 1, K)
println("  Initial parameter: Gauss point in $(poly_degree + 1) dimensions")
println("  Center: $(NAML.center(initial_param))")
println("  Radius: $(NAML.radius(initial_param))")

# Evaluate initial loss
initial_loss = loss.eval([initial_param])[1]
println("  Initial loss: $initial_loss")

# Initialize greedy descent optimizer
state = 1
config = (true, 0)  # Force all coordinates to shrink progressively
optim = NAML.greedy_descent_init(initial_param, loss, state, config)

println("  Optimizer: Greedy Descent")
println("  Strategy: Progressively refine all coordinates")

# ============================================================================
# Training Loop
# ============================================================================
println("\n[5] Running optimization...")
println("  " * "-"^66)

t_start = time()

for epoch in 1:n_epochs
    current_loss = NAML.eval_loss(optim)
    current_radius = NAML.radius(optim.param)

    # Print progress
    println("  Epoch $epoch: loss = $current_loss, radius = $current_radius")

    # Take optimization step
    NAML.step!(optim)
end

t_end = time()
elapsed = t_end - t_start

println("  " * "-"^66)

# ============================================================================
# Results
# ============================================================================
println("\n[6] Optimization completed!")

final_loss = NAML.eval_loss(optim)
final_param = optim.param
final_center = NAML.center(final_param)
final_radius = NAML.radius(final_param)

println("\n  Final Results:")
println("  " * "="^66)
println("  Final loss:         $final_loss")
println("  Time elapsed:       $(round(elapsed, digits=2)) seconds")
println("  Loss improvement:   $(round(initial_loss - final_loss, digits=4))")
println("\n  Learned coefficients (center):")
for (i, coeff) in enumerate(final_center)
    println("    a$(i-1) = $coeff")
end
println("\n  Parameter radius (uncertainty):")
println("    $final_radius")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Summary")
println("="^70)
println("The optimizer explored the parameter space by progressively refining")
println("each coordinate. The final polydisc represents a region where the")
println("learned polynomial f(x) = a₀ + a₁x + ... + a$(poly_degree)x^$(poly_degree)")
println("achieves loss = $final_loss on the training data.")
println("\nThe radius vector shows how much each coefficient was refined.")
println("Larger radius values indicate more exploration in that coordinate.")
println("="^70)
