"""
Quick test script to verify the absolute sum minimization experiment setup.

Tests:
1. Loading all utilities
2. Generating a simple problem
3. Running a few optimization steps

Usage:
    julia --project=../../.. test_setup.jl
"""

println("="^70)
println("Testing Absolute Sum Minimization Experiment Setup")
println("="^70)

# Test 1: Load dependencies
println("\n[1/4] Loading NAML...")
include("../../../src/NAML.jl")
using .NAML
using Oscar
println("✓ NAML loaded")

# Test 2: Load utilities
println("\n[2/4] Loading utilities...")
include("../util.jl")
include("util.jl")
println("✓ Utilities loaded")

# Test 3: Generate a simple problem
println("\n[3/4] Generating test problem...")
println("  - Creating 2-adic field with precision 20")
p = 2
prec = 20
K = PadicField(p, prec)

println("  - Generating 2 linear polynomials in 1 variable")
num_polys = 2
num_vars = 1
degree = 1

loss = generate_random_absolute_sum_problem(p, prec, num_polys, num_vars, degree)
initial_param = generate_initial_point(num_vars, K)
initial_loss_val = loss.eval([initial_param])[1]

println("  - Initial loss: $initial_loss_val")
println("✓ Problem generated successfully")

# Test 4: Run a few optimization steps
println("\n[4/4] Testing optimization...")
println("  - Initializing Greedy descent")
optim = NAML.greedy_descent_init(initial_param, loss, 1, (false, 1))

println("  - Running optimization (up to 3 steps):")
steps = NAML.optimize!(optim, 3; verbose=true)
final_loss = NAML.eval_loss(optim)
println("  - Final loss: $final_loss")
NAML.has_converged(optim) && println("  - Converged after $steps steps")
println("  - Improvement: $(initial_loss_val - final_loss)")
println("✓ Optimization test completed")

println("\n" * "="^70)
println("✓ ALL TESTS PASSED")
println("="^70)
println("\nThe experiment setup is working correctly!")
println("You can now run experiments with:")
println("  julia --project=../../.. run_experiments.jl --quick")
