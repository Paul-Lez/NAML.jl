"""
Sanity check for polynomial solving experiments.

Runs a minimal experiment to verify the pipeline works:
- Generates a polynomial with a known root
- Runs a few epochs with Best-First and MCTS
- Verifies loss decreases

Usage:
    julia --project=. experiments/paper/polynomial_solving/sanity_run.jl
"""

include("../../../src/NAML.jl")
include("../util.jl")
include("util.jl")

using Oscar
using .NAML
using Printf
using Random

Random.seed!(42)

println("="^60)
println("POLYNOMIAL SOLVING - SANITY CHECK")
println("="^60)

# Test each (num_vars, degree) combination
for num_vars in [1, 2, 3]
    for degree in [1, 2, 3]
        println("\n--- $num_vars variable(s), degree $degree ---")

        p = 2
        prec = 20
        K = PadicField(p, prec)

        loss, root = generate_polynomial_solving_problem(p, prec, num_vars, degree)

        initial_param = generate_initial_point(num_vars, K)
        initial_loss = loss.eval([initial_param])[1]
        @printf("  Initial loss: %.6e\n", initial_loss)

        # Run Best-First for 10 steps
        config = (false, 1)
        optim = NAML.greedy_descent_init(initial_param, loss, 1, config)
        for _ in 1:10
            NAML.step!(optim)
        end
        bf_loss = NAML.eval_loss(optim)
        @printf("  Best-First (10 steps): %.6e\n", bf_loss)

        # Run DAG-MCTS for 5 steps
        dag_config = NAML.DAGMCTSConfig(
            num_simulations=20,
            exploration_constant=1.41,
            degree=1,
            persist_table=true,
            selection_mode=NAML.BestValue
        )
        optim2 = NAML.dag_mcts_descent_init(initial_param, loss, dag_config)
        for _ in 1:5
            NAML.step!(optim2)
        end
        dag_loss = NAML.eval_loss(optim2)
        @printf("  DAG-MCTS-20 (5 steps): %.6e\n", dag_loss)

        # Verify loss decreased
        if bf_loss <= initial_loss && dag_loss <= initial_loss
            println("  ✓ Loss decreased for both optimizers")
        else
            println("  ⚠ Loss did not decrease!")
        end
    end
end

println("\n" * "="^60)
println("✓ Sanity check complete!")
println("="^60)
