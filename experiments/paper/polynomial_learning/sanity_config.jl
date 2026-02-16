"""
Sanity Check Configuration: MCTS vs DAG-MCTS

Tests MCTS and DAG-MCTS performance across:
- Different simulation counts: 50, 100, 200
- Different degree parameters: 1, 2, dimension-matched
- Representative problem instances

For polynomial learning, dimension = degree + 1 (number of coefficients)
"""

sanity_experiments = [
    # 2-adic, degree 2 (dimension = 3 coefficients: a₀, a₁, a₂)
    Dict("name" => "sanity_2adic_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),

    # 2-adic, degree 3 (dimension = 4 coefficients)
    Dict("name" => "sanity_2adic_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),

    # 2-adic, degree 5 (dimension = 6 coefficients)
    Dict("name" => "sanity_2adic_deg5", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),
]

# Use sanity check experiments
experiment_configs = sanity_experiments
