"""
Sanity Check Configuration: MCTS vs DAG-MCTS

Tests MCTS and DAG-MCTS performance across:
- Different simulation counts: 50, 100, 200
- Different degree parameters: 1, 2, dimension-matched
- Representative problem instances

For function learning, dimension = n_points (number of coefficients to learn)
"""

sanity_experiments = [
    # 3-point problem (dimension = 3: a₀, a₁, a₂ for degree 2)
    Dict("name" => "sanity_p2_zero_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # 4-point problem (dimension = 4: a₀, a₁, a₂, a₃ for degree 3)
    Dict("name" => "sanity_p2_one_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # 6-point problem (dimension = 6: a₀, ..., a₅ for degree 5)
    Dict("name" => "sanity_p2_zero_deg5", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# Use sanity check experiments
experiment_configs = sanity_experiments
