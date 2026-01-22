module NAML

using Oscar
using LinearAlgebra
using Printf

# Include all source files
include("basic/valuation.jl")
include("basic/polydisc.jl")
include("basic/tangent_vector.jl")
include("basic/functions.jl")
include("optim/model.jl")
include("optim/basic.jl")
include("optim/gradient_descent.jl")
include("optim/greedy_descent.jl")
include("optim/loss.jl")
include("optim/mcts/hoo.jl")
include("optim/mcts/mcts.jl")
include("optim/mcts/uct.jl")
include("optim/mcts/modified_uct.jl")
include("optim/mcts/flat_ucb.jl")
include("statistics/frechet.jl")

# Export types and functions

# From basic/valuation.jl
export valuation

# From basic/polydisc.jl
export ValuationPolydisc, AbsPolydisc
export center, radius, dim, prime
# Note: join is not exported to avoid conflict with Base.join - use NAML.join explicitly
export dist, children, children_along_branch, concatenate

# From basic/tangent_vector.jl
export ValuationTangent
# Note: zero and basis_vector not exported to avoid conflicts with Base - use NAML.zero, NAML.basis_vector

# From basic/functions.jl
export PolydiscFunction, AbsolutePolynomialSum
export evaluate_abs, directional_exponent, directional_derivative, grad, eval_abs
# Note: evaluate not exported to avoid conflicts with Oscar/AbstractAlgebra - use NAML.evaluate

# From optim/model.jl
export AbstractModel, Model
export var_indices, param_indices, set_abstract_model_variable, batch_evaluate_init

# From optim/basic.jl
export Loss, OptimSetup
export eval_loss, update_param!, step!

# From optim/loss.jl
export MSE_loss_init, MPE_loss_init

# From optim/greedy_descent.jl
export greedy_descent, greedy_descent_init

# From optim/gradient_descent.jl
export gradient_param, gradient_descent, gradient_descent_init

# From optim/mcts/hoo.jl
export HOONode, HOOConfig, HOOState
export hoo_descent, hoo_descent_init
export get_tree_size, get_visited_nodes, get_leaf_nodes

# From optim/mcts/mcts.jl
export MCTSNode, MCTSConfig, MCTSState
export SelectionMode, VisitCount, BestValue
export mcts_descent, mcts_descent_init

# From optim/mcts/uct.jl
export UCTNode, UCTConfig, UCTState
export uct_descent, uct_descent_init

# From optim/mcts/modified_uct.jl
export ModifiedUCTNode, ModifiedUCTConfig, ModifiedUCTState
export modified_uct_descent, modified_uct_descent_init

# From optim/mcts/flat_ucb.jl
export FlatUCBNode, FlatUCBConfig, FlatUCBState
export flat_ucb_descent, flat_ucb_descent_init

# From statistics/frechet.jl
export frechet_mean

end # module NAML
