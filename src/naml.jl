using Oscar
using LinearAlgebra
using Random
using Printf
# Include all source files
include("basic/polydisc.jl")
include("basic/tangent_vector.jl")
include("basic/functions.jl")
include("optim/model.jl")
include("optim/basic.jl")
include("optim/gradient_descent.jl")
include("optim/greedy_descent.jl")
include("optim/loss.jl")
include("optim/mcts/mcts.jl")
include("optim/mcts/hoo.jl")
include("optim/mcts/uct.jl")
include("optim/mcts/modified_uct.jl")
include("optim/mcts/flat_ucb.jl")
include("statistics/frechet.jl")
