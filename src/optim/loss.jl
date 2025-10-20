## This file contains various "standard" loss functions for non-Archimedean
## optimisation

############ Loss functions ###################

# Note: The Loss struct is defined in basic.jl
# To specify a loss, one needs to provide a function to evaluate the loss
# and a function to evaluate the gradient of the loss wrt parameters
# (since we don't have any autodiff mechanism implemented yet!)
# Both functions should be closures that capture any necessary data.

#################################################

# Helper functions to construct standard loss functions

@doc raw"""
    MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where {S,T,U}

Initialize a Mean Squared Error (MSE) loss function.

Creates a `Loss` structure with evaluation and gradient functions for MSE loss:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (f(x_i; \theta) - y_i)^2``

# Arguments
- `model::AbstractModel{S}`: The model to optimize
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input, output)` pairs

# Returns
`Loss`: Loss structure with MSE evaluation and gradient functions
"""
function MSE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}) where S where T where U
    # Create a closure that computes the MSE for a given parameter value
    function MSE_compute(param::ValuationPolydisc{S,T}) where S where T
        return 1 / length(data) * sum([(eval_abs(model, val, param) - out)^2 for (val, out) in data])
    end
    # Create a closure that computes the gradient of the loss along a tangent direction v
    # The gradient is evaluated at the current model.param
    # (TODO Paul: Do we want to allow v to weight the sum?)
    function MSE_grad(v::ValuationTangent{S,T}) where S where T
        return 1 / length(data) * sum([2 * (eval_abs(model, val, model.param) - out) * gradient_param(model, val, v) for (val, out) in data])
    end
    return Loss(MSE_compute, MSE_grad)
end

@doc raw"""
    MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where {S,T,U}

Initialize a Mean p-Power Error (MPE) loss function.

Creates a `Loss` structure using the ``\ell^p`` norm instead of ``\ell^2``:
``\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n |f(x_i; \theta) - y_i|^p``

# Arguments
- `model::AbstractModel{S}`: The model to optimize
- `data::Vector{Tuple{ValuationPolydisc{S,T},U}}`: Training data as `(input, output)` pairs
- `p::Int`: The power for the loss (must be finite; for ``p = \infty`` use sup loss - TODO)

# Returns
`Loss`: Loss structure with MPE evaluation and gradient functions
"""
function MPE_loss_init(model::AbstractModel{S}, data::Vector{Tuple{ValuationPolydisc{S,T},U}}, p::Int) where S where T where U
    # MPE is the "Mean p-power error", i.e. same as the MSE but now we use the ℓᵖ norm instead of the ℓ² one.
    # Here we need finite p. For p = ∞, see the sup loss (TODO Paul: implement the sup loss)

    # Create a closure that computes the MPE for a given parameter value
    function MPE_compute(param::ValuationPolydisc{S,T}) where S where T
        return 1 / length(data) * sum([(eval_abs(model, val, param) - out)^p for (val, out) in data])
    end
    # Create a closure that computes the gradient of the loss along a tangent direction v
    # The gradient is evaluated at the current model.param
    # (TODO Paul: Do we want to allow v to weight the sum?)
    function MPE_grad(v::ValuationTangent{S,T}) where S where T
        param = v.point
        return 1 / length(data) * sum([p * (eval_abs(model, val, param) - out)^(p - 1) * gradient_param(model, val, v) for (val, out) in data])
    end
    return Loss(MPE_compute, MPE_grad)
end
