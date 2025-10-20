@doc raw"""
    AbstractModel{S}

A model structure that captures the underlying function and parameter/variable mapping.

Represents a model without specified parameter values. Identifies which variables in the
function are data variables vs. parameters.

# Fields
- `fun::PolydiscFunction{S}`: The underlying function (sum of absolute polynomials)
- `param_info`: Binary vector indicating which variables are data (true/1) vs. parameters (false/0)

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)

# Example
If the function is ``f(x_1, \theta_1, x_2, \theta_2)`` with data variables ``x_1, x_2`` and
parameters ``\theta_1, \theta_2``, then `param_info = [1, 0, 1, 0]`.
"""
struct AbstractModel{S}
    fun::PolydiscFunction{S}
    # The data of which variables are parameters
    # E.g. if the function is f(x_1, θ_1, x_2, θ_2) then param_info = [1, 0, 1, 0]
    param_info
end

@doc raw"""
    Model{S, T}

A complete model with specified parameter values.

Combines an abstract model (function + parameter mapping) with concrete parameter values.
The structure is mutable to allow parameter updates during optimization.

# Fields
- `fun::AbstractModel{S}`: The abstract model (function and parameter info)
- `param::ValuationPolydisc{S, T}`: The current parameter values

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
- `T`: The type for radius/valuation values
"""
mutable struct Model{S, T}
    fun::AbstractModel{S}
    # the values of the parameters
    param::ValuationPolydisc{S, T}
end

@doc raw"""
    update_weights!(m::Model, param)

Update the parameter values of a model in place.

# Arguments
- `m::Model`: The model to update
- `param`: New parameter values

# Notes
Mutates the model structure directly since `Model` is mutable.
"""
function update_weights!(m::Model, param)
    m.param = param
end

@doc raw"""
    var_indices(m::AbstractModel)

Get the indices of data variables in an abstract model.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Indices where `param_info` is true (data variables)

# Example
For `param_info = [true, true, false, false]`, returns `[1, 2]`
"""
function var_indices(m::AbstractModel)
    return findall(x -> x, m.param_info)
end

@doc raw"""
    param_indices(m::AbstractModel)

Get the indices of parameters in an abstract model.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Indices where `param_info` is false (parameters)

# Example
For `param_info = [true, true, false, false]`, returns `[3, 4]`
"""
function param_indices(m::AbstractModel)
    return findall(x -> !x, m.param_info)
end

@doc raw"""
    getkeys(m::AbstractModel)

Map each model variable to its position within data variables or parameters.

For each variable in the model, returns its index within either the data variables or
the parameters, depending on its type.

# Arguments
- `m::AbstractModel`: The abstract model

# Returns
`Vector{Int}`: Array ``[a_1, \ldots, a_n]`` where ``a_i`` is the index of the ``i``-th
variable within its category (data or parameter)

# Example
For ``f(x, \theta, y, z, \phi)`` with parameters ``\theta, \phi``, returns `[1, 1, 2, 3, 2]`
since ``x`` is the 1st data variable, ``\theta`` is the 1st parameter, ``y`` is the 2nd data
variable, ``z`` is the 3rd data variable, and ``\phi`` is the 2nd parameter.
"""
function getkeys(m::AbstractModel)
    vars = var_indices(m)
    param = param_indices(m)
    return [m.param_info[i] ? findfirst(item -> item == i, vars) : findfirst(item -> item == i, param) for i in Base.eachindex(m.param_info)]
end

@doc raw"""
    set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S, T}, param::ValuationPolydisc{S, T}) where {S,T}

Construct a point for evaluation by interleaving data and parameter values.

Given data variable values and parameter values, constructs a point in the full model space
that can be evaluated using polynomial evaluation mechanisms.

# Arguments
- `m::AbstractModel{S}`: The abstract model
- `val::ValuationPolydisc{S, T}`: Data variable values
- `param::ValuationPolydisc{S, T}`: Parameter values

# Returns
`ValuationPolydisc{S, T}`: Point with data and parameters interleaved according to `param_info`

# Example
For model ``f(x, \theta, y, \phi)`` with data ``(x, y) = (1, 2)`` and parameters
``(\theta, \phi) = (3, 4)``, returns the point ``(1, 3, 2, 4)``.
"""
function set_abstract_model_variable(m::AbstractModel{S}, val::ValuationPolydisc{S, T}, param::ValuationPolydisc{S, T}) where S where T
    keys = getkeys(m)
    abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
    abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])
    #println(length(abstract_model_variable_center))
    return ValuationPolydisc{S, T}(abstract_model_variable_center, abstract_model_variable_radius)
end 

@doc raw"""
    set_model_variable(m::Model{S, T}, val::ValuationPolydisc{S, T}) where {S,T}

Construct an evaluation point using a model's current parameters and given data.

Convenience wrapper around `set_abstract_model_variable` that uses the model's stored
parameter values.

# Arguments
- `m::Model{S, T}`: The model (with stored parameters)
- `val::ValuationPolydisc{S, T}`: Data variable values

# Returns
`ValuationPolydisc{S, T}`: Point with data and model parameters interleaved for evaluation
"""
function set_model_variable(m::Model{S, T}, val::ValuationPolydisc{S, T}) where S where T
    return set_abstract_model_variable(m.fun, val, m.param)
end

# # This function is deprecated.
# # TODO Paul: remove this
# function specialise_abstract_model_data(m::AbstractModel{S}, val::ValuationPolydisc{S, T}) where S where T
#     keys = getkeys(m)
#     R = parent(model.fun)
#     x = gens(R)
#     abstract_model_variable_radius = Vector{T}([m.param_info[i] ? val.radius[keys[i]] : param.radius[keys[i]] for i in Base.eachindex(m.param_info)])
#     abstract_model_variable_center = Vector{S}([m.param_info[i] ? val.center[keys[i]] : param.center[keys[i]] for i in Base.eachindex(m.param_info)])

# end

@doc raw"""
    eval_abs(m::AbstractModel, val, param)

Evaluate an abstract model at given data and parameter values.

# Arguments
- `m::AbstractModel`: The abstract model
- `val`: Data variable values
- `param`: Parameter values

# Returns
`Float64`: The model evaluation result

# Notes
Current implementation is specific to absolute polynomial sums. Will need updates for
more general model functions.
"""
function eval_abs(m::AbstractModel, val, param)
    var = set_abstract_model_variable(m, val, param)
    return eval_abs(m.fun, var)
end

@doc raw"""
    eval_abs(m::Model, val)

Evaluate a model at given data using the model's stored parameters.

# Arguments
- `m::Model`: The model (with stored parameters)
- `val`: Data variable values

# Returns
`Float64`: The model evaluation result
"""
function eval_abs(m::Model, val)
    return eval_abs(m.fun, val, m.param)
end