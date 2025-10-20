## This file contains the basic of functions on the polydisc space and their calculus

@doc raw"""
    evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S, T}) where {S,T}

Evaluate the valuation (absolute value) of a multivariate polynomial at a polydisc.

Computes ``\max_{n} |a_n|_p \cdot p^{-\sum_i r_i n_i}`` where ``f = \sum_n a_n T^n`` is the
polynomial expansion around the center, ``|·|_p`` is the p-adic absolute value, and ``r_i``
are the radii.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `p::ValuationPolydisc{S, T}`: The polydisc at which to evaluate

# Returns
`Float64`: The absolute value of the polynomial at the polydisc
"""
function evaluate_abs(f::AbstractAlgebra.Generic.MPoly{S}, p::ValuationPolydisc{S, T}) where S where T
    t = gens(f.parent)
    # Is this the right thing to compute?
    vec = [t[i] + p.center[i] for i in eachindex(p.center)]
    g = AbstractAlgebra.evaluate(f, vec)
    # TODO Paul: check this
    max, _ = findmax([padic_abs(Nemo.coeff(g, v)) * (Float64(prime(p))^(-sum(p.radius .* v))) for v in Nemo.exponent_vectors(g)])
    return max
end

@doc raw"""
    PolydiscFunction{S}

A function on polydisc spaces represented as a sum of absolute values of polynomials.

Represents functions of the form ``F = \sum_i |f_i|`` where each ``f_i`` is a multivariate
polynomial. This is a specialized case that will later be generalized to compositions of
polynomial vectors with differentiable functions.

# Fields
- `polys::Vector{AbstractAlgebra.Generic.MPoly{S}}`: Vector of polynomials whose absolute values are summed

# Type Parameters
- `S`: The coefficient type (typically p-adic numbers)
"""
struct PolydiscFunction{S}
    polys::Vector{AbstractAlgebra.Generic.MPoly{S}}
end

@doc raw"""
    parent(F::PolydiscFunction{S}) where S

Get the parent polynomial ring of the first polynomial in the function.

# Arguments
- `F::PolydiscFunction{S}`: The polydisc function

# Returns
The parent ring of the first polynomial in `F.polys`
"""
function parent(F::PolydiscFunction{S}) where S
    return parent(F[1])
end

# At the moment we work with multiple differential operators: the directional derivative along a tangent vector, and the gradient at a point.


@doc raw"""
    directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S, T}) where {S,T}

Compute the directional exponent of a polynomial along a tangent vector.

The directional exponent is the exponent vector ``n`` such that when moving in the direction
of ``v``, the absolute value ``|f|`` is asymptotically given by a monomial term with exponent ``n``.
This corresponds to the dominant term(s) when expanding around the tangent direction.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `v::ValuationTangent{S, T}`: The tangent vector direction

# Returns
`Vector{Int}`: Indices of the dominant exponent vectors (may return multiple if there's a tie)

# Notes
The directional exponent is not uniquely defined when multiple terms achieve the maximum
absolute value. In this case, all minimal exponents achieving the maximum are returned.
"""
function directional_exponent(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S, T}) where S where T
    t = gens(f.parent)
    g = AbstractAlgebra.evaluate(f, t+v.direction)
    abs_terms = [padic_abs(Nemo.coeff(g, n)) * prod(v.point.radius .^ n)  for n in Nemo.exponent_vectors(g)]
    # Find all exponents at which the max is attained
    max_exponents = findall(a -> a == maximum(abs_terms), abs_terms)
    # In principle this if clause isn't necessary (the "else" part works for all possible cases)
    # However I think this makes things faster.
    # TODO Paul: do some benchmarking to see if that's the case.
    if length(max_exponents) == 1
        return max_exponents
    else
        # Find minimal exponents at which the max is attained. These
        # are the directional exponents.
        return findall(a -> sum(a) == minimum([sum(n) for n in max_exponents]), max_exponents)
    end
end

@doc raw"""
    directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S, T}) where {S,T}

Compute the directional derivative of an absolute polynomial along a tangent vector.

Uses the fact that if locally in the direction of ``v``, ``|f| = a_n r^n`` for some multi-index ``n``,
then the directional derivative is ``d_v |f| = -|n| \cdot |a_n| \cdot r^n`` where ``r`` is the
radius vector at the basepoint of ``v``.

# Arguments
- `f::AbstractAlgebra.Generic.MPoly{S}`: A multivariate polynomial
- `v::ValuationTangent{S, T}`: The tangent vector direction

# Returns
`Float64`: The directional derivative value

# Notes
This implementation is specialized for absolute polynomial functions and will need to be
reimplemented for the general differentiable case.
"""
function directional_derivative(f::AbstractAlgebra.Generic.MPoly{S}, v::ValuationTangent{S, T}) where S where T
    # Recover the variables of the polynomial ring we're working over
    x = gens(f.parent)
    # Compute the expansion of f around the direction a of the tangent vector v, i.e.
    # The coefficients a_n such that f = ∑_n a_n (T-a)^n. We do this by computing the
    # expansion around 0 of the polynomial g(T) = f(T+a).
    #p = [j != i ? point(v)[j] : x for j in eachindex(v.tangents)]
    g = AbstractAlgebra.evaluate(f, x+v.direction)
    # Next we need to compute the directional exponent of f along v
    n = first(directional_exponent(f, v))
    # Use the formula to get d_v
    d_v = - sum(n) * padic_abs(coeff(g, n)) * (Float64(prime(v.point))^(-sum(v.point.radius .* n))) # prod(v.point.radius .^ n)
    return d_v
end

@doc raw"""
    directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S, T}) where {S,T}

Compute the directional derivative of a polydisc function along a tangent vector.

For a function ``F = \sum_i |f_i|``, computes ``d_v F = \sum_i d_v |f_i|``.

# Arguments
- `fun::PolydiscFunction{S}`: The polydisc function (sum of absolute polynomials)
- `v::ValuationTangent{S, T}`: The tangent vector direction

# Returns
`Float64`: The directional derivative value
"""
function directional_derivative(fun::PolydiscFunction{S}, v::ValuationTangent{S, T}) where S where T
    return sum([directional_derivative(f, v) for f in fun.polys])
end

@doc raw"""
    grad(f, v::ValuationTangent{S, T}) where {S,T}

Compute the gradient of a function at a tangent vector.

Computes the gradient by taking directional derivatives along all standard basis directions.

# Arguments
- `f`: The function (polynomial or polydisc function)
- `v::ValuationTangent{S, T}`: The tangent vector providing the basepoint

# Returns
`Vector{Float64}`: The gradient vector with components for each coordinate direction

# Notes
Currently references undefined variables `P` and `Q` - this may need to be fixed.
"""
function grad(f, v::ValuationTangent{S, T}) where S where T
    return [directional_derivative(f, basis_vector(P, Q, i)) for i in Base.eachindex(Q)]
end

@doc raw"""
    partial_gradient(f, v::ValuationTangent{S, T}, gradient_indices) where {S,T}

Compute the partial gradient of a function for a subset of coordinate directions.

Similar to `grad`, but only computes gradient components for the specified coordinate indices.

# Arguments
- `f`: The function (polynomial or polydisc function)
- `v::ValuationTangent{S, T}`: The tangent vector providing the basepoint
- `gradient_indices`: Collection of coordinate indices to include in the partial gradient

# Returns
`Vector{Float64}`: The partial gradient vector with components only for specified indices
"""
function partial_gradient(f, v::ValuationTangent{S, T}, gradient_indices) where S where T
    return [directional_derivative(f, basis_vector(v, i)) for i in gradient_indices]
end

@doc raw"""
    eval_abs(fun::PolydiscFunction{S}, var::ValuationPolydisc{S, T}) where {S,T}

Evaluate a polydisc function at a polydisc.

For a function ``F = \sum_i |f_i|``, computes ``F(var) = \sum_i |f_i|(var)``.

# Arguments
- `fun::PolydiscFunction{S}`: The polydisc function (sum of absolute polynomials)
- `var::ValuationPolydisc{S, T}`: The polydisc at which to evaluate

# Returns
`Float64`: The function value at the polydisc
"""
function eval_abs(fun::PolydiscFunction{S}, var::ValuationPolydisc{S, T}) where S where T
    return sum([evaluate_abs(f, var) for f in fun.polys])
end
