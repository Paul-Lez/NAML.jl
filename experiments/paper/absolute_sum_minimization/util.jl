"""
Utility functions for absolute sum minimization experiments.

Provides:
1. Random polynomial generation (both AbsolutePolynomialSum and LinearAbsolutePolynomialSum)
2. Loss function creation for minimizing sums of absolute polynomials
"""

using Oscar

"""
    generate_random_linear_polynomial(K::PadicField, num_vars::Int) -> NAML.LinearPolynomial

Generate a random linear polynomial with coefficients in K.

Returns a LinearPolynomial representing: a₁x₁ + a₂x₂ + ... + aₙxₙ + b
where coefficients are random p-adic numbers.

# Arguments
- `K::PadicField`: The p-adic field for coefficients
- `num_vars::Int`: Number of variables

# Returns
- A `LinearPolynomial` with random coefficients
"""
function generate_random_linear_polynomial(K::PadicField, num_vars::Int)
    # Generate random coefficients for each variable plus constant
    p = Int(Oscar.prime(K))
    prec = Oscar.precision(K)
    coeffs = [generate_random_padic(p, prec, 0, 8) for _ in 1:num_vars]
    constant = generate_random_padic(p, prec, 0, 8)
    return NAML.LinearPolynomial(coeffs, constant)
end


"""
    generate_random_polynomial(K::PadicField, num_vars::Int, degree::Int, var_names::Vector{String})
    -> AbstractAlgebra.Generic.MPoly

Generate a random polynomial with coefficients in K.

# Arguments
- `K::PadicField`: The p-adic field for coefficients
- `num_vars::Int`: Number of variables
- `degree::Int`: Maximum degree of the polynomial
- `var_names::Vector{String}`: Names for the variables

# Returns
- A multivariate polynomial with random coefficients
"""
function generate_random_polynomial(K::PadicField, num_vars::Int, degree::Int, var_names::Vector{String})
    R, vars = polynomial_ring(K, var_names)
    p = Int(Oscar.prime(K))
    prec = Oscar.precision(K)

    # Generate random polynomial by summing random monomials
    poly = R(0)

    # Add constant term
    poly += generate_random_padic(p, prec, 0, 8)

    # Add linear terms
    for i in 1:num_vars
        coeff = generate_random_padic(p, prec, 0, 8)
        poly += coeff * vars[i]
    end

    # Add higher degree terms if degree > 1
    if degree >= 2
        # Add quadratic terms (including cross terms)
        for i in 1:num_vars
            for j in i:num_vars
                coeff = generate_random_padic(p, prec, 0, 8)
                poly += coeff * vars[i] * vars[j]
            end
        end
    end

    # Could extend to higher degrees if needed
    if degree > 2
        # For now, just add some cubic terms as example
        for i in 1:min(num_vars, 2)  # Limit to avoid too many terms
            coeff = generate_random_padic(p, prec, 0, 8)
            poly += coeff * vars[i]^3
        end
    end

    return poly
end


"""
    create_absolute_sum_loss(polys::Vector{<:NAML.PolydiscFunction{S}}) where S
    -> NAML.Loss

Create a loss function that computes the sum of absolute values: |f₁| + |f₂| + ... + |fₙ|

# Arguments
- `polys::Vector{<:NAML.PolydiscFunction{S}}`: Vector of polynomial functions

# Returns
- A `NAML.Loss` struct that evaluates the sum of absolute values
"""
function create_absolute_sum_loss(polys::Vector{<:NAML.PolydiscFunction{S}}) where S
    # Create the sum: |f₁| + |f₂| + ... + |fₙ|
    total_sum = sum(polys)

    # Create batch evaluator
    batch_eval = NAML.batch_evaluate_init(total_sum)
    batch_fn = (params) -> map(batch_eval, params)

    return NAML.Loss(batch_fn, x -> 0)
end


"""
    generate_random_absolute_sum_problem(p::Int, prec::Int, num_polys::Int,
                                        num_vars::Int, degree::Int)
    -> NAML.Loss

Generate a random absolute sum minimization problem.

Creates random polynomials f₁, ..., fₙ and returns a loss function L(x) = |f₁(x)| + ... + |fₙ(x)|

For linear polynomials (degree=1), uses LinearAbsolutePolynomialSum for efficiency.
For higher degrees, uses AbsolutePolynomialSum.

# Arguments
- `p::Int`: Prime for p-adic field
- `prec::Int`: p-adic precision
- `num_polys::Int`: Number of polynomials in the sum
- `num_vars::Int`: Number of variables (dimension)
- `degree::Int`: Degree of each polynomial (1=linear, 2=quadratic, etc.)

# Returns
- A `NAML.Loss` function that can be minimized
"""
function generate_random_absolute_sum_problem(p::Int, prec::Int, num_polys::Int,
                                             num_vars::Int, degree::Int)
    K = PadicField(p, prec)

    if degree == 1
        # Use LinearAbsolutePolynomialSum for linear case (optimized)
        linear_polys = [generate_random_linear_polynomial(K, num_vars) for _ in 1:num_polys]
        poly_funcs = [NAML.LinearAbsolutePolynomialSum([lp]) for lp in linear_polys]
    else
        # Use AbsolutePolynomialSum for higher degrees
        var_names = ["x$i" for i in 1:num_vars]
        R, _ = polynomial_ring(K, var_names)

        polys = [generate_random_polynomial(K, num_vars, degree, var_names) for _ in 1:num_polys]
        poly_funcs = [NAML.AbsolutePolynomialSum([poly]) for poly in polys]
    end

    return create_absolute_sum_loss(poly_funcs)
end


"""
    generate_initial_point(num_vars::Int, K::PadicField) -> NAML.ValuationPolydisc

Generate an initial point for optimization (Gauss point).

# Arguments
- `num_vars::Int`: Dimension of the space
- `K::PadicField`: The p-adic field

# Returns
- A `ValuationPolydisc` centered at (1, 1, ..., 1) with radius 0
"""
function generate_initial_point(num_vars::Int, K::PadicField)
    center = ntuple(i -> K(1), num_vars)
    radius = ntuple(i -> 0, num_vars)
    return NAML.ValuationPolydisc(center, radius)
end
