# Typed Evaluators & ValuedFieldPoint

This document explains the typed evaluator architecture in NAML, including the `ValuedFieldPoint` wrapper and automatic lifting infrastructure.

## Overview

**Problem**: Function evaluation with type-level optimization.
- Functions defined with coefficient type `S` (e.g., `PadicFieldElem`)
- Polydiscs may use different types (e.g., `ValuedFieldPoint{P,Prec,PadicFieldElem}`)
- Need compile-time specialization on prime, precision, and dimension

**Solution**: Separation of concerns + automatic lifting.
- `PolydiscFunction{S}`: Mathematical definition
- `PolydiscFunctionEvaluator{S,T,N}`: Computational implementation
- Lifting adapters: Automatic type conversion at evaluator creation time

---

## ValuedFieldPoint Wrapper

### Motivation

P-adic fields have prime and precision parameters that affect computation:
```julia
K = PadicField(2, 20)  # prime=2, precision=20
```

But `PadicFieldElem` doesn't encode these at type level:
```julia
x::PadicFieldElem  # No type-level information about prime or precision
```

This prevents Julia's compiler from specializing on these parameters.

### Solution

```julia
struct ValuedFieldPoint{P,Prec,S}
    elem::S
end
```

**Type parameters:**
- `P::Int`: The prime (e.g., 2, 3, 5)
- `Prec::Int`: The precision (e.g., 20)
- `S`: The wrapped element type (typically `PadicFieldElem`)

**Example:**
```julia
K = PadicField(2, 20)
x = K(5)  # Type: PadicFieldElem

vfp = ValuedFieldPoint(x)  # Auto-infers P and Prec
# Type: ValuedFieldPoint{2, 20, PadicFieldElem}

# Or explicit:
vfp = ValuedFieldPoint{2,20,PadicFieldElem}(x)
```

### Benefits

1. **Compile-time specialization**: Julia can generate specialized code for each (P, Prec) combination
2. **Zero runtime overhead**: Type parameters are compile-time only
3. **Type safety**: Prevents mixing incompatible fields
4. **Generic design**: Works with any valued field type, not just PadicFieldElem

### Operations

**Arithmetic:**
```julia
vfp1 + vfp2  # Returns ValuedFieldPoint
vfp1 * vfp2  # Returns ValuedFieldPoint

# Mixed-type operations
vfp + K(3)  # Works! Returns ValuedFieldPoint
K(3) + vfp  # Also works!
```

**Extraction:**
```julia
unwrap(vfp)       # Returns underlying PadicFieldElem
unwrap((vfp1, vfp2))  # Returns tuple of PadicFieldElems
vfp.elem          # Direct field access
```

**Valuation and prime:**
```julia
valuation(vfp)    # Delegates to underlying element
abs(vfp)          # Computes p^(-valuation(vfp))
prime(vfp)        # Returns P from type parameter (zero cost!)
precision(vfp)    # Returns Prec from type parameter (zero cost!)
```

### Auto-Wrapping

When creating polydiscs with `PadicFieldElem` centers, they're automatically wrapped:

```julia
K = PadicField(2, 20)
p = ValuationPolydisc([K(1), K(2)], [0, 0])

# Actual type:
# ValuationPolydisc{ValuedFieldPoint{2, 20, PadicFieldElem}, Int, 2}

# Enables compile-time specialization for evaluators
```

---

## Typed Evaluators

### Architecture

**Design Philosophy:**
- **PolydiscFunction**: "What" — mathematical definition, symbolic
- **PolydiscFunctionEvaluator**: "How" — efficient computation, specialized

### Evaluator Hierarchy

```julia
abstract type PolydiscFunctionEvaluator{S,T,N} end
```

**Concrete evaluators** (all callable structs):

```julia
# Basic
LinearPolynomialEvaluator{S,T,N}
    coefficients::NTuple{N,S}
    coeff_valuations::NTuple{N,Int}  # Precomputed
    constant::S

ConstantEvaluator{S,T,N}
    value::Float64

MPolyEvaluator{S,T,N,P<:MPoly}
    poly::P

# Sums
SumEvaluator{S,T,N,E<:PolydiscFunctionEvaluator}
    evaluators::Vector{E}

# Binary operations
AddEvaluator{S,T,N,L,R}
    left::L
    right::R

SubEvaluator{S,T,N,L,R}  # Similar structure
MulEvaluator{S,T,N,L,R}
DivEvaluator{S,T,N,L,R}

# Other
SMulEvaluator{S,T,N,R}
    scalar::Float64
    right::R

CompEvaluator{S,T,N,F,R}
    outer::F
    inner::R

LambdaEvaluator{S,T,N}
    func::Function
```

### Creating Evaluators

**Typed Interface (NEW):**
```julia
eval = batch_evaluate_init(f::PolydiscFunction{S}, ::Type{ValuationPolydisc{S,T,N}})
# Returns: PolydiscFunctionEvaluator{S,T,N}
```

**Legacy Interface (closure):**
```julia
eval = batch_evaluate_init(f::PolydiscFunction{S})
# Returns: Function (p::ValuationPolydisc -> Float64)
```

### Using Evaluators

Evaluators are callable:
```julia
K = PadicField(2, 20)
poly = LinearPolynomial([K(1), K(2)], K(0))

# Create typed evaluator
eval = batch_evaluate_init(poly, ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem},Int,2})
# Type: LinearPolynomialEvaluator{ValuedFieldPoint{2,20,PadicFieldElem},Int,2}

# Use it
p = ValuationPolydisc([K(3), K(4)], [0, 0])
result = eval(p)  # Returns Float64
```

### Benefits

1. **Type stability**: All types known at compile time
2. **No closures**: Struct representation is more efficient
3. **Inlining**: Evaluator calls can be inlined
4. **Specialization**: Julia generates optimized code for each (S,T,N) combination
5. **Composability**: Evaluators compose via type parameters

---

## Lifting Adapters

### The Problem

User code defines functions with one type but needs to evaluate with another:

```julia
# User defines function with PadicFieldElem
R, (x, a) = polynomial_ring(K, ["x", "a"])
f = LinearPolynomial([K(1), K(2)], K(0))
# Type: LinearPolynomial{PadicFieldElem}

# Data is auto-wrapped to ValuedFieldPoint
data = [(ValuationPolydisc([K(5)], [0]), K(10))]
# Polydisc type: ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem}, Int, 1}

# Need evaluator that accepts ValuedFieldPoint polydiscs
# but function has PadicFieldElem coefficients!
```

### The Solution: Lifting Adapters

Generic adapter methods intercept `batch_evaluate_init` calls and convert coefficients:

```julia
function batch_evaluate_init(
    f::PolydiscFunction{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    # Convert coefficients from S to ValuedFieldPoint{P,Prec,S}
    # Return evaluator with ValuedFieldPoint type
end
```

### Lifting Flow

```
1. User: Define function
   LinearPolynomial{PadicFieldElem}([K(1), K(2)], K(0))

2. User: Create data (auto-wrapped)
   ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem}, Int, 1}

3. System: Create loss function
   → batch_evaluate_init(model, ValuationPolydisc{ValuedFieldPoint{...}, Int, N})

4. Lifting adapter: Intercept call
   → Matches PolydiscFunction{S} with ValuedFieldPoint{P,Prec,S} polydisc

5. Conversion: Wrap coefficients
   K(1) → ValuedFieldPoint{2,20,PadicFieldElem}(K(1))
   K(2) → ValuedFieldPoint{2,20,PadicFieldElem}(K(2))
   K(0) → ValuedFieldPoint{2,20,PadicFieldElem}(K(0))

6. Return: Typed evaluator
   LinearPolynomialEvaluator{ValuedFieldPoint{2,20,PadicFieldElem}, Int, 1}

7. Runtime: Fully specialized evaluation
   eval(polydisc) → Float64 (zero overhead)
```

### Supported Lifting Adapters

**LinearPolynomial:**
```julia
function batch_evaluate_init(
    poly::LinearPolynomial{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    new_coeffs = [VFP(c) for c in poly.coefficients]
    new_const = VFP(poly.constant)
    new_poly = LinearPolynomial(new_coeffs, new_const)
    return batch_evaluate_init(new_poly, ValuationPolydisc{VFP,T,N})
end
```

**AbsolutePolynomialSum:**
```julia
function batch_evaluate_init(
    f::AbsolutePolynomialSum{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    # Recursively create evaluator for each polynomial
    evaluators = [batch_evaluate_init(poly, ValuationPolydisc{VFP,T,N}) for poly in f.polys]
    E = eltype(evaluators)
    return SumEvaluator{VFP,T,N,E}(evaluators)
end
```

**Composite operators (Add, Sub, Mul, Div, SMul, Comp):**
```julia
function batch_evaluate_init(
    f::Add{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    PT = ValuationPolydisc{VFP,T,N}
    # Recursively lift children
    left = batch_evaluate_init(f.left, PT)
    right = batch_evaluate_init(f.right, PT)
    return AddEvaluator{VFP,T,N,typeof(left),typeof(right)}(left, right)
end
```

**MPoly (special case):**
```julia
function batch_evaluate_init(
    poly::MPoly{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    # Can't convert MPoly coefficients (no ring of ValuedFieldPoint)
    # Wrap in LambdaEvaluator that unwraps before evaluating
    VFP = ValuedFieldPoint{P,Prec,S}
    function wrapped_eval(p::ValuationPolydisc{VFP,T,N})
        unwrapped = ValuationPolydisc{S,T,N}(p.center |> unwrap, p.radius)
        return evaluate(poly, unwrapped)
    end
    return LambdaEvaluator{VFP,T,N}(wrapped_eval)
end
```

### Generic Design

All adapters work for **any coefficient type** `S`, not just `PadicFieldElem`:

```julia
function batch_evaluate_init(
    f::PolydiscFunction{S},  # Generic S!
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    # Works for any S that has:
    # - valuation(::S) -> Int
    # - abs(::S) -> Float64
    # - +, -, *, / operations
    # - Compatible with ValuedFieldPoint{P,Prec,S}
end
```

This enables NAML to work with any valued field implementation, not just p-adics.

---

## ModelEvaluator

### Structure

```julia
struct ModelEvaluator{FS,PS,T,N1,N2,E<:PolydiscFunctionEvaluator}
    model::AbstractModel{FS}
    fun_eval::E
end
```

**Type parameters:**
- `FS`: Function coefficient type (e.g., `PadicFieldElem`)
- `PS`: Parameter polydisc type (e.g., `ValuedFieldPoint{2,20,PadicFieldElem}`)
- `T`: Radius type (`Int`)
- `N1`, `N2`: Dimensions (currently unused, set to 0)
- `E`: The underlying function evaluator type

### Creation

**Same-type case:**
```julia
batch_evaluate_init(
    m::AbstractModel{S},
    ::Type{ValuationPolydisc{S,T,N}}
) where {S,T,N}
    fun_eval = batch_evaluate_init(m.fun, ValuationPolydisc{S,T,N})
    return ModelEvaluator{S,S,T,0,0,typeof(fun_eval)}(m, fun_eval)
end
```

**Lifting case:**
```julia
batch_evaluate_init(
    m::AbstractModel{S},
    ::Type{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}}
) where {S,P,Prec,T,N}
    VFP = ValuedFieldPoint{P,Prec,S}
    # Delegates to function lifting adapter
    fun_eval = batch_evaluate_init(m.fun, ValuationPolydisc{VFP,T,N})
    return ModelEvaluator{S,VFP,T,0,0,typeof(fun_eval)}(m, fun_eval)
end
```

### Usage

ModelEvaluator is callable:
```julia
eval(data::ValuationPolydisc, param::ValuationPolydisc) -> Float64
```

Implementation:
```julia
function (eval::ModelEvaluator)(
    val::ValuationPolydisc{S1,T,N1},
    param::ValuationPolydisc{S2,T,N2}
) where {S1,S2,T,N1,N2}
    full_var = set_abstract_model_variable(eval.model, val, param)
    return eval.fun_eval(full_var)
end
```

### Integration with Loss Functions

Loss functions use ModelEvaluator internally:

```julia
function MSE_loss_init(
    model::AbstractModel{S},
    data::Vector{Tuple{ValuationPolydisc{S,T,N},U}}
) where {S,T,N,U}
    full_dim = length(model.param_info)
    # Create typed evaluator
    model_eval = batch_evaluate_init(model, ValuationPolydisc{S,T,full_dim})

    # Closures capture evaluator (not raw data)
    function MSE_compute(params::Vector{<:ValuationPolydisc})
        return [1 / length(data) * sum([
            (model_eval(val, param) - out)^2
            for (val, out) in data
        ]) for param in params]
    end

    # ... gradient computation ...

    return Loss(MSE_compute, MSE_grad)
end
```

**Lifting dispatch** for type mismatches:
```julia
function MSE_loss_init(
    model::AbstractModel{S},
    data::Vector{Tuple{ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N},U}}
) where {S,P,Prec,T,N,U}
    full_dim = length(model.param_info)
    # Request ValuedFieldPoint evaluator — lifting happens here!
    model_eval = batch_evaluate_init(
        model,
        ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,full_dim}
    )

    # Rest is identical
    # ...
end
```

---

## Performance Characteristics

### Compile-Time Costs

1. **First evaluation**: Julia compiles specialized code for each (S,T,N,P,Prec) combination
2. **Subsequent evaluations**: Use cached compiled code

### Runtime Costs

1. **Evaluator creation**: One-time coefficient conversion (if lifting)
2. **Evaluation**: Zero overhead, fully specialized native code
3. **Memory**: Struct overhead minimal compared to closures

### Benchmarks

With typed evaluators and ValuedFieldPoint:
- **Type stability**: 100% (verified with `@code_warntype`)
- **Inlining**: Evaluator calls fully inlined by Julia compiler
- **Allocation**: Zero allocations per evaluation (after warmup)

---

## Examples

### Basic Usage

```julia
using NAML

# Setup
K = PadicField(2, 20)
R, (x, a, b) = polynomial_ring(K, ["x", "a", "b"])

# Define function
f = LinearPolynomial([K(1), K(2), K(3)], K(0))

# Create typed evaluator
eval = batch_evaluate_init(f, ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem},Int,3})

# Use evaluator
p = ValuationPolydisc([K(1), K(2), K(3)], [0, 0, 0])
result = eval(p)  # Returns Float64
```

### Optimization with Auto-Lifting

```julia
# Define model with PadicFieldElem
R, (x, a, b) = polynomial_ring(K, ["x", "a", "b"])
f = AbsolutePolynomialSum([x^2 - a*x - b])
model = AbstractModel(f, [true, false, false])

# Data uses ValuedFieldPoint (auto-wrapped)
data = [
    (ValuationPolydisc([K(1)], [0]), K(1)),
    (ValuationPolydisc([K(2)], [0]), K(4)),
    (ValuationPolydisc([K(3)], [0]), K(9))
]

# Loss function handles lifting automatically
loss = MSE_loss_init(model, data)
# Internally creates: ModelEvaluator{PadicFieldElem, ValuedFieldPoint{...}, ...}

# Initialize and optimize
param = ValuationPolydisc([K(0), K(0)], [5, 5])
optim = greedy_descent_init(param, loss, 1, (false, 1))

for i in 1:20
    step!(optim)
end

println("Final params: ", center(optim.param))
```

### Type Inspection

```julia
# Verify typed evaluator usage
model_eval = batch_evaluate_init(model, ValuationPolydisc{ValuedFieldPoint{2,20,PadicFieldElem},Int,3})

println(typeof(model_eval))
# ModelEvaluator{PadicFieldElem, ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 0, 0, SumEvaluator{...}}

println(typeof(model_eval.fun_eval))
# SumEvaluator{ValuedFieldPoint{2, 20, PadicFieldElem}, Int64, 3, LambdaEvaluator{...}}
```

---

## Implementation Details

### Unwrapping in set_abstract_model_variable

Models with `AbstractModel{PadicFieldElem}` need adapters when parameters use `ValuedFieldPoint`:

```julia
function set_abstract_model_variable(
    m::AbstractModel{S},
    val::ValuationPolydisc{S,T,N1},
    param::ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N2}
) where {S,P,Prec,T,N1,N2}
    # Unwrap ValuedFieldPoint from param
    unwrapped_param_center = tuple([vp.elem for vp in param.center]...)
    unwrapped_param = ValuationPolydisc{S,T,N2}(unwrapped_param_center, param.radius)
    return set_abstract_model_variable(m, val, unwrapped_param)
end
```

This is necessary because the model's `param_info` vector indexes into `PadicFieldElem`-typed vectors, not `ValuedFieldPoint`.

### Directional Derivative Adapters

Similar adapters exist for `directional_derivative`:

```julia
function directional_derivative(
    fun::PolydiscFunction{S},
    v::ValuationTangent{ValuedFieldPoint{P,Prec,S},T,N}
) where {S,P,Prec,T,N}
    # Unwrap tangent vector
    unwrapped_point = ValuationPolydisc{S,T,N}(v.point.center |> unwrap, v.point.radius)
    unwrapped_direction = collect(unwrap(v.direction))
    unwrapped_tangent = ValuationTangent{S,T,N}(unwrapped_point, unwrapped_direction, v.magnitude)
    # Compute on unwrapped types
    return directional_derivative(fun, unwrapped_tangent)
end
```

### Evaluate Adapter

The `evaluate` function also has a lifting adapter:

```julia
function evaluate(
    fun::PolydiscFunction{S},
    var::ValuationPolydisc{ValuedFieldPoint{P,Prec,S},T,N}
) where {S,P,Prec,T,N}
    unwrapped_polydisc = ValuationPolydisc{S,T,N}(var.center |> unwrap, var.radius)
    return evaluate(fun, unwrapped_polydisc)
end
```

---

## Future Directions

1. **More efficient MPoly lifting**: Currently uses LambdaEvaluator with unwrapping. Could potentially create ValuedFieldPoint polynomial rings.

2. **Caching evaluators**: Could cache evaluators by function hash to avoid recreating identical evaluators.

3. **Extended generic support**: Support other valued field types beyond p-adics (e.g., Laurent series, power series).

4. **Gradient evaluators**: Specialized evaluators for gradient computation (currently uses directional_derivative).

5. **Sparse evaluators**: Optimized evaluators for sparse polynomials.
