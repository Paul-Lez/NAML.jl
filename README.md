# Non-Archimedean Machine Learning 

This repository implements some algorithms for machine learning with inputs 
and parameters in some non-Archimedean field (or more generally in some 
polydisc space over a non-Archimedean field). 

## Content

- Basic structures (polydisc, tangent vectors, absolute polynomials) and API are implemented in the files in folder `src/basic`
- The folder `src/optim` contains the infrastructure for training non-Archimedean models. 
    - `basic.jl` develops some objects and API for setting up and training models,
    - `loss.jl` implements several "standard" loss functions,
    - The folder `greedy_descent.jl` implements a "greedy" descent algorithm,
    - The folder `gradient_descent.jl` implements a version of gradient descent.

## See how this works

To see this in practice, run the demo in `test/cubic_learning_experiment.ipynb`. 