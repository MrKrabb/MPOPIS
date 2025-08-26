# Example MPPI–DeePC Update in Julia
#
# This script demonstrates how to use a sampling–based model predictive
# path integral (MPPI) update to find an optimal coefficient vector `g`
# for a Hankel–matrix‐based predictive model, analogous to the Python
# example provided earlier.  The Hankel matrices `U_f` and `Y_f` are
# generated randomly for demonstration purposes.  In a data–enabled
# predictive control (DeePC) setting, these matrices are constructed
# from input/output data and map a coefficient vector `g` to the
# predicted future inputs and outputs.  The algorithm samples random
# perturbations around a nominal `g`, evaluates the resulting
# predicted trajectories using a quadratic cost function, converts
# the costs into exponential weights (as in MPPI), and updates the
# nominal `g` with a weighted average of the perturbations.

using LinearAlgebra, Random

"""
    cost_function(y::Vector{Float64}, u::Vector{Float64}; Q = nothing, R = nothing, y_ref = nothing)

Compute a quadratic cost for a given predicted output trajectory `y` and
control trajectory `u`.  The reference trajectory `y_ref` defaults to
zero, and the weighting matrices `Q` and `R` default to identity
matrices of appropriate dimension.  The returned value is a scalar.
"""
function cost_function(y::Vector{Float64}, u::Vector{Float64}; Q = nothing, R = nothing, y_ref = nothing)
    ny = length(y)
    nu = length(u)
    # Default reference is zero
    y_ref === nothing && (y_ref = zeros(ny))
    Q === nothing && (Q = I(ny))
    R === nothing && (R = I(nu))
    err = y .- y_ref
    # Quadratic cost: err' * Q * err + u' * R * u
    return dot(err, Q * err) + dot(u, R * u)
end

# Set random seed for reproducibility
Random.seed!(0)

# Define dimensions
T = 5        # prediction horizon
m = 1        # input dimension
p = 1        # output dimension
M = 10       # number of columns in Hankel matrices (size of g)

# Random Hankel matrices for demonstration.  In practice these come
# from measured input/output data.
U_f = randn(m*T, M)
Y_f = randn(p*T, M)

# MPPI parameters
num_samples = 50      # number of sampled perturbations
lambda_param = 1.0    # temperature parameter λ
sigma = 0.5           # standard deviation for Gaussian perturbations
iterations = 3        # number of MPPI iterations

# Initialize nominal g (can be zeros or previous best estimate)
g_nominal = zeros(M)

for it in 1:iterations
    # Preallocate arrays for costs and perturbations
    costs = zeros(num_samples)
    noises = sigma .* randn(num_samples, M)
    # Evaluate each sampled perturbation
    for i in 1:num_samples
        g_candidate = g_nominal .+ view(noises, i, :)
        u_future = U_f * g_candidate
        y_future = Y_f * g_candidate
        costs[i] = cost_function(y_future, u_future)
    end
    # Compute MPPI weights
    ρ = minimum(costs)
    weights = exp.(- (costs .- ρ) ./ lambda_param)
    weights ./= sum(weights)
    # Update g_nominal using weighted sum of perturbations
    for j in 1:M
        g_nominal[j] += sum(weights .* noises[:, j])
    end
end

println("Updated g_nominal:", g_nominal)