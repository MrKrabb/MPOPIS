
function action_space_size(act_space::ClosedInterval)
    return length(leftendpoint(act_space))
end
function action_space_size(act_space::Base.OneTo)
    return 1
end

function block_diagm(A::Vector{Float64}, rep_number::Int)
    return diagm(repeat(A, rep_number))
end

function block_diagm(A::Matrix{Float64}, rep_number::Int)
    r = size(A)[1]
    rm1 = r - 1
    B = zeros(Float64, r * rep_number, r * rep_number)
    for ii in 1:r:(r*rep_number)
        B[ii:ii+rm1, ii:ii+rm1] = A
    end
    return B
end

function get_model_controls(action_space::ClosedInterval, V::Vector{Float64})
    return get_model_controls(action_space, V, 1)
end
function get_model_controls(action_space::Base.OneTo, V::Vector{Float64})
    return get_model_controls(action_space, V, 1)
end

function get_model_controls(action_space::ClosedInterval, V::Matrix{Float64})
    min_controls = leftendpoint(action_space)
    max_controls = rightendpoint(action_space)
    control_mat = zeros(size(V))
    for rᵢ ∈ 1:size(V, 2)
        control_mat[:, rᵢ] = clamp.(V[:, rᵢ], min_controls[rᵢ], max_controls[rᵢ])
    end
    return control_mat
end

function get_model_controls(action_space::ClosedInterval, V::Matrix{Float64}, horizon::Int)
    as = action_space_size(action_space)
    min_controls = leftendpoint(action_space)
    max_controls = rightendpoint(action_space)

    control_mat = reshape(V, size(V, 1), as, horizon)

    for t in 1:horizon
        for rᵢ ∈ 1:as
            control_mat[:, rᵢ, t] = clamp.(control_mat[:, rᵢ, t], min_controls[rᵢ], max_controls[rᵢ])
        end
    end
    return control_mat
end

function get_model_controls(action_space::ClosedInterval, V::Vector{Float64}, horizon::Int)
    as = action_space_size(action_space)
    min_controls = leftendpoint(action_space)
    max_controls = rightendpoint(action_space)
    control_mat = reshape(V, as, horizon)
    for rᵢ ∈ 1:size(control_mat)[1]
        control_mat[rᵢ, :] = clamp.(control_mat[rᵢ, :], min_controls[rᵢ], max_controls[rᵢ])
    end
    if as == 1
        control_mat = vec(control_mat)
    end
    return control_mat
end
function get_model_controls(action_space::Base.OneTo, V::Vector{Float64}, horizon::Int)
    as = 1
    min_controls = minimum(action_space)
    max_controls = maximum(action_space)

    control_mat = reshape(V, as, horizon)
    control_mat[1, :] = clamp.(control_mat[1, :], min_controls, max_controls)
    control_mat = round.(Int, vec(control_mat))
    return control_mat
end

function compute_weights(weight_method::Information_Theoretic, costs::Vector{Float64})
    λ = weight_method.λ
    ρ = minimum(costs)
    normalized_costs = -1 / λ * (costs .- ρ)
    weights = exp.(normalized_costs)
    η = sum(weights)
    return weights ./ η
end

function get_controls_roll_U!(pol::AbstractPathIntegralPolicy, weighted_controls::Vector)
    as = pol.params.as
    # Get control (action set for the first time step)
    control = get_model_controls(action_space(pol.env), weighted_controls[1:as])

    # Roll the control policy so next interation we start with a mean of pol.U
    if pol.params.horizon > 1
        pol.U[1:(end-as)] = weighted_controls[(as+1):end]
        pol.U[(end-as):end] = pol.params.U₀[(end-as):end]
    else
        pol.U = weighted_controls
    end
    return control
end

function rollout_model(env::EnvpoolEnv, T::Int, model_controls::Array,
    pol::AbstractPathIntegralPolicy)

    as = pol.params.as
    K = pol.params.num_samples

    traj_cost = zeros(K)
    for t ∈ 1:T
        env(model_controls[:, :, t])
        traj_cost -= reward(env) # Subtracting based on "reward"
        if pol.params.log
            for k ∈ K
                pol.logger.trajectories[k][t, :] = env.state[k, :]
            end
        end
    end
    reset!(env; restore=true)
    return traj_cost
end

function rollout_model(env::AbstractEnv, T::Int, model_controls::Vector,
    pol::AbstractPathIntegralPolicy, k::Int)
    model_controls_mat = reshape(model_controls, size(model_controls, 1), 1)
    rollout_model(env, T, model_controls_mat, pol, k)
end

function rollout_model(env::AbstractEnv, T::Int, model_controls::Matrix,
    pol::AbstractPathIntegralPolicy, k::Int,
)
    as = pol.params.as
    K = pol.params.num_samples
    traj_cost = 0.0
    for t ∈ 1:T
        controls = as == 1 ? model_controls[t] : model_controls[:, t]
        env(controls)
        traj_cost -= reward(env) # Subtracting based on "reward"
        if pol.params.log
            pol.logger.trajectories[k][t, :] = env.state
        end
    end
    return traj_cost
end

"""
    hankel_blocks(data, L)

Build block Hankel of depth L from data (d × N).
Rows stacked: [x_k; x_{k+1}; … ; x_{k+L-1}] per column.
"""
function hankel_blocks(data::AbstractMatrix{T}, L::Int) where T
    d, N = size(data)
    N ≥ L || error("Need N ≥ L (got N=$N, L=$L)")
    cols = N - L + 1
    H = Matrix{T}(undef, d * L, cols)
    @inbounds for j in 1:cols
        # window j … j+L-1
        w = data[:, j:j+L-1]              # d × L
        # reshape into (d*L) vector (column-major already groups by time after permute)
        H[:, j] = reshape(permutedims(w, (2,1)), d*L)
    end
    return H
end

"""
    build_deepc_blocks(u_hist, y_hist; T_ini, N_pred)

Return (U_p,U_f,Y_p,Y_f) wrapped in struct for DeePC.
"""
struct DeePCData{T}
    U_p::Matrix{T}
    U_f::Matrix{T}
    Y_p::Matrix{T}
    Y_f::Matrix{T}
    T_ini::Int
    N_pred::Int
end

function build_deepc_blocks(u_hist::AbstractMatrix, y_hist::AbstractMatrix; T_ini::Int, N_pred::Int)
    @assert size(u_hist, 2) == size(y_hist, 2) "u/y length mismatch"
    L = T_ini + N_pred
    Hu = hankel_blocks(u_hist, L)
    Hy = hankel_blocks(y_hist, L)
    m = size(u_hist, 1); p = size(y_hist, 1)
    U_p = Hu[1:(m*T_ini), :]
    U_f = Hu[(m*T_ini+1):(m*L), :]
    Y_p = Hy[1:(p*T_ini), :]
    Y_f = Hy[(p*T_ini+1):(p*L), :]
    return DeePCData(U_p, U_f, Y_p, Y_f, T_ini, N_pred)
end