
mutable struct MPPI_Logger
    trajectories::Vector{Matrix{Float64}}
    traj_costs::Vector{Float64}
    traj_weights::Vector{Float64}
    U_hankel::Union{Matrix{Float64},Nothing}
    Y_hankel::Union{Matrix{Float64},Nothing}
    sample_controls::Union{Vector{Matrix{Float64}},Nothing}  # per-sample control sequences (T×as)
end

# File-wide dependencies for Hankel/G I/O
using DelimitedFiles

# Convenience constructor preserving old 3-arg usage
function MPPI_Logger(trajectories::Vector{Matrix{Float64}}, traj_costs::Vector{Float64}, traj_weights::Vector{Float64})
    MPPI_Logger(trajectories, traj_costs, traj_weights, nothing, nothing, nothing)
end

struct MPPI_Policy_Params{M<:AbstractWeightMethod}
    num_samples::Int
    horizon::Int
    λ::Float64
    α::Float64
    U₀::Vector{Float64}
    ss::Int
    as::Int
    cs::Int
    weight_method::M
    log::Bool
    # Option D (future prediction shaping) parameters
    pred_future_weight::Float64      # weight w_fut (>=0)
    pred_future_power::Float64       # exponent p (e.g., 1.0 for |d|, 2.0 for d^2)
    pred_future_use_mean::Bool       # use mean over horizon (true) or sum (false)
    pred_future_cap::Float64         # optional cap on per-step dist before power (Inf to disable)
end

"""
MPPI_Policy_Params(env::AbstractEnv, type::Symbol; kwargs...)
    Construct the mppi policy parameter struct
kwargs:
    - num_samples::Int = 50,
    - horizon::Int = 50,
    - λ::Float64 = 1.0,
    - α::Float64 = 1.0,
    - U₀::Vector{Float64} = [0.0],
    - cov_mat::Union{Matrix{Float64},Vector{Float64}} = [1.0],
    - weight_method::Symbol = :IT,
    - elite_threshold::Float64 = 0.8,
    - rng::AbstractRNG = Random.GLOBAL_RNG,
    - log::Bool = false,
"""
function MPPI_Policy_Params(env::AbstractEnv, type::Symbol;
    num_samples::Int=50,
    horizon::Int=50,
    λ::Float64=60.0,        # Temperature for the controller, tune for optimization
    α::Float64=1.0,
    U₀::Vector{Float64}=[0.0],
    cov_mat::Union{Matrix{Float64},Vector{Float64}}=[1.0],
    weight_method::Symbol=:IT,
    elite_threshold::Float64=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    log::Bool=false,
    # Option D (future prediction shaping) kwargs
    pred_future_weight::Float64=0.2,
    pred_future_power::Float64=2.0,
    pred_future_use_mean::Bool=true,
    pred_future_cap::Float64=Inf,
)

    # State space size
    if isa(state(env), Vector)
        ss = length(state(env))
    elseif isa(state(env), Matrix)
        ss = size(state(env), 2)
    else
        error("State must be Vector or Matrix")
    end

    as = action_space_size(action_space(env)) # Action space size
    cs = as * horizon                         # Control size (number of actions per sample)

    if length(U₀) == as
        U₀ = repeat(U₀, horizon)
    end
    length(U₀) == cs || error("U₀ must be length of action space or control space")

    if type == :mppi
        repeat_num = 1
        check_size = as
    elseif type == :gmppi
        repeat_num = horizon
        check_size = cs
    else
        error("Incorrect type for MPPPI")
    end

    if size(cov_mat)[1] == as
        cov_mat = block_diagm(cov_mat, repeat_num)
    end
    size(cov_mat)[1] == check_size || error("Covariance matrix size problem")
    size(cov_mat)[1] == size(cov_mat)[2] || error("Covriance must be square")
    Σ = cov_mat

    if weight_method == :IT
        weight_m = Information_Theoretic(λ)
    elseif cost_method == :CE
        n = round(Int, num_samples * (1 - elite_threshold))
        weight_m = Cross_Entropy(elite_threshold, n)
    else
        error("No cost method implemented for $weight_method")
    end

    log_traj = [Matrix{Float64}(undef, (horizon, ss)) for _ in 1:num_samples]
    log_controls = [Matrix{Float64}(undef, (horizon, as)) for _ in 1:num_samples]
    log_traj_costs = Vector{Float64}(undef, num_samples)
    log_traj_weights = Vector{Float64}(undef, num_samples)
    mppi_logger = MPPI_Logger(log_traj, log_traj_costs, log_traj_weights)
    mppi_logger.sample_controls = log_controls
    # Ensure accumulation buffers are initialized (now handled in struct)

    params = MPPI_Policy_Params(
    num_samples, horizon, λ, α, U₀, ss, as, cs,
    weight_m, log,
    pred_future_weight, pred_future_power, pred_future_use_mean, pred_future_cap
    )
    return params, U₀, Σ, rng, mppi_logger
end

#######################################
# MPPI
#######################################
mutable struct MPPI_Policy{R<:AbstractRNG} <: AbstractPathIntegralPolicy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
    # Last predicted (x,y) per horizon step and per g-sample: size (T, 2, K)
    last_predicted_positions::Union{Array{Float64,3},Nothing}
end

function MPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :mppi; kwargs...)
    return MPPI_Policy(params, env, U₀, Σ, rng, mppi_logger, nothing)
end

function (pol::MPPI_Policy)(env::AbstractEnv)
    # === STEP 0: Unpack core dimensions ===
    # K: number of sampled trajectories this iteration
    # T: planning horizon (steps into the future simulated)
    # as: action dimension (per single time step)
    # cs: total control vector length (as * T)
    K, T = pol.params.num_samples, pol.params.horizon
    as, cs = pol.params.as, pol.params.cs

    # === STEP 1: Sample trajectory noises & forward-simulate to accumulate costs ===
    # Returns trajectory_cost (length K), E (exploration noises), G (Hankel input combos),
    # and Ypos_pred (T×2×K predicted positions from Hankel output combos)
    trajectory_cost, E, G, Ypos_pred = calculate_trajectory_costs(pol, env)

    # === STEP 2: Convert costs into importance weights ===
    # Lower cost => higher weight (Information-Theoretic or other method)
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    weights = reshape(weights, K, 1)  # Ensure weights are column vector

    # === STEP 3: Form weighted control adjustment using Hankel-based g (adaptation kept) ===
    weighted_noise = zeros(Float64, cs)
    for t ∈ 1:T
        tspan = ((t-1)*as+1):(t*as)
        for k ∈ 1:K
            g_tk = G[tspan, k]
            weighted_noise[tspan] += weights[k] .* g_tk
        end
    end

    # === STEP 4: Update nominal control sequence (path integral shift) ===
    # (Using g* now)
    weighted_controls = pol.U + weighted_noise
    # === STEP 5: Receding horizon: extract first action & roll remaining sequence ===
    control = get_controls_roll_U!(pol, weighted_controls)

    # === STEP 6: (Optional) Log sample diagnostics ===
    if pol.params.log
        pol.logger.traj_costs = trajectory_cost
        pol.logger.traj_weights = vec(weights)
    end

    # === STEP 7: Return single-step control to environment ===
    pol.last_predicted_positions = Ypos_pred
    return control
end

# Preserve Envpool path behavior and signatures (E-based aggregation)
function (pol::MPPI_Policy)(env::EnvpoolEnv)
    # === STEP 0: Unpack core dimensions ===
    K, T = pol.params.num_samples, pol.params.horizon
    as, cs = pol.params.as, pol.params.cs

    # === STEP 1: Sample noises & rollout (Envpool specialized cost fn) ===
    trajectory_cost, E = calculate_trajectory_costs(pol, env)

    # === STEP 2: Weights ===
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    weights = reshape(weights, K, 1)

    # === STEP 3: U* from E (original behavior) ===
    weighted_noise = zeros(Float64, cs)
    for t ∈ 1:T
        for k ∈ 1:K
            weighted_noise[((t-1)*as+1):(t*as)] += weights[k] .* E[k, t]
        end
    end

    # === STEP 4: Update + roll ===
    weighted_controls = pol.U + weighted_noise
    control = get_controls_roll_U!(pol, weighted_controls)

    if pol.params.log
        pol.logger.traj_costs = trajectory_cost
        pol.logger.traj_weights = vec(weights)
    end
    return control
end

function calculate_trajectory_costs(pol::MPPI_Policy, env::EnvpoolEnv)
    # === STEP 1A (called by STEP 1 above): Sample noises & rollout simulated trajectories ===
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ * (1 - pol.params.α)

    # Draw exploration noise for each (trajectory k, time t)
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K, T)
    # Increase diversity for Hankel data: scale exploration noise by temperature τ
    τ = 2.5  # temperature factor (>1 increases noise magnitude)
    for t ∈ 1:T
        for k ∈ 1:K
            E[k, t] .*= τ
        end
    end
    Σ_inv = Distributions.invcov(P)

    trajectory_cost = zeros(Float64, K)

    for t ∈ 1:T  # advance simulated time index
        Eₜ = E[:, t]
        # Eᵢ = E[k,t]
        uₜ = pol.U[((t-1)*as+1):(t*as)]

        Vₜ = repeat(uₜ', K) + reduce(hcat, E[:, t])'

        # Original per-sample comprehension (scalar per k):
        # control_costs = [γ * uₜ' * Σ_inv * Eᵢ for Eᵢ in Eₜ]
        # Vectorized equivalent using the already-built Vₜ (K×as):
        # qₜ = (Σ_inv') * uₜ  # as-vector
        # control_costs = γ .* ((Vₜ .- repeat(uₜ', K)) * qₜ)
        qₜ = (Σ_inv') * uₜ
        control_costs = γ .* ((Vₜ .- repeat(uₜ', K)) * qₜ)

        model_controls = get_model_controls(action_space(env), Vₜ)

        env(model_controls)

        # Accumulate running cost: negative reward + control shaping term
        trajectory_cost = trajectory_cost - reward(env) + control_costs
        if pol.params.log
            for k ∈ 1:K
                pol.logger.trajectories[k][t, :] = state(env)[k, :]
            end
        end
    end
    reset!(env; restore=true)

    return trajectory_cost, E
end

function calculate_trajectory_costs(pol::MPPI_Policy, env::AbstractEnv)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ * (1 - pol.params.α)

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K, T)
    Σ_inv = Distributions.invcov(P)

    # Precompute qₜ = (Σ_inv') * uₜ once per timestep (uₜ is shared across all samples k)
    # (Kept for reference, but control_cost will switch to g-based)
    q_per_t = Vector{Vector{Float64}}(undef, T)
    for t ∈ 1:T
        uₜ = pol.U[((t-1)*as+1):(t*as)]
        q_per_t[t] = (Σ_inv') * uₜ
    end

    # Storage to build Hankel data from rollouts (inputs and outputs)
    # We log per-sample, per-time controls and states already in logger when pol.params.log
    # Here we will always capture minimal U/Y for Hankel regardless of log flag.
    U_roll = [Matrix{Float64}(undef, T, as) for _ in 1:K]
    Y_roll = [Matrix{Float64}(undef, T, pol.params.ss) for _ in 1:K]

    trajectory_cost = zeros(Float64, K)
    # Original model-based rollout to compute reward contribution
    # Threads.@threads for k ∈ 1:K
    for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        for t ∈ 1:T
            Eᵢ = E[k, t]
            uₜ = pol.U[((t-1)*as+1):(t*as)]
            Vₜ = uₜ + Eᵢ
            model_controls = get_model_controls(action_space(sim_env), Vₜ)
            sim_env(model_controls)
            # Log for Hankel
            @inbounds U_roll[k][t, :] = Vₜ
            @inbounds Y_roll[k][t, :] = state(sim_env)
            # Reward contribution (negative for cost)
            trajectory_cost[k] = trajectory_cost[k] - reward(sim_env)
            if pol.params.log
                pol.logger.trajectories[k][t, :] = state(sim_env)
                pol.logger.sample_controls[k][t, :] = Vₜ
            end
        end
    end

    # === Build Hankel matrices (U_f, Y_f) per DeePC “shallows” ===
    # Adopt T_ini = 0 and prediction horizon N_pred = T, so the future block is the
    # whole rollout window for constructing combinations.
    # Stack all rollouts vertically across samples to get a single data matrix for Hankel.
    # IMPORTANT: only record the steering angle as input (first action dim); ignore pedal.
    Ucat_full = reduce(vcat, [U_roll[k] for k in 1:K])'  # shape: as × (K*T)
    Ucat = @view Ucat_full[1:1, :]                         # shape: 1 × (K*T) (steering only)
    Ycat = reduce(vcat, [Y_roll[k] for k in 1:K])'        # shape: ss × (K*T)

    T_ini = 0; N_pred = T
    # With T_ini=0, the Hankel future block from hankel_blocks is essentially windowed columns.
    Uf = hankel_blocks(Ucat, max(1, N_pred))          # (1*N_pred) × (K*T - N_pred + 1)
    Yf = hankel_blocks(Ycat, max(1, N_pred))          # (ss*N_pred) × (K*T - N_pred + 1)

    # Save Hankel blocks for inspection (optional best-effort)
    try
        mkpath(joinpath(dirname(@__FILE__), "..", "hankel_data"))
        writedlm(joinpath(dirname(@__FILE__), "..", "hankel_data", "U_f_rollout.csv"), Uf, ',')
        writedlm(joinpath(dirname(@__FILE__), "..", "hankel_data", "Y_f_rollout.csv"), Yf, ',')
    catch err
        Base.@warn "Failed to save Hankel data" err
    end

    # === Generate G: random linear combinations of Hankel columns (inputs+outputs) ===
    # Let H_full = [Uf; Yf]; sample random a_k and form ĝ_k = H_full * a_k.
    # Then extract the input (Uf) segment as g_k for cost shaping.
    H_full = vcat(Uf, Yf)
    cols = size(H_full, 2)
    cs = pol.params.cs                 # as*T (full action dim for downstream update)
    in_len = size(Uf, 1)               # input segment length in H_full (here T since 1×T)
    G = Matrix{Float64}(undef, cs, K)
    # Store predicted positions (x,y) over horizon per sample: (T, 2, K)
    Ypos_pred = Array{Float64}(undef, T, 2, K)
    for k ∈ 1:K
        a_k = randn(pol.rng, cols)
        g_hat = H_full * a_k  # length in_len + ss*N_pred
        # Input part (steering only): length = in_len (= T)
        g_steer = g_hat[1:in_len]
        # Expand to full action-dim per timestep vector with zeros for non-steering controls
        g_full = zeros(cs)
        @inbounds for t ∈ 1:T
            tspan_start = (t-1)*as + 1
            g_full[tspan_start] = g_steer[t]   # steering is first action component
            # remaining action components (e.g., pedal) stay zero
        end
        G[:, k] = g_full

        # Extract output segment and reshape to (ss, T) to get (x,y) per step
        y_hat = g_hat[(in_len+1):end]
        Y_hat_mat = reshape(y_hat, pol.params.ss, T)  # (ss, T)
        # positions are state indices 1:2 => shape to (T, 2)
        Ypos_pred[:, :, k] = permutedims(Y_hat_mat[1:2, :], (2,1))
    end

    # Persist G for debugging
    try
        writedlm(joinpath(dirname(@__FILE__), "..", "hankel_data", "H_full_rollout.csv"), H_full, ',')
        writedlm(joinpath(dirname(@__FILE__), "..", "hankel_data", "G_rollout.csv"), G, ',')
    catch err
        Base.@warn "Failed to save G data" err
    end

    # Add g-based control shaping cost to be consistent with g aggregation
    for k ∈ 1:K
        acc = 0.0
        for t ∈ 1:T
            tspan = ((t-1)*as+1):(t*as)
            qₜ = q_per_t[t]
            g_tk = G[tspan, k]
            acc += dot(g_tk, qₜ)
        end
        trajectory_cost[k] += γ * acc
    end

    # Option D: Future-shaping using predicted centerline distances over the horizon
    # Add cost = w_fut * aggregator_t ( clamp(dist,cap)^p )
    w_fut = pol.params.pred_future_weight
    if w_fut > 0
        p_pow = pol.params.pred_future_power
        use_mean = pol.params.pred_future_use_mean
        cap = pol.params.pred_future_cap
        for k ∈ 1:K
            acc = 0.0
            for t ∈ 1:T
                pos_t = Ypos_pred[t, :, k]  # (x,y)
                wt = within_track(env.track, pos_t)  # returns .dist
                d = wt.dist
                if isfinite(cap)
                    d = clamp(d, 0.0, cap)
                end
                acc += d^p_pow
            end
            if use_mean
                acc /= T
            end
            trajectory_cost[k] += w_fut * acc
        end
    end

    return trajectory_cost, E, G, Ypos_pred
end

#######################################
# GMPPI Policies
#######################################
function (pol::AbstractGMPPI_Policy)(env::AbstractEnv)
    cs = pol.params.cs
    trajectory_cost, E, weights = calculate_trajectory_costs(pol, env)

    # Weight the noise based on the calcualted weights
    weighted_noise = zeros(Float64, cs)
    for rᵢ ∈ 1:cs
        weighted_noise[rᵢ] = weights' * E[rᵢ, :]
    end
    weighted_controls = pol.U + weighted_noise
    control = get_controls_roll_U!(pol, weighted_controls)

    if pol.params.log
        pol.logger.traj_costs = trajectory_cost
        pol.logger.traj_weights = weights
    end
    return control
end

function simulate_model(pol::AbstractGMPPI_Policy, env::EnvpoolEnv,
    E::Matrix{Float64}, Σ_inv::Matrix{Float64}, U_orig::Vector{Float64},
)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ * (1 - pol.params.α)

    trajectory_cost = zeros(Float64, K)

    control_costs = [γ * U_orig' * Σ_inv * (pol.U + E[:, k] .- U_orig) for k in 1:K]

    # Vₖ = pol.U + E[:,k]
    Vₖ = repeat(pol.U', K) + E'

    model_controls = get_model_controls(action_space(env), Vₖ, T)
    trajectory_cost = rollout_model(env, T, model_controls, pol)
    trajectory_cost += control_costs  # Adding based on "cost"

    return trajectory_cost
end

function simulate_model(pol::AbstractGMPPI_Policy, env::AbstractEnv,
    E::Matrix{Float64}, Σ_inv::Matrix{Float64}, U_orig::Vector{Float64},
)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ * (1 - pol.params.α)

    trajectory_cost = zeros(Float64, K)
    Threads.@threads for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        Vₖ = pol.U + E[:, k]
        control_cost = γ * U_orig' * Σ_inv * (Vₖ .- U_orig)
        model_controls = get_model_controls(action_space(sim_env), Vₖ, T)
        trajectory_cost[k] = rollout_model(sim_env, T, model_controls, pol, k)
        trajectory_cost[k] += control_cost  # Adding based on "cost"
    end
    return trajectory_cost
end

"""
GMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Generalized MPPI policy strcut
"""
mutable struct GMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
end

"""
GMPPI_Policy(env::AbstractEnv; kwargs...)
    - env::AbstractEnv
kwargs are passed to MPPI_Policy_params
"""
function GMPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return GMPPI_Policy(params, env, U₀, Σ, rng, mppi_logger)
end

function calculate_trajectory_costs(pol::GMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K)
    Σ_inv = Distributions.invcov(P)

    # Use the samples to simulate our model to get the costs
    trajectory_cost = simulate_model(pol, env, E, Σ_inv, pol.U)
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
IMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Iterative version of MPPI. Same as MPPI, but multple iterations. Updating the mean only
"""
mutable struct IMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    rng::R
    logger::MPPI_Logger
end

"""
IMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function IMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    pol = IMPPI_Policy(params, env, U₀, Σ, opt_its, rng, mppi_logger)
    return pol
end


function calculate_trajectory_costs(pol::IMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its

    U_orig = pol.U
    P = Distributions.MvNormal(pol.Σ)
    Σ_inv = Distributions.invcov(P)

    trajectory_cost = Vector{Float64}(undef, K)
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        E = rand(pol.rng, P, K)
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
            pw = StatsBase.ProbabilityWeights(ws)
            (μ′, Σ′) = StatsBase.mean_and_cov(E, pw, 2)
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
CEMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Cross-Entropy version of MPOPI
"""
mutable struct CEMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    ce_elite_threshold::Float64
    Σ_estimation_method::Union{LinearShrinkage,SimpleCovariance}
    rng::R
    logger::MPPI_Logger
end

"""
CEMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    ce_elite_threshold::Float64 = 0.8,
    Σ_est::Symbol = :mle,
    kwargs...
kwargs passed to MPPI_Policy_Params

Options for Σ_est
    - :mle = maximum liklihood estimation
    - :lw = Lediot & Wolf (http://www.ledoit.net/honey.pdf)
    - :ss = Schaffer & Strimmer (https://strimmerlab.github.io/)
    - :rblw = Rao-Blackwell estimator (https://arxiv.org/pdf/0907.4698.pdf)
    - :oas = Oracle-Approximating (https://arxiv.org/pdf/0907.4698.pdf)
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/lshrink/
"""
function CEMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    ce_elite_threshold::Float64=0.8,
    Σ_est::Symbol=:mle,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    if Σ_est == :mle
        Σ_est_method = SimpleCovariance()
    elseif Σ_est == :lw
        Σ_est_method = LinearShrinkage(DiagonalUnequalVariance(), :lw)
    elseif Σ_est == :ss
        Σ_est_method = LinearShrinkage(DiagonalUnequalVariance(), :ss)
    elseif Σ_est == :rblw
        Σ_est_method = LinearShrinkage(DiagonalCommonVariance(), :rblw)
    elseif Σ_est == :oas
        Σ_est_method = LinearShrinkage(DiagonalCommonVariance(), :oas)
    else
        error("CEMPPI_Policy - Not a valid Σ estimation method")
    end
    pol = CEMPPI_Policy(params, env, U₀,
        Σ, opt_its, ce_elite_threshold,
        Σ_est_method, rng, mppi_logger,
    )
    return pol
end

function calculate_trajectory_costs(pol::CEMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    m_elite = round(Int, K * (1 - pol.ce_elite_threshold))

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            # Select elite samples, fit new distribution
            order = sortperm(trajectory_cost)
            elite = E[:, order[1:m_elite]]

            elite_traj_cost = trajectory_cost[order[1:m_elite]]
            if maximum(abs.(diff(elite_traj_cost, dims=1))) < 10e-3
                break
            end

            # Transposing elite based on needed format (n x p)
            Σ′ = cov(pol.Σ_estimation_method, elite') + 10e-9 * I
            pol.U = pol.U + vec(mean(elite, dims=2))
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
CMAMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Covariance Matrix Adaptation version of MPOPI
"""
mutable struct CMAMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    σ::Float64
    m_elite::Int
    ws::Vector{Float64}
    μ_eff::Float64
    cσ::Float64
    dσ::Float64
    cΣ::Float64
    c1::Float64
    cμ::Float64
    E::Float64
    rng::R
    logger::MPPI_Logger
end

"""
CMAMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    σ::Float64 = 1.0,
    elite_perc_threshold::Float64 = 0.8,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function CMAMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    σ::Float64=1.0,
    elite_perc_threshold::Float64=0.8,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    m = params.num_samples
    n = params.cs
    m_elite = round(Int, (1.0 - elite_perc_threshold) * m)
    ws = log((m + 1) / 2) .- log.(1:m)
    ws[1:m_elite] ./= sum(ws[1:m_elite])
    μ_eff = 1 / sum(ws[1:m_elite] .^ 2)
    cσ = (μ_eff + 2) / (n + μ_eff + 5)
    dσ = 1 + 2max(0, sqrt((μ_eff - 1) / (n + 1)) - 1) + cσ
    cΣ = (4 + μ_eff / n) / (n + 4 + 2μ_eff / n)
    c1 = 2 / ((n + 1.3)^2 + μ_eff)
    cμ = min(1 - c1, 2 * (μ_eff - 2 + 1 / μ_eff) / ((n + 2)^2 + μ_eff))
    ws[m_elite+1:end] .*= -(1 + c1 / cμ) / sum(ws[m_elite+1:end])
    E = n^0.5 * (1 - 1 / (4n) + 1 / (21 * n^2))

    pol = CMAMPPI_Policy(params, env, U₀, Σ, opt_its, σ, m_elite,
        ws, μ_eff, cσ, dσ, cΣ, c1, cμ, E, rng, mppi_logger)
    return pol
end

function calculate_trajectory_costs(pol::CMAMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    cs = pol.params.cs
    σ = pol.σ
    m_elite = pol.m_elite
    ws, μ_eff, cσ, dσ = pol.ws, pol.μ_eff, pol.cσ, pol.dσ
    cΣ, c1, cμ, E_cma = pol.cΣ, pol.c1, pol.cμ, pol.E

    # Initial covariance of distribution
    U_orig = pol.U
    Σ = pol.Σ

    pσ, pΣ = zeros(pol.params.cs), zeros(pol.params.cs)
    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        if N > 1
            P = Distributions.MvNormal(σ^2 * Σ)
        else
            P = Distributions.MvNormal(Σ)
        end
        Σ_inv = Distributions.invcov(P)
        E = rand(pol.rng, P, K)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)

        if n < N
            # Reorder, select elite samples, fit new distribution
            order = sortperm(trajectory_cost)
            elite_E = E[:, order[1:m_elite]]

            elite_traj_cost = trajectory_cost[order[1:m_elite]]
            if maximum(abs.(diff(elite_traj_cost, dims=1))) < 10e-3
                break
            end

            # selection and mean update
            δs = elite_E / σ
            δw = zeros(Float64, cs)
            for rᵢ ∈ 1:cs
                δw[rᵢ] = ws[1:m_elite]' * elite_E[rᵢ, :]
            end
            pol.U += σ * δw

            # step-size control
            C = Σ^-0.5
            pσ = (1 - cσ) * pσ + sqrt(cσ * (2 - cσ) * μ_eff) * C * δw
            σ *= exp(cσ / dσ * (norm(pσ) / E_cma - 1))

            # covariance adaptation
            hσ = Int(norm(pσ) / sqrt(1 - (1 - cσ)^(2n)) < (1.4 + 2 / (cs + 1)) * E_cma)
            pΣ = (1 - cΣ) * pΣ + hσ * sqrt(cΣ * (2 - cΣ) * μ_eff) * δw

            temp_sum = 0
            for ii in 1:K
                if ws[ii] ≥ 0
                    w0 = ws[ii]
                else
                    w0 = n * ws[ii] / norm(C * δs[order[ii]])^2
                end
                temp_sum += w0 * δs[order[ii]] * δs[order[ii]]'
            end

            Σ = (1 - c1 - cμ) * Σ + c1 * (pΣ * pΣ' + (1 - hσ) * cΣ * (2 - cΣ) * Σ) .+ cμ * temp_sum
            Σ = triu(Σ) + triu(Σ, 1)' # enforce symmetry
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
μAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Simple, mean-only M-PMC AIS version of MPOPI
"""
mutable struct μAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
μAISMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function μAISMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    λ_ais::Float64=20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return μAISMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::μAISMPPI_Policy, env::AbstractEnv)
Simple AIS strategy in which only the new mean is adapted at each
iteration. New samples are then taken from the new distribution.
"""
function calculate_trajectory_costs(pol::μAISMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    U_orig = pol.U
    P = Distributions.MvNormal(pol.Σ)
    Σ_inv = Distributions.invcov(P)

    trajectory_cost = Vector{Float64}(undef, K)
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        E = rand(pol.rng, P, K)
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            pw = StatsBase.ProbabilityWeights(ws)
            (μ′, Σ′) = StatsBase.mean_and_cov(E, pw, 2)
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
μΣAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Mean and Covariance M-PMC AIS with one distribution version of MPOPI
"""
mutable struct μΣAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
μΣAISMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
    kwargs passed to MPPI_Policy_Params
"""
function μΣAISMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    λ_ais::Float64=20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return μΣAISMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::μΣAISMPPI_Policy, env::AbstractEnv)
Simple AIS strategy in which the new mean and covariance are adapted at each
iteration. New samples are then taken from the new distribution.
"""
function calculate_trajectory_costs(pol::μΣAISMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = Vector{Float64}(undef, K)
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            pw = StatsBase.ProbabilityWeights(ws)
            (μ′, Σ′) = StatsBase.mean_and_cov(E, pw, 2)
            Σ′ = Σ′ + +10e-9 * I
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
PMCMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    PMC with multinomial resampling AIS with one distribution version of MPOPI
"""
mutable struct PMCMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
PMCMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function PMCMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    λ_ais::Float64=20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return PMCMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::PMCMPPI_Policy, env::AbstractEnv)
Generic PMC strategy.
O Cappé, A Guillin, J. M Marin & C. P Robert (2004)
Population Monte Carlo, Journal of Computational and Graphical Statistics,
13:4, 907-929, DOI: 10.1198/106186004X12803
"""
function calculate_trajectory_costs(pol::PMCMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = Vector{Float64}(undef, K)
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            resample_cat_dist = Categorical(ws)
            resample_idxs = rand(pol.rng, resample_cat_dist, K)
            E′ = E[:, resample_idxs]
            (μ′, Σ′) = StatsBase.mean_and_cov(E′, 2)
            Σ′ = Σ′ + +10e-9 * I
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
NESMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Natual evolution strategy version of MPOPI
"""
mutable struct NESMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    A::Matrix{Float64}
    opt_its::Int
    step_factor::Float64
    rng::R
    logger::MPPI_Logger
end

"""
NESMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    step_factor::Float64 = 0.01,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function NESMPPI_Policy(env::AbstractEnv;
    opt_its::Int=10,
    step_factor::Float64=0.01,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    A = sqrt(Σ)
    pol = NESMPPI_Policy(params, env, U₀, Σ, A, opt_its, step_factor, rng, mppi_logger)
    return pol
end

function calculate_trajectory_costs(pol::NESMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its

    # Initial covariance and principal matrix
    U_orig = pol.U
    Σ′ = pol.Σ
    A′ = pol.A

    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if maximum(abs.(diff(trajectory_cost, dims=1))) < 10e-3
            break
        end

        if n < N
            ∇μlog_p_x = zeros(Float64, size(pol.U))
            ∇Alog_p_x = zeros(Float64, size(pol.Σ))
            for k ∈ 1:K
                ∇μlog_p_x .+= Σ_inv * E[:, k] .* trajectory_cost[k]
                ∇Σlog_p_x = 1 / 2 * Σ_inv * E[:, k] * E[:, k]' * Σ_inv - 1 / 2 * Σ_inv
                ∇Alog_p_x += A′ * (∇Σlog_p_x + ∇Σlog_p_x') * trajectory_cost[k]
            end
            A′ -= pol.step_factor / K .* ∇Alog_p_x ./ K
            Σ′ = A′' * A′
            pol.U -= pol.step_factor / K .* ∇μlog_p_x
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
    build_hankel(data::Vector{Vector{Float64}}, T::Int)

Construct a Hankel matrix from a vector of vectors.
Each column is a window of length T from the sequence.
"""
function build_hankel(data::Vector{Vector{Float64}}, T::Int)
    N = length(data)
    if N < T
        error("Not enough data to build Hankel matrix (need at least $T samples).")
    end
    hankel = [reduce(vcat, data[i:i+T-1]) for i in 1:(N-T+1)]
    return hcat(hankel...)
end

function save_hankel_to_logger!(logger::MPPI_Logger, controls::Vector{Vector{Float64}}, states::Vector{Vector{Float64}}, hankel_window::Int)
    logger.U_hankel = build_hankel(controls, hankel_window)
    logger.Y_hankel = build_hankel(states, hankel_window)
end

"""
    show_hankel_matrices(controls, states, hankel_window)

Build and print Hankel matrices for controls and states.
"""
function show_hankel_matrices(controls::Vector{Vector{Float64}}, states::Vector{Vector{Float64}}, hankel_window::Int)
    U_hankel = build_hankel(controls, hankel_window)
    Y_hankel = build_hankel(states, hankel_window)
    println("Hankel matrix for inputs (U): size = ", size(U_hankel))
    println("First few columns of U_hankel:")
    println(U_hankel[:, 1:min(3, size(U_hankel, 2))])
    println("Hankel matrix for states (Y): size = ", size(Y_hankel))
    println("First few columns of Y_hankel:")
    println(Y_hankel[:, 1:min(3, size(Y_hankel, 2))])
end