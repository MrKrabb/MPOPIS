
"""
Example simulating the car racing environment.

Dependencies:
 - MPOPIS
 - Printf
 - Random
 - Plots
 - ProgressMeter
 - Dates
 - LinearAlgebra
 - Distributions
 - example_utils.jl
"""

""" 
simulate_car_racing(; kwargs...)
    Simulate a car racing scenarion with 1 or multiple cars
kwargs:
 - num_trials = 1,                                   # Number of trials
 - num_steps = 200,                                  # Max number of steps per scenario
 - num_cars = 1,                                     # Number of cars
 - laps = 2,                                         # Max number of laps (if using curve.csv track)
 - policy_type = :cemppi,                            # Type of policy (see `get_policy` for options)
 - num_samples = 150,                                # Number of initial samples for the policy
 - horizon = 50,                                     # Time step horizon
 - λ = 10.0,                                         # Inverve temperature settings for IT weighting
 - α = 1.0,                                          # Control cost parameter
 - U₀ = zeros(Float64, num_cars*2),                  # Default initial contorl
 - cov_mat = block_diagm([0.0625, 0.1], num_cars),   # Control covariance matrix
 - ais_its = 10,                                     # Number of AIS algo iterations
 - λ_ais = 20.0,                                     # Inverse temperature for AIS algo (if applicable)
 - ce_elite_threshold = 0.8,                         # CE elite threshold (if applicable)
 - ce_Σ_est = :ss,                                   # CE Σ estimation methods (if applicable)
 - cma_σ = 0.75,                                     # CMA step factor (if applicable)
 - cma_elite_threshold = 0.8,                        # CMA elite threshold (if applicable)
 - state_x_sigma = 0.0,                              # Add normal noise std dev to x position at each time step (only for single car)
 - state_y_sigma = 0.0,                              # Add normal noise std dev to y position at each time step (only for single car)
 - state_ψ_sigma = 0.0,                              # Add normal noise std dev to heading at each time step (only for single car)
 - seed = Int(rand(1:10e10)),                        # Seed algorithm and envrionment (incremtented by trial number)
 - log_runs = true,                                  # Output results at each of each trial
 - plot_steps = false,                               # Plot each step (NOT RECOMMENDED FOR LARGE RUNS)
 - pol_log = false,                                  # Turn on policy logger (slows down the process)
 - plot_traj = false,                                # Plot the samples trajectories
 - plot_traj_perc = 1.0,                             # Percentage of samples trajectories to plot (if applicable)
 - text_with_plot = true,                            # Plot car 1 state info on plot
 - text_on_plot_xy = (80.0, -60.0)                   # XY position of output text (if applicable)
 - save_gif = false,                                 # Save gif
"""
function hankel_blocks_local(data::AbstractMatrix{T}, L::Int) where T
    d, N = size(data)
    N ≥ L || error("Need N ≥ L (got N=$N, L=$L)")
    cols = N - L + 1
    H = Matrix{T}(undef, d*L, cols)
    @inbounds for j in 1:cols
        w = data[:, j:j+L-1]              # d × L
        H[:, j] = reshape(permutedims(w, (2,1)), d*L)
    end
    return H
end

# Combine DeePC blocks back into full Hankel (inputs & outputs) and data matrix
function combine_deepc_blocks(U_p::AbstractMatrix, U_f::AbstractMatrix, Y_p::AbstractMatrix, Y_f::AbstractMatrix)
    U_block = vcat(U_p, U_f)  # (m*(T_ini+N_pred) × cols)
    Y_block = vcat(Y_p, Y_f)  # (p*(T_ini+N_pred) × cols)
    W = vcat(U_p, Y_p, U_f, Y_f)  # Data matrix used in DeePC constraints
    return U_block, Y_block, W
end

# Generate row labels for DeePC combined matrix W = [U_p; Y_p; U_f; Y_f]
# Inputs assumed order: steering, pedal
# States order (CarRacingEnv): x, y, psi, Vx, Vy, psi_dot, delta, pedal_state
function deepc_W_row_labels(m::Int, p::Int, T_ini::Int, N_pred::Int;
    input_names = ["steer", "pedal"],
    state_names = ["x","y","psi","Vx","Vy","psi_dot","delta","pedal_state"],
    past_tag = "p", future_tag = "f")
    length(input_names) == m || error("input_names length $(length(input_names)) != m = $m")
    length(state_names) == p || error("state_names length $(length(state_names)) != p = $p")
    labels = String[]
    # Past inputs
    for i in 1:m
        for t in 0:T_ini-1
            push!(labels, "$(input_names[i])_$(past_tag)_t$(t)")
        end
    end
    # Past states
    for i in 1:p
        for t in 0:T_ini-1
            push!(labels, "$(state_names[i])_$(past_tag)_t$(t)")
        end
    end
    # Future inputs
    for i in 1:m
        for t in 0:N_pred-1
            push!(labels, "$(input_names[i])_$(future_tag)_t$(t)")
        end
    end
    # Future states
    for i in 1:p
        for t in 0:N_pred-1
            push!(labels, "$(state_names[i])_$(future_tag)_t$(t)")
        end
    end
    return labels
end

# Write annotated CSV: first column row_label, first row header with window indices
function write_W_with_labels(path::AbstractString, W::AbstractMatrix, labels::Vector{String})
    size(W,1) == length(labels) || error("Row label count $(length(labels)) != number of W rows $(size(W,1))")
    open(path, "w") do io
        cols = size(W,2)
        # Header
        print(io, "row_label")
        for j in 1:cols
            print(io, ",win$(j)")  # winj corresponds to window starting at time j
        end
        print(io, "\n")
        # Data rows
        @inbounds for r in 1:length(labels)
            print(io, labels[r])
            for c in 1:cols
                print(io, ",", W[r,c])
            end
            print(io, "\n")
        end
    end
    return nothing
end

"""
        deepc_form_g(W, u_past, y_past, m, p, T_ini; λ_reg=1e-6, enforce_sum=true, nonneg=false)

Compute the DeePC combination vector g using the past blocks within the combined
DeePC data matrix W = [U_p; Y_p; U_f; Y_f]. Only U_p and Y_p are required to
solve for g (future blocks are unused for the data equation).

Arguments:
    W        :: AbstractMatrix   Combined data matrix (rows = stacked blocks)
    u_past   :: AbstractVector   Length m*T_ini (most recent past inputs, oldest→newest)
    y_past   :: AbstractVector   Length p*T_ini (most recent past states, oldest→newest)
    m        :: Int              Number of inputs
    p        :: Int              Number of states (outputs)
    T_ini    :: Int              Past window length

Keyword arguments:
    λ_reg        :: Real  Ridge regularization (improves conditioning)
    enforce_sum  :: Bool  Impose 1' g = 1 via KKT (common in DeePC)
    nonneg       :: Bool  Project g onto nonnegative simplex (g ≥ 0, sum=1)

Returns:
    g :: Vector (length = number of columns in W)

Notes:
    - Assumes W row ordering produced by this code: first m*T_ini rows = U_p,
        next p*T_ini rows = Y_p.
    - Future blocks (following rows) are ignored when forming g.
    - If you loaded W from CSV, ensure u_past / y_past use the SAME ordering
        (input/ state for times t=0..T_ini-1). This code expects them stacked by
        variable then time exactly as in the internal logging (see hankel builder).
"""
function deepc_form_g(W::AbstractMatrix, u_past::AbstractVector, y_past::AbstractVector,
        m::Int, p::Int, T_ini::Int; λ_reg=1e-6, enforce_sum::Bool=true, nonneg::Bool=false)

        n_up_rows = m * T_ini
        n_yp_rows = p * T_ini
        size(W,1) >= n_up_rows + n_yp_rows || error("W does not contain enough rows for U_p and Y_p blocks")
        U_p = W[1:n_up_rows, :]
        Y_p = W[n_up_rows+1 : n_up_rows + n_yp_rows, :]

        length(u_past) == n_up_rows || error("u_past length $(length(u_past)) ≠ m*T_ini = $(n_up_rows)")
        length(y_past) == n_yp_rows || error("y_past length $(length(y_past)) ≠ p*T_ini = $(n_yp_rows)")

        # Data equation A g = b
        A = vcat(U_p, Y_p)              # ((m+p)T_ini × N)
        b = vcat(u_past, y_past)        # ((m+p)T_ini)
        N = size(A,2)

        # Normal equations with ridge: (A'A + λI) g = A' b, add sum constraint if desired
        AtA = A' * A
        Q = AtA .+ (λ_reg * I)          # (N×N)
        Atb = A' * b
        if enforce_sum
                KKT = [Q ones(N); ones(N)' 0.0]
                rhs = [Atb; 1.0]
                sol = KKT \ rhs
                g = sol[1:N]
        else
                g = Q \ Atb
        end

        if nonneg
                # Project onto simplex: g >= 0, sum(g)=1 (simple sorting projection)
                g = max.(g, 0)
                s = sum(g)
                if s > 0
                        g ./= s
                else
                        g .= 1 / N
                end
                if enforce_sum == false
                        # ensure sum normalization even if enforce_sum was false
                        g ./= sum(g)
                end
        end
        return g
end

"""
    deepc_random_g(W; rng=Random.GLOBAL_RNG, simplex=true)

Return a random linear combination vector g (length = number of columns of W).

If simplex=true (default), g is sampled from a uniform Dirichlet on the
probability simplex (g ≥ 0, sum(g)=1), matching the common DeePC constraint
1' g = 1 and nonnegativity (though DeePC does not always require g ≥ 0).

If simplex=false, returns a zero‑mean Gaussian vector normalized so sum(g)=1
but entries can be negative.
"""
function deepc_random_g(W::AbstractMatrix; rng=Random.GLOBAL_RNG, simplex::Bool=true)
    N = size(W,2)
    if simplex
        g = rand(rng, Dirichlet(ones(N)))
    else
        g = randn(rng, N)
        s = sum(g)
        s ≈ 0 && (g[1] += 1.0; s = sum(g))
        g ./= s
    end
    return g
end

"""
    calculate_trajectory_costs_deppi(U_p, Y_p, gs; u_past=nothing, y_past=nothing, target=:last, pnorm=2, lambda_g=0.0)

Compute residual costs for a set of linear combinations g based on the DeePC
data equation: cost(g) = || [U_p; Y_p] g - [u_past; y_past] ||_p^p (p=2 by default).

If u_past/y_past are not provided, the target is taken from one column of
U_p, Y_p specified by `target` (:last or an Int column index).
`gs` may be a Vector of vectors, or a matrix with each column a g.
Adds an L1 regularization term lambda_g * ||g||_1 outside the residual sum (DeePC-style).
"""
function calculate_trajectory_costs_deppi(U_p::AbstractMatrix, Y_p::AbstractMatrix, gs;
    u_past=nothing, y_past=nothing, target=:last, pnorm::Int=2, lambda_g::Real=0.0)

    A = vcat(U_p, Y_p)
    # Determine target vector b
    if u_past === nothing || y_past === nothing
        col = target === :last ? size(U_p,2) : Int(target)
        @assert 1 <= col <= size(U_p,2) "target column out of range"
        b = vcat(U_p[:, col], Y_p[:, col])
    else
        b = vcat(u_past, y_past)
    end

    # Normalize gs input
    g_list = isa(gs, AbstractMatrix) ? [view(gs, :, j) for j in 1:size(gs,2)] : gs
    costs = similar(collect(1:length(g_list)), Float64)
    @inbounds for (i, g) in enumerate(g_list)
        r = A * g - b
        base = pnorm == 2 ? sum(abs2, r) : (pnorm == 1 ? sum(abs, r) : sum(abs.(r) .^ pnorm))
        reg  = lambda_g == 0 ? 0.0 : lambda_g * sum(abs, g)
        costs[i] = base + reg
    end
    return costs
end

# Softmin-style weights from costs (Information-Theoretic-like)
function deepc_softmin_weights(costs::AbstractVector{<:Real}; λ_w::Real=10.0)
    cmin = minimum(costs)
    ex = exp.(-(costs .- cmin) ./ max(λ_w, eps()))
    s = sum(ex)
    if s <= 0 || !isfinite(s)
        return fill(1.0 / length(costs), length(costs))
    end
    return ex ./ s
end

# Random g candidates of length N; simplex=true → Dirichlet, else normalized Gaussian
function deepc_random_g_len(N::Int; rng=Random.GLOBAL_RNG, simplex::Bool=true)
    if simplex
        return rand(rng, Dirichlet(ones(N)))
    else
        g = randn(rng, N)
        s = sum(g)
        s ≈ 0 && (g[1] += 1.0; s = sum(g))
        return g ./ s
    end
end

# One online update step for nominal g_c using sampled candidates and softmin weights
function deepc_update_nominal_g!(g_c::AbstractVector{<:Real}, U_p::AbstractMatrix, Y_p::AbstractMatrix,
    u_ini::AbstractVector, y_ini::AbstractVector;
    num_samples::Int=64, λ_w::Real=10.0, lambda_g::Real=0.0, simplex::Bool=true, step::Real=1.0, rng=Random.GLOBAL_RNG,
    project_sum1::Bool=true)

    N = length(g_c)
    # Build candidate set (include current g_c)
    G = Matrix{Float64}(undef, N, num_samples)
    for j in 1:num_samples-1
        G[:, j] = deepc_random_g_len(N; rng=rng, simplex=simplex)
    end
    G[:, num_samples] = Float64.(g_c)
    # Costs for each candidate against executed past
    costs = calculate_trajectory_costs_deppi(U_p, Y_p, G; u_past=u_ini, y_past=y_ini, pnorm=2, lambda_g=lambda_g)
    # Weights and weighted estimate
    w = deepc_softmin_weights(costs; λ_w=λ_w)
    g_hat = G * w
    # Update
    g_new = (1 - step) .* g_c .+ step .* g_hat
    if project_sum1
        s = sum(g_new)
        s ≈ 0 && (g_new .= 1 / N)
        g_new ./= sum(g_new)
    end
    g_c .= g_new
    return g_c
end

"""
    save_trajectory_costs_deppi(path::AbstractString, indices::AbstractVector, costs::AbstractVector)

Save (index, cost) pairs to CSV.
"""
function save_trajectory_costs_deppi(path::AbstractString, indices::AbstractVector, costs::AbstractVector)
    @assert length(indices) == length(costs)
    data = hcat(indices, costs)
    writedlm(path, data, ',')
    return nothing
end

"""
    original_trajectory_costs_deppi(pol, env)

Call the existing calculate_trajectory_costs(pol, env) and return only the
trajectory cost vector, regardless of the specific policy overload.
"""
function original_trajectory_costs_deppi(pol, env)
    res = calculate_trajectory_costs(pol, env)
    if res isa Tuple
        return res[1]
    else
        return res
    end
end

"""
    describe_deepc_layout(U_p, U_f, Y_p, Y_f)

Print an ASCII layout showing how rows correspond to inputs & states over past (T_ini) and future (N_pred) windows.
"""
function describe_deepc_layout(U_p, U_f, Y_p, Y_f)
    T_ini = size(U_p, 1) ÷ max(1, size(U_f,1) == 0 ? 1 : (size(U_p,1) ÷ (size(U_p,1) ÷ (size(U_p,1)))))  # placeholder to avoid division by zero
    # Recover m, p, T_ini, N_pred robustly
    # U_p rows = m*T_ini, U_f rows = m*N_pred
    m = size(U_p,1)
    p = size(Y_p,1)
    # Infer m, p by gcd with U_f/Y_f if possible
    if size(U_f,1) > 0
        g = gcd(size(U_p,1), size(U_f,1))
        m = g
    end
    if size(Y_f,1) > 0
        g2 = gcd(size(Y_p,1), size(Y_f,1))
        p = g2
    end
    T_ini = size(U_p,1) ÷ m
    N_pred = size(U_f,1) ÷ m
    cols = size(U_p,2)
    println("DeePC Blocks Layout (columns = $cols overlapping windows):")
    println("  U_p: m*T_ini = $(m)*$(T_ini) rows (past inputs)")
    println("  Y_p: p*T_ini = $(p)*$(T_ini) rows (past states)")
    println("  U_f: m*N_pred = $(m)*$(N_pred) rows (future inputs)")
    println("  Y_f: p*N_pred = $(p)*$(N_pred) rows (future states)")
    println()    
    println("Combined Hankel (inputs only): U_block = vcat(U_p, U_f) size = ", (m*(T_ini+N_pred), cols))
    println("Combined Hankel (states only): Y_block = vcat(Y_p, Y_f) size = ", (p*(T_ini+N_pred), cols))
    println("DeePC Data Matrix W = vcat(U_p, Y_p, U_f, Y_f) size = ", (m*T_ini + p*T_ini + m*N_pred + p*N_pred, cols))
    println()    
    println("Row grouping in W:")
    r1 = 1; r2 = m*T_ini
    println("  Rows $r1:$r2 => U_p (past inputs)")
    r1 = r2 + 1; r2 += p*T_ini
    println("  Rows $r1:$r2 => Y_p (past states)")
    r1 = r2 + 1; r2 += m*N_pred
    println("  Rows $r1:$r2 => U_f (future inputs)")
    r1 = r2 + 1; r2 += p*N_pred
    println("  Rows $r1:$r2 => Y_f (future states)")
    println()    
    println("Within each block, consecutive groups of $m (inputs) or $p (states) rows correspond to time t = 0,1,... within that window.")
end

function simulate_car_racing(;
    num_trials = 1,
    num_steps = 200,
    num_cars = 1,
    policy_type = :cemppi,
    laps = 2,
    num_samples = 150, 
    horizon = 50,
    λ = 10.0,
    α = 1.0,
    U₀ = zeros(Float64, num_cars*2),
    cov_mat = block_diagm([0.0625, 0.1], num_cars),
    ais_its = 10,
    λ_ais = 20.0,
    ce_elite_threshold = 0.8,
    ce_Σ_est = :ss,
    cma_σ = 0.75,
    cma_elite_threshold = 0.8,
    state_x_sigma = 0.0,
    state_y_sigma = 0.0,
    state_ψ_sigma = 0.0,
    seed = Int(rand(1:10e10)),
    log_runs = true,
    plot_steps = false,
    pol_log = false,
    plot_traj = false,
    plot_traj_perc = 1.0,
    text_with_plot = true,
    text_on_plot_xy = (80.0, -60.0),
    save_gif = false,
    collect_hankel = false,
    T_ini = 20,
    N_pred = 15,
    save_hankel = false,              # If true and collect_hankel, save Hankel blocks to disk
    hankel_dir = "hankel_data",       # Output directory for Hankel CSV files
    hankel_prefix = "car",            # File name prefix
    show_deepc_layout = false,         # Print DeePC layout summary when Hankels built
    save_combined_hankel = false,      # Save combined DeePC data matrix W = [U_p; Y_p; U_f; Y_f]
    label_combined_hankel = false,     # Additionally write labeled CSV for W (row labels + window headers)
    generate_random_gs = false,        # Generate and save random DeePC combination vectors g
    num_random_g = 50,                 # Number of additional indices after start (0..50 makes 51)
    g_start_index = 0,                 # Starting index for naming (g_0, g_1, ...)
    g_simplex = true,                  # Sample g on simplex (Dirichlet) vs Gaussian normalized
    lambda_g = 0.0,                    # L1 regularization weight on g in _deppi costs
    # Input excitation to improve persistent excitation of data (applied to executed control)
    excite_inputs::Bool = false,       # If true, add zero-mean Gaussian dither to applied control
    excite_std = (0.05, 0.05),         # Std dev per input (tuple length must equal m at runtime)
    excite_decay::Float64 = 1.0,       # Per-step multiplicative decay (<=1.0); 1.0 = no decay
    hankel_accumulate_trials = false,  # If true, accumulate Hankel windows (columns) across trials
    save_accumulated_hankel = false,   # If true, save the accumulated W at the end of all trials
    hankel_target_cols = 0,            # If >0, when saving, select this many columns from Hankels
    hankel_select_strategy::Symbol = :first,  # :first | :even | :random selection when trimming columns
    hankel_goal_cols::Int = 0,         # If >0, extend steps/laps this trial so N-L+1 >= hankel_goal_cols
    hankel_stride::Int = 1,            # If >1, subsample time-series before Hankel to reduce correlation
    pe_diagnostics::Bool = false,      # If true, print rank/SVD diagnostics for input Hankel
)

    if num_cars > 1
        sim_type = :mcr
    else
        sim_type = :cr
    end

    @printf("\n")
    @printf("%-30s%s\n", "Sim Type:", sim_type)
    @printf("%-30s%d\n", "Num Cars:", num_cars)
    @printf("%-30s%d\n", "Num Trails:", num_trials)
    @printf("%-30s%d\n", "Num Steps:", num_steps)
    @printf("%-30s%d\n", "Max Num Laps:", laps)
    @printf("%-30s%s\n","Policy Type:", policy_type)
    @printf("%-30s%d\n", "Num samples", num_samples)
    @printf("%-30s%d\n", "Horizon", horizon)
    @printf("%-30s%.2f\n", "λ (inverse temp):", λ)
    @printf("%-30s%.2f\n", "α (control cost param):", α)
    if policy_type != :mppi && policy_type != :gmppi
        @printf("%-30s%d\n", "# AIS Iterations:", ais_its)
        if policy_type ∈ [:μΣaismppi, :μaismppi, :pmcmppi]
            @printf("%-30s%.2f\n", "λ_ais (ais inverse temp):", λ_ais)
        elseif policy_type == :cemppi
            @printf("%-30s%.2f\n", "CE Elite Threshold:", ce_elite_threshold)
            @printf("%-30s%s\n", "CE Σ Est Method:", ce_Σ_est)
        elseif policy_type == :cmamppi
            @printf("%-30s%.2f\n", "CMA Step Factor (σ):", cma_σ)
            @printf("%-30s%.2f\n", "CMA Elite Perc Thres:", cma_elite_threshold)
        end
    end
    @printf("%-30s[%.4f, ..., %.4f]\n", "U₀", U₀[1], U₀[end])
    @printf("%-30s%s([%.4f %.4f; %.4f %.4f], %d)\n", "Σ", "block_diagm",
        cov_mat[1,1], cov_mat[1,2], cov_mat[2,1], cov_mat[2,2], num_cars)
    if num_cars == 1
        @printf("%-30s%.4f\n", "Noise, State X σ:", state_x_sigma)
        @printf("%-30s%.4f\n", "Noise, State Y σ:", state_y_sigma)
        @printf("%-30s%.4f\n", "Noise, Heading σ:", state_ψ_sigma)
    end
    @printf("%-30s%d\n", "Seed:", seed)
    @printf("\n")
    
    # Must have policy log on if plotting trajectories
    if plot_traj
        pol_log = true
    end
    
    gif_name = "$sim_type-$num_cars-$policy_type-$num_samples-$horizon-$λ-$α-"
    if policy_type != :mppi && policy_type != :gmppi
        gif_name = gif_name * "$ais_its-"
    end
    if policy_type == :cemppi
        gif_name = gif_name * "$ce_elite_threshold-"
        gif_name = gif_name * "$ce_Σ_est-"
    elseif policy_type ∈ [:μΣaismppi, :μaismppi, :pmcmppi]
        gif_name = gif_name * "$λ_ais-"
    elseif policy_type == :cmamppi
        gif_name = gif_name * "$cma_σ-"
        gif_name = gif_name * "$cma_elite_threshold-"
    end
    gif_name = gif_name * "$num_trials-$laps.gif"
    anim = Animation()

    rews = zeros(Float64, num_trials)
    steps = zeros(Float64, num_trials)
    rews_per_step = zeros(Float64, num_trials)
    lap_ts = [zeros(Float64, num_trials) for _ in 1:laps]
    mean_vs = zeros(Float64, num_trials)
    max_vs = zeros(Float64, num_trials)
    mean_βs = zeros(Float64, num_trials)
    max_βs = zeros(Float64, num_trials)
    β_viols = zeros(Float64, num_trials)
    T_viols = zeros(Float64, num_trials)
    C_viols = zeros(Float64, num_trials)
    exec_times = zeros(Float64, num_trials)  

    @printf("Trial    #: %12s : %7s: %12s", "Reward", "Steps", "Reward/Step")
    for ii ∈ 1:laps
        @printf(" : %6s%d", "lap ", ii)
    end
    @printf(" : %7s : %7s", "Mean V", "Max V")
    @printf(" : %7s : %7s", "Mean β", "Max β")
    @printf(" : %7s : %7s", "β Viol", "T Viol")
    if sim_type == :mcr
        @printf(" : %7s", "C Viol")
    end
    @printf(" : %7s", "Ex Time")
    @printf("\n")

    # Optional cross-trial accumulation of DeePC data windows
    W_accum = nothing  # will hold vcat(U_p;Y_p;U_f;Y_f) with columns concatenated across trials
    m_acc = nothing    # remember m and p for labeling accumulated W
    p_acc = nothing

    # Helper to select desired column indices given a target
    function _select_indices(ncols::Int, target::Int, strat::Symbol)
        if target <= 0 || target >= ncols
            return collect(1:ncols)
        end
        if strat === :first
            return collect(1:target)
        elseif strat === :even
            idxs = unique(round.(Int, range(1, ncols, length=target)))
            # ensure exactly target by trimming/padding (padding by dropping duplicates at end)
            if length(idxs) > target
                idxs = idxs[1:target]
            elseif length(idxs) < target
                # fallback to first to reach target
                need = target - length(idxs)
                idxs = vcat(idxs, collect(1:need))
            end
            return sort(idxs)
        elseif strat === :random
            return sort(randperm(ncols)[1:target])
        else
            @warn "Unknown hankel_select_strategy=$(strat); using :first"
            return collect(1:target)
        end
    end

    # If a per-trial Hankel column goal is set, compute effective steps/laps
    # Columns = N - L + 1 with L = T_ini + N_pred and N ≈ number of collected samples
    L_goal = T_ini + N_pred
    eff_num_steps = num_steps
    eff_laps = laps
    if hankel_goal_cols > 0
        required_N = hankel_goal_cols + L_goal - 1
        # We collect one sample per step; N ≈ eff_num_steps + 1, so steps ≈ required_N - 1
        eff_num_steps = max(num_steps, required_N - 1)
        # Avoid early termination due to lap count if user asked for more columns
        eff_laps = max(laps, typemax(Int) ÷ 2)
        @printf("Hankel goal: target columns ≥ %d ⇒ setting effective steps=%d (L=%d)\n",
                hankel_goal_cols, eff_num_steps, L_goal)
    end

    for k ∈ 1:num_trials
        
        if sim_type == :cr
            env = CarRacingEnv(rng=MersenneTwister())
        elseif sim_type == :mcr
            env = MultiCarRacingEnv(num_cars, rng=MersenneTwister())
        end

    pol = get_policy(
            policy_type,
            env,num_samples, horizon, λ, α, U₀, cov_mat, pol_log, 
            ais_its, 
            λ_ais, 
            ce_elite_threshold, ce_Σ_est,
            cma_σ, cma_elite_threshold,  
        )
    # Determine per-trial effective steps/laps to satisfy desired Hankel columns
    # First honor an explicit hankel_goal_cols if provided; otherwise use formula N_col = (m+1)*(L+n) - 1
    L_goal = T_ini + N_pred
    # Base on previously computed global effective steps if any
    eff_num_steps_k = eff_num_steps
    eff_laps_k = eff_laps
    # Determine m,n from the environment state/action spaces
    m_est = action_space_size(action_space(env))
    n_est = isa(state(env), Vector) ? length(state(env)) : size(state(env), 2)
    if collect_hankel && hankel_goal_cols == 0
        N_col_target = (m_est + 1) * (L_goal + n_est) - 1
        required_N = N_col_target + L_goal - 1  # ensure N - L + 1 ≥ N_col_target
        eff_num_steps_k = max(eff_num_steps_k, required_N - 1)
        eff_laps_k = max(eff_laps_k, typemax(Int) ÷ 2)
        @printf("Hankel formula: m=%d n=%d L=%d ⇒ target columns=%d ⇒ effective steps=%d\n",
            m_est, n_est, L_goal, N_col_target, eff_num_steps_k)
    end
        # Hankel collection init (step 4)
        if collect_hankel
            m = action_space_size(action_space(env))
            p = isa(state(env), Vector) ? length(state(env)) : size(state(env), 2)
            u_hist = Matrix{Float64}(undef, m, 0)   # (m × N)
            y_hist = Matrix{Float64}(undef, p, 0)   # (p × N)
        end

        seed!(env, seed + k)
        seed!(pol, seed + k)

    pm = Progress(eff_num_steps_k, 1, "Trial $k ....", 50)
        # Start timer
        time_start = Dates.now()
        
        lap_time = zeros(Int, laps)
        v_mean_log = Vector{Float64}()
        v_max_log = Vector{Float64}()
        β_mean_log = Vector{Float64}()
        β_max_log = Vector{Float64}()
        rew, cnt, lap, prev_y = 0, 0, 0, 0
        trk_viol, β_viol, crash_viol = 0, 0, 0

        # Main simulation loop
    while !env.done && cnt <= eff_num_steps_k
            # Get action from policy
            act = pol(env)
            # Optionally add small input excitation (Gaussian dither) to improve data richness
            if excite_inputs
                # Build noise vector matching action dimension
                m_act = isa(act, AbstractVector) ? length(act) : length(vec(act))
                stds = collect(excite_std)
                if length(stds) != m_act
                    # If mismatch, pad/trim to match
                    if length(stds) < m_act
                        append!(stds, fill(stds[end], m_act - length(stds)))
                    else
                        stds = stds[1:m_act]
                    end
                end
                decay_fac = excite_decay <= 0 ? 1.0 : (excite_decay^max(0, cnt))
                noise = decay_fac .* (randn(m_act) .* stds)
                if isa(act, AbstractVector)
                    act = act .+ noise
                else
                    a = vec(act) .+ noise
                    act = reshape(a, size(act))
                end
                # Optional conservative clamp to [-1,1] per element (adjust if env uses different bounds)
                act = clamp.(act, -1.0, 1.0)
            end
            # Apply action to envrionment
            env(act)
            cnt += 1
            if collect_hankel
                u_vec = isa(act, AbstractVector) ? act : vec(act)
                u_hist = hcat(u_hist, u_vec)
                y_vec = isa(state(env), AbstractVector) ? copy(state(env)) : copy(state(env))[:]
                y_hist = hcat(y_hist, y_vec)
            end

            # Get reward at the step
            step_rew = reward(env)
            rew += step_rew

            # Plot or collect the plot for the animation
            if plot_steps || save_gif
                if plot_traj
                    p = plot(env, pol, plot_traj_perc, text_output=text_with_plot, text_xy=text_on_plot_xy)
                else 
                    p = plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)
                end
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end

            if sim_type == :cr
                env.state[1] += state_x_sigma * randn(env.rng)
                env.state[2] += state_y_sigma * randn(env.rng)

                δψ = state_ψ_sigma * randn(env.rng)
                env.state[3] += δψ
                
                # Passive rotation matrix
                rot_mat = [ cos(δψ) sin(δψ) ;
                           -sin(δψ) cos(δψ) ]
                V′ = rot_mat*[env.state[4]; env.state[5]]
                env.state[4:5] = V′
            end

            next!(pm)

            # Get logging information
            curr_y = env.state[2]
            if sim_type == :mcr
                curr_y = minimum([en.state[2] for en ∈ env.envs])    
                vs = [norm(en.state[4:5]) for en ∈ env.envs]
                βs = [abs(calculate_β(en)) for en ∈ env.envs]
            else
                vs = norm(env.state[4:5])
                βs = abs(calculate_β(env))
            end
            push!(v_mean_log, mean(vs))
            push!(v_max_log, maximum(vs))
            push!(β_mean_log, mean(βs))
            push!(β_max_log, maximum(βs))
            
            # Determine if violations occurred
            if step_rew < -4000
                ex_β = exceed_β(env)
                within_t = sim_type == :cr ? within_track(env).within : within_track(env)
                if ex_β β_viol += 1 end
                if !within_t trk_viol += 1 end
                temp_rew = step_rew + ex_β*5000 + !within_t*1000000
                if temp_rew < -10500 crash_viol += 1 end
            end

            if sim_type == :mcr
                # Not exact, but works
                d = minimum([norm(en.state[1:2]) for en ∈ env.envs])
            else
                d = norm(env.state[1:2])
            end

            # Estimate to increment lap count on curve.csv
            if prev_y < 0.0 && curr_y >= 0.0 && d <= 15.0
                lap += 1
                lap_time[lap] = cnt
            end
            if lap >= eff_laps_k || trk_viol > 10 || β_viol > 50
                env.done = true
            end
            prev_y = curr_y
        end

        # Step 6: Build (and optionally save) Hankel matrices for this trial
        if collect_hankel
            m = size(u_hist, 1)
            p = size(y_hist, 1)
            L = T_ini + N_pred
            if size(u_hist, 2) >= L && size(y_hist, 2) >= L
                # Optional subsampling/stride to decorrelate windows
                u_src = hankel_stride > 1 ? u_hist[:, 1:hankel_stride:end] : u_hist
                y_src = hankel_stride > 1 ? y_hist[:, 1:hankel_stride:end] : y_hist
                U_block = hankel_blocks_local(u_src, L)  # (m*L × cols)
                Y_block = hankel_blocks_local(y_src, L)  # (p*L × cols)
                U_p = U_block[1:m*T_ini, :]
                U_f = U_block[m*T_ini+1:end, :]
                Y_p = Y_block[1:p*T_ini, :]
                Y_f = Y_block[p*T_ini+1:end, :]
                @printf("Hankel (trial %d): U_p %s, U_f %s, Y_p %s, Y_f %s\n", k, size(U_p), size(U_f), size(Y_p), size(Y_f))
                    # Executed trajectory split into past/future (for reference and [u_ini;y_ini])
                    U_p_exec = U_block[1:m*T_ini, :]
                    Y_p_exec = Y_block[1:p*T_ini, :]
                    U_f_exec = U_block[m*T_ini+1:end, :]
                    Y_f_exec = Y_block[p*T_ini+1:end, :]
                    # Default to executed data
                    U_p = U_p_exec
                    Y_p = Y_p_exec
                    U_f = U_f_exec
                    Y_f = Y_f_exec
                    # If available, build DeePC blocks directly from MPPI rollouts across samples:
                    #   - U_p, Y_p: first T_ini steps of each rollout
                    #   - U_f, Y_f: last N_pred steps of each rollout
                    if pol_log && pol.logger.sample_controls !== nothing
                        K = length(pol.logger.sample_controls)
                        # total rollout length (time steps) as logged
                        Ttot = size(pol.logger.sample_controls[1], 1)
                        if Ttot < T_ini + N_pred
                            @warn "Logged rollout length Ttot=$(Ttot) < T_ini+N_pred=$(T_ini+N_pred). Falling back to executed Hankels for this trial."
                        else
                            # Build from controls
                            U_p_roll = Matrix{Float64}(undef, m*T_ini, K)
                            U_f_roll = Matrix{Float64}(undef, m*N_pred, K)
                            for kk in 1:K
                                Useg = pol.logger.sample_controls[kk]           # (Ttot × m)
                                U_p_roll[:, kk] = vec(Useg[1:T_ini, :])         # first T_ini steps
                                U_f_roll[:, kk] = vec(Useg[Ttot-N_pred+1:Ttot, :])  # last N_pred steps
                            end
                            # Build from predicted states
                            has_traj = pol.logger.trajectories !== nothing && length(pol.logger.trajectories) == K
                            if has_traj
                                Y_p_roll = Matrix{Float64}(undef, p*T_ini, K)
                                Y_f_roll = Matrix{Float64}(undef, p*N_pred, K)
                                for kk in 1:K
                                    Yseg = pol.logger.trajectories[kk]          # (Ttot × p)
                                    Y_p_roll[:, kk] = vec(Yseg[1:T_ini, :])
                                    Y_f_roll[:, kk] = vec(Yseg[Ttot-N_pred+1:Ttot, :])
                                end
                            else
                                @warn "State trajectories not logged; using executed Y_p/Y_f for all columns."
                                Y_p_roll = repeat(Y_p_exec[:, end:end], 1, K)
                                Y_f_roll = repeat(Y_f_exec[:, end:end], 1, K)
                            end
                            U_p = U_p_roll; Y_p = Y_p_roll; U_f = U_f_roll; Y_f = Y_f_roll
                        end
                    end
                if pe_diagnostics
                    # Check rank and singular values of input Hankel of order L
                    desired_rank = m * L
                    # Numeric rank using tolerance based on eps * max(size)
                    F = svd(U_block)
                    tol = maximum(size(U_block)) * eps(eltype(U_block)) * maximum(F.S)
                    rnk = count(>(tol), F.S)
                    @printf("[PE] U_block rank=%d (desired m*L=%d), min singular=%.3e, cols=%d\n",
                            rnk, desired_rank, minimum(F.S), size(U_block, 2))
                end
                # Always build combined blocks for optional saving
                U_block_full, Y_block_full, W = combine_deepc_blocks(U_p, U_f, Y_p, Y_f)
                if show_deepc_layout
                    describe_deepc_layout(U_p, U_f, Y_p, Y_f)
                end
                if save_hankel || save_combined_hankel || generate_random_gs
                    isdir(hankel_dir) || mkpath(hankel_dir)
                end
        # Initialize canonical DeePC combination vector g_c (all ones)
        g_c = ones(Float64, size(W, 2))
        # Perform one nominal g update using sampled candidates and executed past
                try
                    u_ini_vec = U_p_exec[:, end]
                    y_ini_vec = Y_p_exec[:, end]
                    deepc_update_nominal_g!(g_c, U_p, Y_p, u_ini_vec, y_ini_vec;
            num_samples=64, λ_w=10.0, lambda_g=lambda_g, simplex=true, step=1.0, rng=env.rng, project_sum1=true)
                catch err
                    @warn "g_c update skipped due to error: $(err)"
                end
                # Determine column indices for optional trimming when saving
                sel_idxs = _select_indices(size(W,2), hankel_target_cols, hankel_select_strategy)
                if save_hankel
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_p.csv"), U_p[:, sel_idxs], ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_f.csv"), U_f[:, sel_idxs], ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_p.csv"), Y_p[:, sel_idxs], ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_f.csv"), Y_f[:, sel_idxs], ',')
                end
                if save_combined_hankel
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_W.csv"), W[:, sel_idxs], ',')
                    # Save g_c (nominal DeePC combination vector)
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_g_c.csv"), g_c, ',')
                    # Also compute its DeePC residual cost against executed past and compare to MPPI
                    try
                        # Use executed past from last window for residual target
                        u_ini_loc = U_p_exec[:, end]
                        y_ini_loc = Y_p_exec[:, end]
                        gc_cost = calculate_trajectory_costs_deppi(U_p, Y_p, reshape(g_c, :, 1); u_past=u_ini_loc, y_past=y_ini_loc, pnorm=2, lambda_g=lambda_g)[1]
                        orig_costs = original_trajectory_costs_deppi(pol, env)
                        mppi_mean = mean(orig_costs)
                        mppi_min = minimum(orig_costs)
                        comp_path2 = joinpath(hankel_dir, "g_c_cost_comparison.csv")
                        open(comp_path2, "w") do io
                            println(io, "deppi_cost_g_c,mppi_mean_cost,mppi_min_cost,diff_min_minus_g_c,diff_mean_minus_g_c")
                            println(io, string(gc_cost, ",", mppi_mean, ",", mppi_min, ",", (mppi_min - gc_cost), ",", (mppi_mean - gc_cost)))
                        end
                    catch err
                        @warn "g_c cost comparison skipped due to error: $(err)"
                    end
                    if label_combined_hankel
                        labels = deepc_W_row_labels(m, p, T_ini, N_pred)
                        write_W_with_labels(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_W_labeled.csv"), W[:, sel_idxs], labels)
                    end
                    # Also save the DeePC vector [u_ini; y_ini; u; y]
                    # u_ini, y_ini = actually executed last T_ini results from environment (not simulated)
                    u_ini = U_p_exec[:, end]
                    y_ini = Y_p_exec[:, end]
                    # Choose rollout for [u; y]: best (min-cost) if available, else first
                    sel_col = 1
                    if pol_log && pol.logger.traj_costs !== nothing && length(pol.logger.traj_costs) > 0
                        sel_col = argmin(pol.logger.traj_costs)
                    end
                    # Ensure [u; y] correspond to last N_pred steps of the selected rollout
                    if pol_log && pol.logger.sample_controls !== nothing
                        Ttot = size(pol.logger.sample_controls[1], 1)
                        if Ttot >= N_pred
                            Useg_sel = pol.logger.sample_controls[sel_col]
                            u_vec = vec(Useg_sel[Ttot-N_pred+1:Ttot, :])
                        else
                            u_vec = U_f[:, sel_col]
                        end
                    else
                        u_vec = U_f[:, sel_col]
                    end
                    if pol_log && pol.logger.trajectories !== nothing
                        Ttot = size(pol.logger.trajectories[1], 1)
                        if Ttot >= N_pred
                            Yseg_sel = pol.logger.trajectories[sel_col]
                            y_vec = vec(Yseg_sel[Ttot-N_pred+1:Ttot, :])
                        else
                            y_vec = Y_f[:, sel_col]
                        end
                    else
                        y_vec = Y_f[:, sel_col]
                    end
                    uy = vcat(u_ini, y_ini, u_vec, y_vec)
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_uy_vec.csv"), uy, ',')
                    # Optionally also save the full input/state Hankels (commented out)
                    # writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_block.csv"), U_block_full, ',')
                    # writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_block.csv"), Y_block_full, ',')
                end
                # Optionally accumulate columns across trials to increase sample windows
                if hankel_accumulate_trials
                    if W_accum === nothing
                        W_accum = copy(W)
                        m_acc = m; p_acc = p
                    else
                        size(W_accum,1) == size(W,1) || @warn "Accumulated W row size $(size(W_accum,1)) != current W $(size(W,1)); skipping accumulation for trial $(k)."
                        if size(W_accum,1) == size(W,1)
                            W_accum = hcat(W_accum, W)
                        end
                    end
                end
                if generate_random_gs
                    # Generate indices from g_start_index to g_start_index + num_random_g (inclusive)
                    idx_range = g_start_index:(g_start_index + num_random_g)
                    # Use selected W for g sampling consistency when trimming
                    W_sel = W[:, sel_idxs]
                    G = Matrix{Float64}(undef, size(W_sel,2), length(idx_range))
                    indices = collect(idx_range)
                    for (col, n) in enumerate(idx_range)
                        g = deepc_random_g(W_sel; rng=env.rng, simplex=g_simplex)
                        G[:, col] = g
                        writedlm(joinpath(hankel_dir, "g_$(n).csv"), g, ',')
                    end
                    # Compute residual costs per g against last window of [U_p; Y_p]
                    g_costs = calculate_trajectory_costs_deppi(U_p[:, sel_idxs], Y_p[:, sel_idxs], G; target=:last, pnorm=2, lambda_g=lambda_g)
                    save_trajectory_costs_deppi(joinpath(hankel_dir, "g_costs.csv"), indices, g_costs)

                    # Compute original MPPI-style trajectory costs to compare
                    # Note: This is the vector of costs for the current policy call, simulated fresh.
                    orig_costs = original_trajectory_costs_deppi(pol, env)
                    # Align lengths by truncation/padding if necessary
                    Lmin = min(length(orig_costs), length(g_costs))
                    comp = zeros(Float64, Lmin, 3)
                    comp[:, 1] = orig_costs[1:Lmin]
                    comp[:, 2] = g_costs[1:Lmin]
                    comp[:, 3] = comp[:, 1] .- comp[:, 2]
                    comp_path = joinpath(hankel_dir, "g_costs_comparison.csv")
                    open(comp_path, "w") do io
                        println(io, "mppi_cost,deppi_cost,diff_mppi_minus_deppi")
                        writedlm(io, comp, ',')
                    end
                end
            else
                @printf("Trial %d: Not enough samples for Hankel (need ≥ %d, have %d). Skipping.\n", k, L, size(u_hist,2))
            end
        end

        if collect_hankel
            L = T_ini + N_pred
            if size(u_hist, 2) ≥ L
                Hu = hankel_blocks_local(u_hist, L)
                Hy = hankel_blocks_local(y_hist, L)
                # Partition (optional DeePC blocks)
                m = size(u_hist,1); p = size(y_hist,1)
                U_p = Hu[1:(m*T_ini), :]; U_f = Hu[(m*T_ini+1):(m*L), :]
                Y_p = Hy[1:(p*T_ini), :]; Y_f = Hy[(p*T_ini+1):(p*L), :]
                @printf("[Hankel] Trial %d: Hu=(%d,%d) Hy=(%d,%d)\n",
                        k, size(Hu,1), size(Hu,2), size(Hy,1), size(Hy,2))
                # (Optional) store or save:
                # pol.logger.U_hankel = Hu
                # pol.logger.Y_hankel = Hy
            else
                @printf("[Hankel] Trial %d: not enough data (need %d, have %d)\n",
                        k, L, size(u_hist,2))
            end
        end
        
        # Stop timer
        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000

        rews[k] = rew
        steps[k] = cnt-1
        rews_per_step[k] = rews[k]/steps[k]
        exec_times[k] = seconds_ran 
        if sim_type ∈ (:cr, :mcr)
            for ii ∈ 1:laps
                lap_ts[ii][k] = lap_time[ii]
            end
            mean_vs[k] = mean(v_mean_log)
            max_vs[k] = maximum(v_max_log)
            mean_βs[k] = mean(β_mean_log)
            max_βs[k] = maximum(β_max_log)
            β_viols[k] = β_viol
            T_viols[k] = trk_viol
            C_viols[k] = crash_viol
        end 

        # For clearing the progress bar
    if cnt > eff_num_steps_k
            print("\u1b[1F") # Moves cursor to beginning of the line n lines up 
            print("\u1b[0K") # Clears  part of the line. n=0: clear from cursor to end
        else
            print("\e[2K") # clear whole line
            print("\e[1G") # move cursor to column 1
        end
        if log_runs
            @printf("Trial %4d: %12.2f : %7d: %12.2f", k, rew, cnt-1, rew/(cnt-1))
            for ii ∈ 1:laps
                @printf(" : %7d", lap_time[ii])
            end
            @printf(" : %7.2f : %7.2f", mean(v_mean_log), maximum(v_max_log))
            @printf(" : %7.2f : %7.2f",  mean(β_mean_log), maximum(β_max_log))
            @printf(" : %7d : %7d", β_viol, trk_viol)
            if sim_type == :mcr
                @printf(" : %7d", crash_viol)
            end
            @printf(" : %7.2f", seconds_ran)
            @printf("\n")
        end
    end

    # Output summary results
    @printf("-----------------------------------\n")
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "AVE", mean(rews), mean(steps), mean(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", mean(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", mean(mean_vs), mean(max_vs))
    @printf(" : %7.2f : %7.2f",  mean(mean_βs), mean(max_βs))
    @printf(" : %7.2f : %7.2f", mean(β_viols), mean(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", mean(C_viols))
    end
    @printf(" : %7.2f\n", mean(exec_times))    
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "STD", std(rews), std(steps), std(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", std(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", std(mean_vs), std(max_vs))
    @printf(" : %7.2f : %7.2f",  std(mean_βs), std(max_βs))
    @printf(" : %7.2f : %7.2f", std(β_viols), std(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", std(C_viols))
    end
    @printf(" : %7.2f\n", std(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MED", 
        quantile_ci(rews)[2], quantile_ci(steps)[2], quantile_ci(rews_per_step)[2])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[2])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[2], quantile_ci(max_vs)[2])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[2], quantile_ci(max_βs)[2])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[2], quantile_ci(T_viols)[2])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[2])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[2])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "L95", 
        quantile_ci(rews)[1], quantile_ci(steps)[1], quantile_ci(rews_per_step)[1])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[1])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[1], quantile_ci(max_vs)[1])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[1], quantile_ci(max_βs)[1])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[1], quantile_ci(T_viols)[1])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[1])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[1])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "U95", 
        quantile_ci(rews)[3], quantile_ci(steps)[3], quantile_ci(rews_per_step)[3])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[3])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[3], quantile_ci(max_vs)[3])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[3], quantile_ci(max_βs)[3])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[3], quantile_ci(T_viols)[3])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[3])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[3])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MIN", 
        minimum(rews), minimum(steps), minimum(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", minimum(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", minimum(mean_vs), minimum(max_vs))
    @printf(" : %7.2f : %7.2f",  minimum(mean_βs), minimum(max_βs))
    @printf(" : %7.2f : %7.2f", minimum(β_viols), minimum(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", minimum(C_viols))
    end
    @printf(" : %7.2f\n", minimum(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MAX", maximum(rews), maximum(steps), maximum(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", maximum(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", maximum(mean_vs), maximum(max_vs))
    @printf(" : %7.2f : %7.2f",  maximum(mean_βs), maximum(max_βs))
    @printf(" : %7.2f : %7.2f", maximum(β_viols), maximum(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", maximum(C_viols))
    end
    @printf(" : %7.2f\n", maximum(exec_times))

    if save_gif
        println("Saving gif...$gif_name")
        gif(anim, gif_name, fps=10)
    end

    # Save accumulated Hankel (across trials) if requested
    if hankel_accumulate_trials && save_accumulated_hankel && W_accum !== nothing
        isdir(hankel_dir) || mkpath(hankel_dir)
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_alltrials_W.csv"), W_accum, ',')
        if label_combined_hankel && m_acc !== nothing && p_acc !== nothing
            labels = deepc_W_row_labels(m_acc, p_acc, T_ini, N_pred)
            write_W_with_labels(joinpath(hankel_dir, "$(hankel_prefix)_alltrials_W_labeled.csv"), W_accum, labels)
        end
        @printf("Saved accumulated DeePC data matrix W with size %s to %s\n", size(W_accum), joinpath(hankel_dir, "$(hankel_prefix)_alltrials_W.csv"))
    end
end

