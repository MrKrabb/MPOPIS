
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
        # Hankel collection init (step 4)
        if collect_hankel
            m = action_space_size(action_space(env))
            p = isa(state(env), Vector) ? length(state(env)) : size(state(env), 2)
            u_hist = Matrix{Float64}(undef, m, 0)   # (m × N)
            y_hist = Matrix{Float64}(undef, p, 0)   # (p × N)
        end

        seed!(env, seed + k)
        seed!(pol, seed + k)

        pm = Progress(num_steps, 1, "Trial $k ....", 50)
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
        while !env.done && cnt <= num_steps
            # Get action from policy
            act = pol(env)
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
            if lap >= laps || trk_viol > 10 || β_viol > 50
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
                U_block = hankel_blocks_local(u_hist, L)  # (m*L × cols)
                Y_block = hankel_blocks_local(y_hist, L)  # (p*L × cols)
                U_p = U_block[1:m*T_ini, :]
                U_f = U_block[m*T_ini+1:end, :]
                Y_p = Y_block[1:p*T_ini, :]
                Y_f = Y_block[p*T_ini+1:end, :]
                @printf("Hankel (trial %d): U_p %s, U_f %s, Y_p %s, Y_f %s\n", k, size(U_p), size(U_f), size(Y_p), size(Y_f))
                # Always build combined blocks for optional saving
                U_block_full, Y_block_full, W = combine_deepc_blocks(U_p, U_f, Y_p, Y_f)
                if show_deepc_layout
                    describe_deepc_layout(U_p, U_f, Y_p, Y_f)
                end
                if save_hankel || save_combined_hankel || generate_random_gs
                    isdir(hankel_dir) || mkpath(hankel_dir)
                end
                if save_hankel
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_p.csv"), U_p, ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_f.csv"), U_f, ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_p.csv"), Y_p, ',')
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_f.csv"), Y_f, ',')
                end
                if save_combined_hankel
                    writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_W.csv"), W, ',')
                    if label_combined_hankel
                        labels = deepc_W_row_labels(m, p, T_ini, N_pred)
                        write_W_with_labels(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_W_labeled.csv"), W, labels)
                    end
                    # Optionally also save the full input/state Hankels (commented out)
                    # writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_U_block.csv"), U_block_full, ',')
                    # writedlm(joinpath(hankel_dir, "$(hankel_prefix)_trial$(k)_Y_block.csv"), Y_block_full, ',')
                end
                if generate_random_gs
                    # Generate indices from g_start_index to g_start_index + num_random_g (inclusive)
                    idx_range = g_start_index:(g_start_index + num_random_g)
                    G = Matrix{Float64}(undef, size(W,2), length(idx_range))
                    indices = collect(idx_range)
                    for (col, n) in enumerate(idx_range)
                        g = deepc_random_g(W; rng=env.rng, simplex=g_simplex)
                        G[:, col] = g
                        writedlm(joinpath(hankel_dir, "g_$(n).csv"), g, ',')
                    end
                    # Compute residual costs per g against last window of [U_p; Y_p]
                    g_costs = calculate_trajectory_costs_deppi(U_p, Y_p, G; target=:last, pnorm=2, lambda_g=lambda_g)
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
                    writedlm(joinpath(hankel_dir, "g_costs_comparison.csv"), comp, ',')
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
        if cnt > num_steps
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
end

