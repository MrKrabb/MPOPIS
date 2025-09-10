"""
DeePC control example derived from car_example.jl

Two phases:
 1. Data collection (MPPI) to build DeePC data matrix W = [U_p; Y_p; U_f; Y_f]
 2. DeePC control phase applying u = (U_f * g_c)[1:m] where g_c is updated online

Assumptions:
 - Functions hankel_blocks_local, combine_deepc_blocks, deepc_update_nominal_g!,
   calculate_trajectory_costs_deppi already loaded (via including car_example.jl or
   being in the same module from previous load).

Usage sketch (inside Julia REPL after using MPOPIS):
  simulate_car_racing_deepc(collect_steps=400, control_steps=300, T_ini=20, N_pred=15,
      num_samples=150, horizon=50, λ=10.0, α=1.0, deepc_step=0.5)

Outputs (written to hankel_data/ by default):
  *_deepc_W.csv          DeePC data matrix
  *_deepc_g_c_final.csv  Final nominal g_c
  *_deepc_g_c_hist.csv   History of g_c (columns = time index 0..control_steps)
  *_deepc_u_applied.csv  Applied controls during DeePC control phase (m x control_steps)
  *_deepc_costs.csv      Columns: step, gc_residual_cost
"""
function simulate_car_racing_deepc(; 
    # Phase lengths
    collect_steps::Int = 400,
    control_steps::Int = 300,
    # DeePC horizons
    T_ini::Int = 20,
    N_pred::Int = 15,
    # MPPI parameters for data collection
    num_samples::Int = 150,
    horizon::Int = 50,
    λ::Float64 = 10.0,
    α::Float64 = 1.0,
    cov_mat = block_diagm([0.0625, 0.1], 1),
    U₀ = zeros(2),
    policy_type::Symbol = :mppi,
    # DeePC update parameters
    deepc_num_samples::Int = 64,
    deepc_step::Float64 = 0.5,
    deepc_λw::Float64 = 10.0,
    deepc_lambda_g::Float64 = 0.0,
    deepc_simplex::Bool = true,
    # IO
    hankel_dir::String = "hankel_data",
    hankel_prefix::String = "car",
    save_outputs::Bool = true,
    rng = Random.GLOBAL_RNG,
    verbose::Bool = true,
    save_gif::Bool = false,
    plot_traj::Bool = false,
    plot_traj_perc::Float64 = 1.0,
    text_with_plot::Bool = true,
    text_on_plot_xy = (80.0, -60.0),
    constant_velocity::Union{Nothing,Float64}=nothing,
)
    # (Removed internal `using` statements; dependencies are already imported at module level.)
    verbose && @printf("[DeePC] Phase 1: Collecting data with MPPI for %d steps...\n", collect_steps)
    # ------------------------------------------------------------------
    # 1) Environment + data collection (rollouts for DeePC data)
    # ------------------------------------------------------------------
    # Build environment
    env = CarRacingEnv(rng=MersenneTwister(), constant_velocity=constant_velocity)
    # Enable logging so sample_controls & trajectories are captured
    pol = get_policy(policy_type, env, num_samples, horizon, λ, α, U₀, cov_mat, true,
        10, # ais_its (default placeholder)
        20.0, # λ_ais
        0.8, :ss, # ce_elite_threshold, ce_Σ_est
        0.75, 0.8) # cma_σ, cma_elite_threshold

    m = action_space_size(action_space(env))
    p = length(state(env))
    L = T_ini + N_pred

    u_hist = Matrix{Float64}(undef, m, 0)
    y_hist = Matrix{Float64}(undef, p, 0)
    step_ct = 0
    while step_ct < collect_steps && !env.done
        act = pol(env)
        env(act)
        step_ct += 1
        u_hist = hcat(u_hist, isa(act, AbstractVector) ? act : vec(act))
        y_hist = hcat(y_hist, copy(state(env)))
    end

    size(u_hist,2) >= L || error("Not enough collected data (have $(size(u_hist,2)), need ≥ $(L))")

    # ------------------------------------------------------------------
    # 2) HANKEL / DeePC DATA MATRIX CREATION
    # ------------------------------------------------------------------
    # Build Hankel blocks from executed trajectory (for reference) and from rollouts for futures/pasts per sample
    U_block = hankel_blocks_local(u_hist, L)
    Y_block = hankel_blocks_local(y_hist, L)
    U_p_exec = U_block[1:m*T_ini, :]
    U_f_exec = U_block[m*T_ini+1:end, :]
    Y_p_exec = Y_block[1:p*T_ini, :]
    Y_f_exec = Y_block[p*T_ini+1:end, :]

    # Rollout-derived DeePC blocks (each sample k -> past window first T_ini steps; future window last N_pred steps)
    if pol.logger.sample_controls === nothing
        @warn "Policy logger has no sample_controls; using executed Hankel windows only."
        U_p = U_p_exec; Y_p = Y_p_exec; U_f = U_f_exec; Y_f = Y_f_exec
    else
        K = length(pol.logger.sample_controls)
        Ttot = size(pol.logger.sample_controls[1], 1)
        Ttot >= T_ini + N_pred || @warn "Rollout length $(Ttot) < T_ini+N_pred=$(T_ini+N_pred); falling back to executed for missing parts."
        U_p = Matrix{Float64}(undef, m*T_ini, K)
        U_f = Matrix{Float64}(undef, m*N_pred, K)
        Y_p = Matrix{Float64}(undef, p*T_ini, K)
        Y_f = Matrix{Float64}(undef, p*N_pred, K)
        has_traj = pol.logger.trajectories !== nothing && length(pol.logger.trajectories) == K
        for kk in 1:K
            Ulog = pol.logger.sample_controls[kk]
            # Past
            if Ttot >= T_ini
                U_p[:, kk] = vec(Ulog[1:T_ini, :])
            else
                U_p[:, kk] = vec(vcat(Ulog, repeat(Ulog[end:end, :], T_ini - Ttot, 1)))
            end
            # Future
            if Ttot >= N_pred
                U_f[:, kk] = vec(Ulog[Ttot-N_pred+1:Ttot, :])
            else
                U_f[:, kk] = vec(vcat(repeat(Ulog[1:1, :], N_pred - Ttot, 1), Ulog))
            end
            if has_traj
                Ylog = pol.logger.trajectories[kk]
                if Ttot >= T_ini
                    Y_p[:, kk] = vec(Ylog[1:T_ini, :])
                else
                    Y_p[:, kk] = vec(vcat(Ylog, repeat(Ylog[end:end, :], T_ini - Ttot, 1)))
                end
                if Ttot >= N_pred
                    Y_f[:, kk] = vec(Ylog[Ttot-N_pred+1:Ttot, :])
                else
                    Y_f[:, kk] = vec(vcat(repeat(Ylog[1:1, :], N_pred - Ttot, 1), Ylog))
                end
            else
                Y_p[:, kk] = Y_p_exec[:, end]
                Y_f[:, kk] = Y_f_exec[:, end]
            end
        end
    end

    # Combine into DeePC data matrix W = [U_p; Y_p; U_f; Y_f]
    _, _, W = combine_deepc_blocks(U_p, U_f, Y_p, Y_f)  # <- (Hankel / data matrix done)

    # ------------------------------------------------------------------
    # 3) DEFINE NOMINAL g_c
    # ------------------------------------------------------------------
    ncols = size(W,2)
    g_c = ones(Float64, ncols)         # Initial nominal DeePC combination vector g_c
    g_hist = Matrix{Float64}(undef, ncols, control_steps+1)
    g_hist[:, 1] = g_c                 # Log initial g_c

    # Phase 2: DeePC control
    verbose && @printf("[DeePC] Phase 2: DeePC control for %d steps...\n", control_steps)
    env2 = CarRacingEnv(rng=MersenneTwister(), constant_velocity=constant_velocity)
    # Warm-up using MPPI for T_ini steps to seed past buffers
    pol2 = get_policy(policy_type, env2, num_samples, horizon, λ, α, U₀, cov_mat, true,
        10, 20.0, 0.8, :ss, 0.75, 0.8)
    u_buf = zeros(Float64, m, 0)
    y_buf = zeros(Float64, p, 0)
    warm = 0
    while warm < T_ini && !env2.done
        act = pol2(env2)
        env2(act)
        u_buf = hcat(u_buf, isa(act, AbstractVector) ? act : vec(act))
        y_buf = hcat(y_buf, copy(state(env2)))
        warm += 1
    end
    size(u_buf,2) == T_ini || error("Could not collect full warm-up window (have $(size(u_buf,2)), need $(T_ini))")

    function flatten_past(U::AbstractMatrix)
        # U: m x T_ini -> vector variable-wise blocks each of length T_ini
        return vcat([vec(U[i, :]) for i in 1:size(U,1)]...)
    end

    u_applied = Matrix{Float64}(undef, m, control_steps)
    gc_costs = zeros(Float64, control_steps)

    # Optional GIF animation setup
    anim = save_gif ? Animation() : nothing

    for t in 1:control_steps
        # Form past vectors
        u_past_vec = flatten_past(u_buf)
        y_past_vec = flatten_past(y_buf)
        # ------------------------------------------------------------------
        # 4) SAMPLE CANDIDATE g's & UPDATE g_c
        #    deepc_update_nominal_g! internally:
        #      - samples 'deepc_num_samples' candidate g vectors (Dirichlet / Gaussian)
        #      - evaluates residual cost ||[U_p;Y_p] g - [u_past;y_past]||
        #      - computes softmin weights (λ_w) and updates g_c ← (1-step)*g_c + step*g_hat
        # ------------------------------------------------------------------
        deepc_update_nominal_g!(g_c, U_p, Y_p, u_past_vec, y_past_vec;
            num_samples=deepc_num_samples, λ_w=deepc_λw, lambda_g=deepc_lambda_g,
            simplex=deepc_simplex, step=deepc_step, rng=rng, project_sum1=true)
        g_hist[:, t+1] = g_c
        # ------------------------------------------------------------------
        # 5) EXTRACT CONTROL INPUT FROM g_c
        #    Future input sequence = U_f * g_c  (stacked m*N_pred vector)
        #    Apply only the first control (receding horizon): u_now = (U_f * g_c)[1:m]
        # ------------------------------------------------------------------
        u_future = U_f * g_c              # length m*N_pred (predicted future inputs)
        u_now = u_future[1:m]             # first m entries used as current control
        env2(u_now)
        u_applied[:, t] = u_now
        # Observe new state and update buffers (drop oldest column)
        y_new = copy(state(env2))
        u_buf = hcat(u_buf[:, 2:end], u_now)
        y_buf = hcat(y_buf[:, 2:end], y_new)
        # Compute residual cost of updated g_c
        gc_costs[t] = calculate_trajectory_costs_deppi(U_p, Y_p, reshape(g_c, :, 1); u_past=u_past_vec, y_past=y_past_vec, pnorm=2, lambda_g=deepc_lambda_g)[1]
        # Frame capture
        if save_gif
            pplt = plot_traj ? plot(env2, pol2, plot_traj_perc, text_output=text_with_plot, text_xy=text_on_plot_xy) : plot(env2, text_output=text_with_plot, text_xy=text_on_plot_xy)
            frame(anim, pplt)
        end
        env2.done && break
    end

    if save_outputs
        isdir(hankel_dir) || mkpath(hankel_dir)
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deepc_W.csv"), W, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deepc_g_c_final.csv"), g_c, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deepc_g_c_hist.csv"), g_hist, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deepc_u_applied.csv"), u_applied, ',')
        open(joinpath(hankel_dir, "$(hankel_prefix)_deepc_costs.csv"), "w") do io
            println(io, "step,g_c_residual_cost")
            for t in 1:control_steps
                println(io, string(t, ",", gc_costs[t]))
            end
        end
    end

    if save_gif && anim !== nothing
        gif_name = "$(hankel_prefix)_deepc.gif"
        verbose && println("[DeePC] Saving gif..." * gif_name)
        gif(anim, gif_name, fps=10)
    end

    return (W=W, g_c=g_c, g_hist=g_hist, u_applied=u_applied, gc_costs=gc_costs)
end

"""
    simulate_deppi_car(; T_ini=20, N_pred=15, horizon=50, steps=300, num_samples=150, λ=10.0, α=1.0,
        U₀=zeros(2), cov_mat=block_diagm([0.0625, 0.1], 1),
        deepc_num_samples=64, deepc_step=0.5, deepc_λw=10.0, deepc_lambda_g=0.0, deepc_simplex=true,
        hankel_dir="hankel_data", hankel_prefix="car", save_outputs=true, rng=MersenneTwister(), verbose=true)

Minimal DeePPi (DeePC) driver:
 - Executes MPPI only long enough to gather T_ini executed steps (warm-up) and log rollouts once (horizon steps).
 - Builds W = [U_p; Y_p; U_f; Y_f] from sampled rollouts (first T_ini & last N_pred per sample).
 - Runs DeePC control for `steps` steps updating g_c each step and applying u from U_f * g_c.

Requirements: horizon ≥ T_ini + N_pred.
Returns: Named tuple (W, g_hist, u_applied, gc_costs, g_c).
"""
function simulate_deppi_car(; T_ini=20, N_pred=15, horizon=50, steps=300, num_samples=150, λ=10.0, α=1.0,
    U₀=zeros(2), cov_mat=block_diagm([0.0625, 0.1], 1),
    deepc_num_samples=64, deepc_step=0.5, deepc_λw=10.0, deepc_lambda_g=0.0, deepc_simplex=true,
    hankel_dir="hankel_data", hankel_prefix="car", save_outputs=true, rng=MersenneTwister(), verbose=true,
    save_gif::Bool=false, save_traj_png::Union{Nothing,String}=nothing, plot_traj::Bool=false, plot_traj_perc::Float64=1.0, text_with_plot::Bool=true, text_on_plot_xy=(80.0,-60.0), constant_velocity::Union{Nothing,Float64}=nothing, show_progress::Bool=true, deepc_temperature::Union{Nothing,Float64}=nothing,
    deepc_rollout_lambda::Union{Nothing,Float64}=nothing,        # Override λ only for initial MPPI rollouts (data collection)
    deepc_rollout_cov_scale::Float64=1.0,                        # Scale factor for covariance during data collection only
    deepc_use_saved_W::Bool=false,                              # If true, load pre-saved combined W instead of building from rollouts
    deepc_W_file::Union{Nothing,String}=nothing                 # Path to CSV file with W (rows = [U_p;Y_p;U_f;Y_f])
 )
    # Validate horizon
    horizon >= T_ini + N_pred || error("horizon must be ≥ T_ini+N_pred")
    if deepc_temperature !== nothing
        deepc_λw = deepc_temperature
    end
    verbose && println("[DeePPi] Softmin temperature λ_w = $(deepc_λw)")

    # Environment and rollout policy (for data collection + warm start)
    env = CarRacingEnv(rng=MersenneTwister(), constant_velocity=constant_velocity)
    λ_roll = deepc_rollout_lambda === nothing ? λ : deepc_rollout_lambda
    cov_roll = deepc_rollout_cov_scale == 1.0 ? cov_mat : (cov_mat .* deepc_rollout_cov_scale)
    pol = get_policy(:mppi, env, num_samples, horizon, λ_roll, α, U₀, cov_roll, true, 10, 20.0, 0.8, :ss, 0.75, 0.8)
    verbose && deepc_rollout_lambda !== nothing && println("[DeePPi] Rollout λ override: $(λ) -> $(λ_roll)")
    verbose && deepc_rollout_cov_scale != 1.0 && println("[DeePPi] Rollout covariance scaled by $(deepc_rollout_cov_scale)")

    m = action_space_size(action_space(env))
    p = length(state(env))

    # 1) Collect executed past window (warm-up)
    u_buf = zeros(Float64, m, 0)
    y_buf = zeros(Float64, p, 0)
    while size(u_buf,2) < T_ini && !env.done
        show_progress && print("[DeePPi] Warm-up $(size(u_buf,2)+1)/$T_ini\r")
        act = pol(env)
        env(act)
        u_buf = hcat(u_buf, isa(act, AbstractVector) ? act : vec(act))
        y_buf = hcat(y_buf, copy(state(env)))
    end
    show_progress && println("[DeePPi] Warm-up complete.")
    size(u_buf,2) == T_ini || error("Could not warm-up to T_ini steps")

    # 2) Build or load DeePC data matrices
    if deepc_use_saved_W
        deepc_W_file === nothing && error("deepc_use_saved_W=true but deepc_W_file not provided")
        isfile(deepc_W_file) || error("Provided deepc_W_file does not exist: $(deepc_W_file)")
        verbose && println("[DeePPi] Loading pre-saved W from $(deepc_W_file)")
        W_raw = readdlm(deepc_W_file, ',')
        W = Matrix{Float64}(W_raw)
        expected_rows = m*T_ini + p*T_ini + m*N_pred + p*N_pred
        size(W,1) == expected_rows || error("Loaded W row count $(size(W,1)) ≠ expected $(expected_rows) for (m,p,T_ini,N_pred)=($(m),$(p),$(T_ini),$(N_pred))")
        # Split rows into blocks
        r1 = 1; r2 = m*T_ini
        U_p = W[r1:r2, :]
        r1 = r2 + 1; r2 = r1 + p*T_ini - 1
        Y_p = W[r1:r2, :]
        r1 = r2 + 1; r2 = r1 + m*N_pred - 1
        U_f = W[r1:r2, :]
        r1 = r2 + 1; r2 = r1 + p*N_pred - 1
        Y_f = W[r1:r2, :]
    else
        pol.logger.sample_controls === nothing && error("Policy logging off; enable pol_log=true")
        K = length(pol.logger.sample_controls)
        Ttot = size(pol.logger.sample_controls[1], 1)
        Ttot >= T_ini + N_pred || @warn "Rollout length $(Ttot) < T_ini+N_pred; padding may occur"
        U_p = Matrix{Float64}(undef, m*T_ini, K)
        U_f = Matrix{Float64}(undef, m*N_pred, K)
        Y_p = Matrix{Float64}(undef, p*T_ini, K)
        Y_f = Matrix{Float64}(undef, p*N_pred, K)
        has_traj = pol.logger.trajectories !== nothing && length(pol.logger.trajectories) == K
        for kk in 1:K
            Ulog = pol.logger.sample_controls[kk]
            if size(Ulog,1) >= T_ini
                U_p[:,kk] = vec(Ulog[1:T_ini, :])
            else
                U_p[:,kk] = vec(vcat(Ulog, repeat(Ulog[end:end,:], T_ini-size(Ulog,1),1)))
            end
            if size(Ulog,1) >= N_pred
                U_f[:,kk] = vec(Ulog[end-N_pred+1:end, :])
            else
                U_f[:,kk] = vec(vcat(repeat(Ulog[1:1,:], N_pred-size(Ulog,1),1), Ulog))
            end
            if has_traj
                Ylog = pol.logger.trajectories[kk]
                if size(Ylog,1) >= T_ini
                    Y_p[:,kk] = vec(Ylog[1:T_ini, :])
                else
                    Y_p[:,kk] = vec(vcat(Ylog, repeat(Ylog[end:end,:], T_ini-size(Ylog,1),1)))
                end
                if size(Ylog,1) >= N_pred
                    Y_f[:,kk] = vec(Ylog[end-N_pred+1:end, :])
                else
                    Y_f[:,kk] = vec(vcat(repeat(Ylog[1:1,:], N_pred-size(Ylog,1),1), Ylog))
                end
            else
                Y_p[:,kk] = repeat(y_buf[:,end:end], T_ini, 1)[:]
                Y_f[:,kk] = repeat(y_buf[:,end:end], N_pred, 1)[:]
            end
        end
        _, _, W = combine_deepc_blocks(U_p, U_f, Y_p, Y_f)
    end

    # 3) Initialize g_c and logging structures
    ncols = size(W,2)
    g_c = ones(Float64, ncols)
    g_hist = Matrix{Float64}(undef, ncols, steps+1)
    g_hist[:,1] = g_c
    u_applied = Matrix{Float64}(undef, m, steps)
    gc_costs = zeros(Float64, steps)

    # Executed path storage (capture x,y from state if available)
    path_x = Float64[]; path_y = Float64[]
    if p >= 2
        push!(path_x, y_buf[1,end]); push!(path_y, y_buf[2,end])
    end

    flatten_past(A) = vcat([vec(A[i,:]) for i in 1:size(A,1)]...)

    anim = save_gif ? Animation() : nothing
    for t in 1:steps
        if show_progress && (t == 1 || t % 10 == 0 || t == steps)
            print("[DeePPi] Control step $t/$steps  \r")
        end
        u_past_vec = flatten_past(u_buf)
        y_past_vec = flatten_past(y_buf)
        deepc_update_nominal_g!(g_c, U_p, Y_p, u_past_vec, y_past_vec; num_samples=deepc_num_samples,
            λ_w=deepc_λw, lambda_g=deepc_lambda_g, simplex=deepc_simplex, step=deepc_step, rng=rng, project_sum1=true)
        g_hist[:, t+1] = g_c
        u_future = U_f * g_c
        u_now = u_future[1:m]
        env(u_now)
        u_applied[:, t] = u_now
        y_new = copy(state(env))
        u_buf = hcat(u_buf[:,2:end], u_now)
        y_buf = hcat(y_buf[:,2:end], y_new)
        if p >= 2
            push!(path_x, y_new[1]); push!(path_y, y_new[2])
        end
        gc_costs[t] = calculate_trajectory_costs_deppi(U_p, Y_p, reshape(g_c, :, 1); u_past=u_past_vec, y_past=y_past_vec, pnorm=2, lambda_g=deepc_lambda_g)[1]
        if save_gif
            pplt = plot_traj ? plot(env, pol, plot_traj_perc, text_output=text_with_plot, text_xy=text_on_plot_xy) : plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)
            frame(anim, pplt)
        end
        env.done && break
    end

    if save_outputs
        isdir(hankel_dir) || mkpath(hankel_dir)
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deppi_W.csv"), W, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deppi_g_c_final.csv"), g_c, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deppi_g_c_hist.csv"), g_hist, ',')
        writedlm(joinpath(hankel_dir, "$(hankel_prefix)_deppi_u_applied.csv"), u_applied, ',')
        open(joinpath(hankel_dir, "$(hankel_prefix)_deppi_costs.csv"), "w") do io
            println(io, "step,g_c_residual_cost")
            for t in 1:steps
                println(io, string(t, ",", gc_costs[t]))
            end
        end
    end
    if save_gif && anim !== nothing
        gif_name = "$(hankel_prefix)_deppi.gif"
        verbose && println("[DeePPi] Saving gif..." * gif_name)
        gif(anim, gif_name, fps=10)
    end
    if save_traj_png !== nothing
        # Final snapshot: always base environment, then overlay executed path (line)
        pplt_final = plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)
        if !isempty(path_x)
            try
                plot!(pplt_final, path_x, path_y, color=:red, lw=2, label="executed path")
            catch e
                @warn "Failed to overlay executed path: $(e)"
            end
        end
        try
            savefig(pplt_final, save_traj_png)
            verbose && println("[DeePPi] Saved trajectory PNG => " * save_traj_png)
        catch e
            @warn "Failed to save trajectory PNG: $(e)"
        end
    end
    show_progress && println("\n[DeePPi] Control loop complete.")
    verbose && @printf("[DeePPi] Completed %d steps. Final residual=%.4f\n", steps, gc_costs[min(steps,end)])
    return (W=W, g_c=g_c, g_hist=g_hist, u_applied=u_applied, gc_costs=gc_costs)
end
