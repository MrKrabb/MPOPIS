#!/usr/bin/env julia
# hg_demo.jl
# Minimal demonstration of H * g multiplication used in the project
# Shows layout and exactly where timestep/action entries live in the product
using Pkg
Pkg.activate(".")

include("src/MPOPIS.jl")    

using .MPOPIS
using JLD2
using LinearAlgebra
using Printf
using Clustering, Distances, Statistics
using Random

# small example sizes
T = 3      # horizon (timesteps)
as = 2     # action dims per timestep
N = 4      # number of stored trajectories (columns in H)
K = 3      # number of sampled rollouts (columns in g)

# Build full stored trajectory matrix H: each h is T x P where
# columns 1:ss are state components and columns (ss+1):(ss+as) are actions
ss = 3                       # state dimension (matches original file's state_cols = 1:3)
P = ss + as                  # total columns per stored trajectory (states + actions)

# Fill H with readable numbers so mapping is obvious
# We'll make h[t, p, i] = 100*i + 10*t + p so each column has unique values
H = Array{Float64}(undef, T, P, N)
for i in 1:N
    for p in 1:P
        for t in 1:T
            H[t, p, i] = 100*i + 10*t + p
        end
    end
end

println("H shape: ", size(H), "  (T, P, N), with P = ss+as = $(ss)+$(as)")
println("\nH[:,:,1] (trajectory 1) -- columns: [state1,state2,state3,act1,act2]:")
println(H[:,:,1])

# Define the same indexing used in the original file
state_cols = 1:ss
action_cols = (ss+1):(ss+as)

# Extract H_states and H_actions like the project code does
H_states = Array{Float64}(undef, T, ss, N)
H_actions = Array{Float64}(undef, T, as, N)
for i in 1:N
    H_states[:, :, i] = H[:, state_cols, i]
    H_actions[:, :, i] = H[:, action_cols, i]
end

println("\nH_states[:,:,1] (states for trajectory 1):")
println(H_states[:,:,1])
println("\nH_actions[:,:,1] (actions for trajectory 1):")
println(H_actions[:,:,1])

# Linearize to matrices for multiplication: (T*ss) x N and (T*as) x N
Hmat_states = reshape(H_states, T * ss, N)
Hmat_actions = reshape(H_actions, T * as, N)
println("\nHmat_states shape: ", size(Hmat_states), "  (T*ss, N)")
println("Hmat_actions shape: ", size(Hmat_actions), "  (T*as, N)")

# Build a toy selector matrix g (N x K)
g = [ 1.0  0.0 -1.0;
      0.0  1.0  1.0;
      1.0  1.0  0.0;
     -1.0  0.5  0.5 ]
println("\n g (N x K) =")
println(g)

# Multiply to get sampled (flattened) states and actions
S_states = Hmat_states * g   # (T*ss) x K
S_actions = Hmat_actions * g # (T*as) x K
println("\nS_states = Hmat_states * g  (T*ss x K):")
println(S_states)
println("\nS_actions = Hmat_actions * g  (T*as x K):")
println(S_actions)

# Labels for state and action components (same names as in original code comments)
state_labels = ["x1", "x2", "x3"]
action_labels = ["u1", "u2"]

# For each sampled rollout k, reshape back and print labeled rows
for k in 1:K
    sampled_states_k = reshape(S_states[:, k], T, ss)
    sampled_actions_k = reshape(S_actions[:, k], T, as)
    println("\n--- Sampled rollout k=$k ---")
    println("sampled_states (T x ss) with columns ", state_labels, ":")
    println(sampled_states_k)
    println("sampled_actions (T x as) with columns ", action_labels, ":")
    println(sampled_actions_k)

    # show next input and second input labeled
    println("next state (t=1) for k=$k -> ")
    for a in 1:ss
        println("  ", state_labels[a], " = ", sampled_states_k[1, a])
    end
    println("next input (t=1) for k=$k -> ")
    for a in 1:as
        println("  ", action_labels[a], " = ", sampled_actions_k[1, a])
    end

    println("second state (t=2) for k=$k -> ")
    for a in 1:ss
        println("  ", state_labels[a], " = ", sampled_states_k[2, a])
    end
    println("second input (t=2) for k=$k -> ")
    for a in 1:as
        println("  ", action_labels[a], " = ", sampled_actions_k[2, a])
    end

    # show exact indices mapping in S_actions for clarity
    println("Mapping of S_actions rows to (t,a) for this k (row_index => (t,a)):")
    for a in 1:as
        for t in 1:T
            idx = (a-1)*T + t
            @printf("  S_actions[%2d, %d] => sampled_actions[%d, %d] (%s) = %g\n", idx, k, t, a, action_labels[a], S_actions[idx, k])
        end
    end

    # show how to extract next input directly from S_actions without reshape
    next_input_direct = [ S_actions[(a-1)*T + 1, k] for a in 1:as ]
    println("next_input_direct (from S_actions): ", next_input_direct)
end

println("\nDone.")
