include("MPOPIS.jl")    

using .MPOPIS
using JLD2
using LinearAlgebra
using Clustering, Distances, Statistics

"""
    subsample_kmeans_traj(X::Vector{Matrix{Float64}}; k)

Führt ein K-Means-Clustering über Trajektorien durch (alle gleich lang)
und wählt pro Cluster eine repräsentative Trajektorie.
"""
function subsample_kmeans_traj(X::Vector{Matrix{Float64}}; k::Int)
    N = length(X)
    T = size(X[1], 2)
    d = size(X[1], 1)

    # Überprüfen, ob alle gleich lang sind
    @assert all(size(x) == (d, T) for x in X) "Alle Trajektorien müssen gleiche Dimension haben."

    # Jede Trajektorie in Vektorform bringen
    X_flat = hcat(vec.(X)...)

    # Clustering im abgeflachten Raum
    res = kmeans(X_flat, k; maxiter=100, init=:rand)

    # Pro Cluster die repräsentativste Trajektorie auswählen
    reps = Int[]
    for i in 1:k
        inds = findall(res.assignments .== i)
        if isempty(inds)
            continue
        end
        D = pairwise(SqEuclidean(), X_flat[:, inds], res.centers[:, i:i])
        push!(reps, inds[argmin(D)])
    end

    return X[reps]
end

horizon=50
num_samples = 50

simulate_car_racing(save_gif=true, plot_traj=true, policy_type=:mppi,horizon=horizon,num_samples = num_samples)
@load "trajectories.jld2" trajectories

H=Vector{Matrix{Float64}}()
for j in 1:length(trajectories)
    for k in 1:length(trajectories[j])
        #println("Trajectory $j, Sample $k:")
        push!(H, trajectories[j][k][:,4:8])  
    end
end

X_sub = subsample_kmeans_traj(H; k=300)
@save "H_sub.jld2" X_sub=X_sub

# @save "H.jld2" H=H

simulate_car_racing(save_gif=true, plot_traj=true, policy_type=:data,horizon=horizon,num_samples = num_samples)



