using Pkg
Pkg.activate(".")

include("src/MPOPIS.jl")    

using .MPOPIS
using JLD2

infile  = "H_sub_v12.jld2"   # file we just copied
outfile = "H_sub_v12.jld2"   # overwrite the copy (or change name to a new file)

@load infile X_sub

@show length(X_sub)
for i in eachindex(X_sub)
    # X_sub[i] is a matrix with columns [Vx, Vy, Ψ_dot, δ, pedal]
    X_sub[i][:, 1] .= 12.0   # set Vx to 12.0 for all timesteps
end

@save outfile X_sub=X_sub

println("Wrote ", outfile)
