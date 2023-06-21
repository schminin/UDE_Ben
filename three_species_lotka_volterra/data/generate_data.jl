using DifferentialEquations
using Plots
using DelimitedFiles

include("../reference.jl")


function dynamics!(du, u, p, t)
    r1, a1_1, a1_2, a1_3, r2, a2_1, a2_2, a2_3, r3, a3_1, a3_2, a3_3 = p
    
    du[1] = -r1*u[1] + a1_3*u[1]*u[3] + a1_2*u[1]*u[2] - a1_1*u[1]*u[1]
    du[2] = -r2*u[2] + a2_3*u[2]*u[3] - a2_1*u[2]*u[1] - a2_2*u[2]*u[2]
    du[3] = r3*u[3] - a3_2*u[3]*u[2] - a3_1*u[3]*u[1] - a3_3*u[3]*u[3]
end


# Simulate dense reference data
tspan = (0.0, 10.0)
t = tspan[1]:0.05:tspan[end]

prob = ODEProblem(dynamics!,u0, tspan, parameter_values)
sol = solve(prob, Tsit5(), reltol=1e-8, saveat = t)
sol = Array(sol)

# Store data
open(joinpath("lotka_volterra_reference.csv"), "w") do io
    header = ["t" states...]
    writedlm(io, header, ",")
    writedlm(io, [t transpose(sol)], ",")
end

# Visualize results
plt = plot(t, sol[1,:], label="u1", color=1)
plot!(t, sol[2,:], label="u2", color=2)
plot!(t, sol[3,:], label="u3", xlabel="time", color=3)
savefig(plt, joinpath("reference.png")) 

# Simulate noisy and sparse data