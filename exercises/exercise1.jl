using Quantica
using GLMakie
using CairoMakie

lat = LP.linear()
ϵ = 1
t = 2
model = onsite(ϵ) - hopping(t)
h = hamiltonian(lat, model)
ϕpoints = range(-π, π, 199)
b = bands(h, ϕpoints)
plot =qplot(b, hide = :nodes)