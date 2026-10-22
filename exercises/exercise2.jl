using Quantica, GLMakie, StaticArrays
a = 1
N = 5
lat = LP.square() |> supercell(region = r -> ((r[1] == 0 || r[1] == N - 1) && 0 <= r[2] <= N - 1) || 
                           ((r[2] == 0 || r[2] == N - 1) && 0 <= r[1] <= N - 1))
h = lat |> (@onsite((; ϵ = 0.0) -> ϵ) + @hopping((r, dr; t = 1.0, B = 0.0) -> t * cis(-B * r[2] * dr[1]), range = 1.0)) |> transform(r -> r/(N-1))


Bs = range(-pi,pi, length=199)
params = (;ϵ=0, t=2, B=0)
b = bands(h, Bs; mapping = B ->ftuple(;params...,B))




flux_ratios = range(0.0, 1.0, length = 301)
b = bands(h, flux_ratios; mapping = p_ratio -> ftuple(; params..., B = 2π * p_ratio / (a^2)))

fig1 = qplot(b, hide=:nodes, color=:black)
fig2 = qplot(h)

save("./Figures/palq_butterfly.png", fig1)
save("./Figures/plaq.png", fig2)