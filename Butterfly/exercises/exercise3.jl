using Quantica, GLMakie, StaticArrays
a = 1.0
N = 5
M = 15

function unit_coords(N, a)
    coords = Tuple{Float64, Float64}[(0.0, 0.0)]
    for i in 1:(N - 2)
        push!(coords, (0.0, a * i / (N - 1)))
        push!(coords, (a * i / (N - 1), 0.0))
    end
    return coords
end

pts = unit_coords(N, a)
lat = lattice(sublat(pts), bravais = ([0.0, a], [a, 0.0]))
sl = lat |> supercell(region = r -> (0 <= r[1] <= M * a) && (0 <= r[2] <= M * a))
h = sl |> (@onsite((; e = 0.0) -> e) - @hopping((r, dr; t = 1.0, B = 0.0) -> t * cis(-B * r[2] * dr[1])))

flux_ratios = range(0.0, 1.0, length = 301)
params = (; e = -1.5, t = 1.0)

b = bands(h, flux_ratios; mapping = p_ratio -> ftuple(; params..., B = 2π * p_ratio / (a^2)))

edge_tol = a / (N - 1) / 2
is_bulk(r) = edge_tol < r[1] < M * a - edge_tol && edge_tol < r[2] < M * a - edge_tol
bulk_orbitals = findall(is_bulk, sites(lattice(h)))
bulk_weight(ψ, ϵ, ϕs) = (sum(abs2, ψ[bulk_orbitals]) / sum(abs2, ψ))^10

fig1 = qplot(
    b;
    hide = :bands,
    color = :black,
    axis = (; xlabel = "ϕ/ϕ₀"),
    opacity = ranged(bulk_weight, (0.0, 1.0)),
)

fig2 = qplot(
    h;
    siteopacity = (i, r) -> is_bulk(r) ? 1.0 : 0.0,
)

save("./Figures/square_butterfly.png", fig1)
save("./Figures/square_mesh.png", fig2)
