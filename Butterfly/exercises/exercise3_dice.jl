using Quantica, GLMakie, StaticArrays, LinearAlgebra

"""
    unit_coords(N, d_nn)

Build the positions inside one periodic unit cell for the dice-lattice
construction.

Returns three groups of sites: `special` (the three corner/central sites), `edge`, inside`.

`N`: the number of subdivisions along each nearest-neighbor bond
`d_nn`: the nearest-neighbor distance
"""
function unit_coords(N, d_nn)
    # Reference vertical nearest-neighbor bond.
    startA = SVector(0.0, -d_nn / 2)
    startB = SVector(0.0,  d_nn / 2)

    # Bond directions used to draw the triangular pieces of the unit cell.
    d1 = d_nn .* SVector(sqrt(3) / 2, -0.5)
    d2 = d_nn .* SVector(sqrt(3) / 2,  0.5)
    d3 = d_nn .* startB-startA

    # Third special point of the triangular unit-cell motif.
    center = startB + d1

    edge = SVector{2, Float64}[]
    special = SVector{2, Float64}[startA, startB, center]
    inside = SVector{2, Float64}[]

    # Step vectors for subdividing each nearest-neighbor segment.
    a1 = d1 / (N - 1)
    a3 = d3 / (N - 1)
    a2 = d2 / (N - 1)

    for i in 1:(N - 2)
        # Outer-edge subdivision points.
        pt_n = startA + i .* a3
        pt_m = startA + i .* a1
        pt_k = startB + i .* a2

        # Inner subdivision points mirrored from the center site.
        pt_p = center - i .* a1
        pt_q = center - i .* a3
        pt_r = center + i .* a2
        push!(edge, pt_n, pt_m, pt_k) 
        push!(inside, pt_p, pt_q, pt_r)
    end
    return special, edge, inside
end

function hexagon_region(r, center, d_nn; eps = 1e-5)

    dx = r[1] - center[1]
    dy = r[2] - center[2]

    # Distance from the center to an edge, with a small tolerance.
    r_in = d_nn * (sqrt(3) / 2) + eps

    # Projections along the three independent edge normals.
    c1 = abs(dx)
    c2 = abs(0.5 * dx + (sqrt(3) / 2) * dy)
    c3 = abs(-0.5 * dx + (sqrt(3) / 2) * dy)

    return (c1 <= r_in) && (c2 <= r_in) && (c3 <= r_in)
end

function multi_hexagon_region(r, centers, d_nn; eps = 1e-5)
    return any(c -> hexagon_region(r, c, d_nn; eps = eps), centers)
end


# Geometry parameters.
d_nn = 1.0
a = sqrt(3) * d_nn
M = 12
N = 3
A1 = SVector(a * cos(π/3), a * sin(π/3))
A2 = SVector(a * cos(π/3), -a * sin(π/3))

# Build the infinite periodic lattice from the three site groups and clipping it 
spec, edg, ins = unit_coords(N, d_nn)
lat = lattice(sublat(spec, name = :S), sublat(edg, name=:E),sublat(ins,name=:Ins), bravais = (A1, A2))

center_0 = (A1+A2)/2
hex_centers = [center_0 + i * A1 + j * A2 for i in 0:(M-1), j in 0:(M-1)][:]

slat = lat |> supercell(region = r -> multi_hexagon_region(r, hex_centers, d_nn))

# Symmetric-gauge Peierls phase
peierls(B, r, dr) = cis(0.5 * B * (r[1] * dr[2] - r[2] * dr[1]))

# Narrow hopping shell around the subdivided nearest-neighbor spacing.
l = (d_nn/(N-1)-0.00002,d_nn/(N-1)+0.00002)

# Tight-binding model:
# - onsite energy on every site,
# - edge-edge and inside-inside hopping chains,
# - bidirectional couplings between special sites and the two subdivided groups.
model = 
    @onsite((; e = 0.0) -> e)+
    @hopping((r, dr; t = 1.0, B = 0.0) -> -t * peierls(B, r, dr), range = l, sublats = :E .=> (:E, :S)) +
    @hopping((r, dr; t = 1.0, B = 0.0) -> -t * peierls(B, r, dr), range = l, sublats = :Ins .=> (:Ins, :S))+
    @hopping((r, dr; t = 1.0, B = 0.0) -> -t * peierls(B, r, dr), range = l, sublats = :S .=> (:E, :S, :Ins))
h = slat |>  model

# Sweep magnetic flux and plot the Hofstadter spectrum.
flux_ratios = range(0.0, 1.0, length = 301)
S_rom = (sqrt(3) / 2) * (d_nn^2)
mesh = qplot(h)

# Save the spectrum and a real-space view of the final finite lattice/model.
ϵ = 0
params = (; e = ϵ, t = 1)
b = bands(h, flux_ratios; mapping = p_ratio -> ftuple(; params..., B = 2π* p_ratio / S_rom))


center_0 = (A1+A2)
hex_centers = [center_0 + i * A1 + j * A2 for i in 0:(M-2), j in 0:(M-2)][:]
is_bulk(r) = multi_hexagon_region(r, hex_centers, d_nn)
bulk_orbitals = findall(is_bulk, sites(lattice(h)))
bulk_weight(ψ, ϵ, ϕs) = (sum(abs2, ψ[bulk_orbitals]) / sum(abs2, ψ))

fig1 = qplot(
    b;
    hide = :bands,
    color = :black,
    axis = (; xlabel = "ϕ/ϕ₀"),
)

fig2 = qplot(
    h(B=0.2),
    siteopacity = (i, r) -> is_bulk(r) ? 1.0 : 0.0,
)
save("Figures/metal/dice_butterfly.png", fig1)


save("Figures/metal/dice_mesh.png", fig2)

