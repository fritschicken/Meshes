using Quantica, GLMakie, StaticArrays
import Quantica.σ

# lattice setup
function _unit_coords_dice(N, d_nn)
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

function _hexagon_region(r, center, d_nn; eps = 1e-5)

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

function _multi_hexagon_region(r, centers, d_nn; eps = 1e-5)
    return any(c -> _hexagon_region(r, c, d_nn; eps = eps), centers)
end

function _dice_lattice(N, M, d_nn)
    a = sqrt(3) * d_nn
    A1 = SVector(a * cos(π/3), a * sin(π/3))
    A2 = SVector(a * cos(π/3), -a * sin(π/3))

    spec, edg, ins = _unit_coords_dice(N, d_nn)
    center_0 = (A1+A2)/2
    hex_centers = [center_0 + i * A1 + j * A2 for i in 0:(M-1), j in 0:(M-1)][:]

    lat = lattice(sublat(spec, name = :S), sublat(edg, name=:E),sublat(ins,name=:Ins), bravais = (A1, A2))
    slat = lat |> supercell(region = r -> _multi_hexagon_region(r, hex_centers, d_nn))
    return slat
end
function _unit_coords_square(N, a)
    coords = Tuple{Float64, Float64}[(0.0, 0.0)]
    for i in 1:(N - 2)
        push!(coords, (0.0, a * i / (N - 1)))
        push!(coords, (a * i / (N - 1), 0.0))
    end
    return coords
end
function _square_lattice(N, M, a)
    pts = _unit_coords_square(N, a)
    lat = lattice(sublat(pts), bravais = ([0.0, a], [a, 0.0]))
    sl = lat |> supercell(region = r -> (0 <= r[1] <= M * a) && (0 <= r[2] <= M * a))
    return sl
end

function _unit_coords_hex(N, d_nn)
    #hexagon part
    startA = SVector(0.0, -d_nn / 2)
    startB = SVector(0.0,  d_nn / 2)

    a1 = SVector(sqrt(3) / 2, -0.5)
    a2 = SVector(sqrt(3) / 2,  0.5)
    a3 = startB-startA

    coords = SVector{2, Float64}[startA, startB]

    m = d_nn .* a1 / (N - 1)
    n = d_nn .* a3 / (N - 1)
    k = d_nn .* a2 / (N - 1)

    for i in 1:(N - 2)
        pt_n = startA + i .* n
        pt_m = startA + i .* m
        pt_k = startB + i .* k
        push!(coords, pt_n, pt_m, pt_k)
    end
    return coords
end

function _hex_lattice(N, M, d_nn)
    a = sqrt(3) * d_nn
    A1 = SVector(a * cos(π/3), a * sin(π/3))
    A2 = SVector(a * cos(π/3), -a * sin(π/3))

    coords = _unit_coords_hex(N, d_nn)
    center_0 = (A1+A2)/2
    hex_centers = [center_0 + i * A1 + j * A2 for i in 0:(M-1), j in 0:(M-1)][:]

    lat = lattice(sublat(coords), bravais = (A1, A2))
    slat = lat |> supercell(region = r -> _multi_hexagon_region(r, hex_centers, d_nn))
    return slat
end

function hex_ham(N, M, d_nn, phi_sc)
    _theta_hex(r, dr, B)= σ.z .* (-B/2 * SA[-r[2], r[1]]' * dr + (phi_sc(r + dr) - phi_sc(r))/2)

    hopping_model = @hopping((r, dr; t = 1.0, B = 0.0) -> -t * cis(_theta_hex(r, dr, B)); range = d_nn/(N-1)+0.00002)
    onsite_model = @onsite((r; e = 0.0, del = 0.5) -> e *σ.z +  del * σ.x)

    lat = _hex_lattice(N, M, d_nn)
    h = hamiltonian(lat, onsite_model+hopping_model, orbitals=2)
    return h
end


function dice_ham(N, M, d_nn, phi_sc)
    _theta_hex(r, dr, B)= σ.z .* (-B/2 * SA[-r[2], r[1]]' * dr + (phi_sc(r + dr) - phi_sc(r))/2)
    l = (d_nn / (N - 1) - 0.00002, d_nn / (N - 1) + 0.00002)

    onsite_model = @onsite((r; e = 0.0, del = 0.5) -> e *σ.z +  del * σ.x)

    hopping_model=
        @hopping((r, dr; t = 1.0, B = 0.0) -> -t * cis(_theta_hex(r, dr, B)); range = l, sublats = :E .=> (:E, :S)) +
        @hopping((r, dr; t = 1.0, B = 0.0) -> -t * cis(_theta_hex(r, dr, B)); range = l, sublats = :Ins .=> (:Ins, :S))+
        @hopping((r, dr; t = 1.0, B = 0.0) -> -t * cis(_theta_hex(r, dr, B)); range = l, sublats = :S .=> (:E, :S, :Ins))
    
    lat = _dice_lattice(N, M, d_nn)
    h = hamiltonian(lat, onsite_model+hopping_model, orbitals=2)
    return h
end

function square_ham(N, M, a, phi_sc)
    theta_square(r, dr, B) =
        σ.z .* (-B * r[2] * dr[1] + (phi_sc(r + dr) - phi_sc(r)) / 2)

    onsite_model = @onsite((r; e = 0.0, del = 0.5) -> e * σ.z + del * σ.x)
    hopping_model = @hopping(
        (r, dr; t = 1.0, B = 0.0) -> -t * cis(theta_square(r, dr, B));
        range = a / (N - 1) + 0.00002,
    )

    lat = _square_lattice(N, M, a)
    h = hamiltonian(lat, onsite_model + hopping_model, orbitals = 2)
    return h
end

function save_case(name, h, flux_ratios, S; params = (; e = 0.0, t = 2.0))
    b = bands(h, flux_ratios; mapping = p_ratio -> ftuple(; params..., B = 2π * p_ratio / S))
    butterfly = qplot(b, hide=:bands, color=:black, axis = (; xlabel = "ϕ/ϕ₀"))
    mesh = qplot(h)

    save("Figures/superc/$(name)_super_butterfly.png", butterfly; px_per_unit = 1)
    save("Figures/superc/$(name)_super_mesh.png", mesh; px_per_unit = 1)
end

#example usage
d_nn = 1.0
a = 1.0
M = 3
N = 3
const r0 = (10.0 * a, 10.0 * a)
phi_sc(r) = atan(r[2] - r0[2], r[1] - r0[1])

flux_ratios = range(0.0, 1.0, length = 601)
params = (; e = 0.0, t = 2.0)

S_square = (a)^2
S_hex = (3 * sqrt(3) / 2) * (d_nn)^2
S_dice = (sqrt(3) / 2) * (d_nn)^2

save_case("square", square_ham(N, M, a, phi_sc), flux_ratios, S_square; params)
save_case("hex", hex_ham(N, M, d_nn, phi_sc), flux_ratios, S_hex; params)
save_case("dice", dice_ham(N, M, d_nn, phi_sc), flux_ratios, S_dice; params)
