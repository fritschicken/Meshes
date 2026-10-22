using Quantica, GLMakie, StaticArrays, LinearMaps
using CairoMakie
using ArnoldiMethod
using Arpack
using Printf
using JLD2
using Dates
using ColorSchemes

includet("../arXiv_2506.15502/code.jl")

function data_figure2(nev = 8, npoints = 30; params...)
    (; hN, ϵ1) = build(; params...);
    _, psi = lowest_eigenpair(hN(; params...), nev)
    ns = [range(0, 0.49999, npoints+1); range(0.5000001, 1.49999, 2npoints + 1)]
    solver = ES.ShiftInvert(ES.Arpack(nev = 4+nev), 0)
    b = bands(hN, ns; mapping = n -> ftuple(; params..., n), solver)
    Δ = NamedTuple(params).Δ
    params´ = (; params..., Δ0 = Δ, Δ1 = 0.9*Δ)
    nt = NamedTuple(params);
    return (; hN, psi, ns, b, ϵ1, params = nt)
end

function free_energy(h_inst, solver)
    sp = spectrum(h_inst; solver = solver)
    # E_GS = 0.5 * sum of negative BdG eigenvalues
    return 0.5 * sum(filter(<(0), real(energies(sp))))
end

function fourier(F_vals, phases, max_harmonics, N)
    # ---------------------------------------------------------
    # Discrete Fourier Decomposition of F(ϕ)
    # F(ϕ) = A0 + ∑ [ A_k cos(kϕ) + B_k sin(kϕ) ]
    # ---------------------------------------------------------
    A0 = sum(F_vals) / N
    A_k = zeros(Float64, max_harmonics)
    B_k = zeros(Float64, max_harmonics)

    for k in 1:max_harmonics
        A_k[k] = (2.0 / N) * sum(F_vals .* cos.(k .* phases))
        B_k[k] = (2.0 / N) * sum(F_vals .* sin.(k .* phases))
    end

    # ---------------------------------------------------------
    # Supercurrent Harmonics: I(ϕ) = 2 * ∂F/∂ϕ (ħ = e = 1)
    # ∂F/∂ϕ = ∑ k * [ -A_k sin(kϕ) + B_k cos(kϕ) ]
    # I(ϕ) = ∑ [ I_k^(s) sin(kϕ) + I_k^(c) cos(kϕ) ]
    # ---------------------------------------------------------
    I_sin = -2.0 .* (1:max_harmonics) .* A_k
    I_cos =  2.0 .* (1:max_harmonics) .* B_k
    I_amp = sqrt.(I_sin.^2 .+ I_cos.^2)

    # Current evaluated from reconstructed Fourier components
    I_recon = zeros(Float64, N)
    for k in 1:max_harmonics
        I_recon .+= I_sin[k] .* sin.(k .* phases) .+ I_cos[k] .* cos.(k .* phases)
    end

    # Numerical central difference derivative of F (for cross-validation)
    Δϕ = 2π / N
    I_numeric = zeros(Float64, N)
    for i in 1:N
        ip = mod1(i + 1, N)
        im = mod1(i - 1, N)
        I_numeric[i] = 2.0 * (F_vals[ip] - F_vals[im]) / (2.0 * Δϕ)
    end
    return I_numeric, I_recon, A0, A_k, B_k, I_sin, I_cos, I_amp
end

function detailed_fourier_analysis(hN;
        npoints = 128, max_harmonics = 12, solver,
        savepath = joinpath(@__DIR__, "figures/fourier_"), params...)

    phases = collect(range(0, 2π, length = npoints + 1)[1:end-1])
    N = length(phases)
    parameter_values = (; params...)

    println("Calculating free energy across $(N) phase points...")
    println("System parameters:")
    show(stdout, MIME"text/plain"(), parameter_values)
    println("\nFourier settings: npoints = $npoints, max_harmonics = $max_harmonics")
    F_vals = zeros(Float64, N)
    for (i, ϕ_val) in enumerate(phases)
        h_at_ϕ = hN(; params..., ϕ = ϕ_val)
        F_vals[i] = free_energy(h_at_ϕ, solver)
    end

    I_numeric, I_recon, A0, A_k, B_k, I_sin, I_cos, I_amp = fourier(F_vals, phases, max_harmonics, N)

    # ---------------------------------------------------------
    # Visualization Dashboard
    # ---------------------------------------------------------
    fig = Figure(size = (1000, 600))

    ax1 = Axis(fig[1, 1],
        xlabel = "Phase difference, ϕ",
        ylabel = "Ground-state free energy [meV]",
        title = "Ground-State Free Energy and Current vs Phase",
        xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]),
        ytickformat = values -> [@sprintf("%.3g", value) for value in values])
    free_energy_line = lines!(ax1, phases, F_vals;
        color = :royalblue, linewidth = 2)

    ax2 = Axis(fig[1, 1],
        ylabel = "Josephson current [meV]",
        yaxisposition = :right,
        xaxisposition = :top,
        backgroundcolor = :transparent,
        ytickformat = values -> [@sprintf("%.3g", value) for value in values])
    hidexdecorations!(ax2)
    hidespines!(ax2, :l)
    linkxaxes!(ax1, ax2)

    numeric_line = lines!(ax2, phases, I_numeric;
        color = :green, linestyle= Linestyle([1,3,5,7,9,11]), linewidth = 2, alpha = 1)
    two_e_line = lines!(ax2, phases, I_sin[1] .* sin.(phases) .+ I_cos[1] .* cos.(phases);
        linestyle = (:dash, :loose), alpha=0.7, linewidth= 2)
    four_e_line = lines!(ax2, phases, I_sin[2] .* sin.(2 .* phases) .+ I_cos[2] .* cos.(2 .* phases);
        color = :red, linestyle = Linestyle([0, 2, 4, 6, 8, 10, 20]),linewidth= 2, alpha=1)

    legend_plots = [free_energy_line, numeric_line, two_e_line, four_e_line]
    legend_labels = [
        "Ground-state free energy",
        "Josephson Current",
        "2e Doublet component",
        "4e Quartet component",
    ]
    Legend(fig[2, 1], legend_plots, legend_labels;
        orientation = :horizontal, nbanks = 2, labelsize = 11)

    ax3 = Axis(fig[1, 2],
        xlabel = "Harmonic order, k",
        ylabel = "Current amplitude, |Iₖ| [meV]",
        yscale = log10,
        title = "Current Harmonic Amplitudes",
        xticks = 1:max_harmonics)
    barplot!(ax3, 1:max_harmonics, max.(I_amp, 1e-12), color = :teal)

    display(fig)

    return (; fig, phases, F_vals, I_numeric, I_recon, A0, A_k, B_k,
        I_sin, I_cos, I_amp, params = parameter_values,
    )
end

# Josephson-current harmonics |I_k| over a (Vg, µ) grid for a single case (Δn = n - n0 and any other overrides),
# summed separately over even k (4e, 8e, ...) and odd k (2e, 6e, ...)
# (needs the full spectrum, so use ES.LinearAlgebra() as solver)
function sweep_vg_mu_harmonic_amplitudes(hN, case; Vg_values, µ_values, n0 = 0,
        npoints = 64, max_harmonics = 8, solver, savepath = nothing, params...)
    npoints > 2max_harmonics || throw(ArgumentError("npoints must be larger than 2max_harmonics"))
    phases = 2π .* (0:(npoints - 1)) ./ npoints
    harmonics = 1:max_harmonics
    (; Δn) = case
    overrides = Base.structdiff(case, NamedTuple{(:Δn,)})

    amplitudes = Array{Float64}(undef, length(Vg_values), length(µ_values), max_harmonics)
    for (iµ, µ_value) in enumerate(µ_values), (iVg, Vg_value) in enumerate(Vg_values)
        p = (; params..., overrides..., n0, n = n0 + Δn, Vg = Vg_value, µN = µ_value)
        free_energies = [free_energy(hN(; p..., ϕ = ϕ_value), solver) for ϕ_value in phases]
        A = [(2 / npoints) * sum(free_energies .* cos.(k .* phases)) for k in harmonics]
        B = [(2 / npoints) * sum(free_energies .* sin.(k .* phases)) for k in harmonics]
        I_sin = -2 .* harmonics .* A
        I_cos = 2 .* harmonics .* B
        amplitudes[iVg, iµ, :] .= hypot.(I_sin, I_cos)
    end
    I_odd = dropdims(sum(amplitudes[:, :, isodd.(harmonics)], dims = 3), dims = 3)
    I_even = dropdims(sum(amplitudes[:, :, iseven.(harmonics)], dims = 3), dims = 3)

    common = (; xlabel = L"V_g \text{ [meV]}", ylabel = L"\mu \text{ [meV]}", xlabelsize = 16, ylabelsize = 16)
    fig = Figure(size = (1100, 450))
    Label(fig[0, 1:4], @sprintf("Δn = %g", Δn); fontsize = 16)
    ax1 = Axis(fig[1, 1]; title = L"\sum_{k \text{ odd}} |I_k| \;\; (2e, 6e, \ldots)", common...)
    hm1 = heatmap!(ax1, Vg_values, µ_values, I_odd)
    Colorbar(fig[1, 2], hm1; label = L"\text{[meV]}")
    ax2 = Axis(fig[1, 3]; title = L"\sum_{k \text{ even}} |I_k| \;\; (4e, 8e, \ldots)", common...)
    hm2 = heatmap!(ax2, Vg_values, µ_values, I_even)
    Colorbar(fig[1, 4], hm2; label = L"\text{[meV]}")
    display(fig)
    savepath === nothing || save(savepath, fig)

    return (; fig, Vg_values, µ_values, amplitudes, I_odd, I_even)
end

# Flux points, split just around the fluxoid jumps of both rings (as in data_figure2)
function flux_grid(nmax, npoints, area_ratio; δn = 1e-6)
    jumps = sort(unique(filter(x -> 0 < x < nmax,
        [[k + 0.5 for k in 0:ceil(Int, nmax)]; [(k + 0.5) / area_ratio for k in 0:ceil(Int, nmax * area_ratio)]])))
    edges = [0.0; jumps; nmax]
    segments = [range(edges[k] + (k > 1 ? δn : 0.0), edges[k+1] - (k < length(edges) - 1 ? δn : 0.0),
        length = max(2, round(Int, npoints * (edges[k+1] - edges[k]) / nmax))) for k in 1:length(edges)-1]
    return reduce(vcat, segments), jumps, segments
end

# Fluxoid quanta bars: winding of each pairing term (Δ0 ↔ n0, Δ1 ↔ n) at each flux
function fluxoid_bar!(ax, xs, windings)
    edges = [first(xs); [(xs[i] + xs[i+1]) / 2 for i in 1:length(xs)-1 if windings[i] != windings[i+1]]; last(xs)]
    starts = [1; [i + 1 for i in 1:length(xs)-1 if windings[i] != windings[i+1]]]
    for (k, i) in enumerate(starts)
        w = windings[i]
        poly!(ax, Rect(edges[k], 0, edges[k+1] - edges[k], 1),
            color = (Makie.wong_colors()[mod1(w + 1, 7)], 0.6), strokecolor = :black, strokewidth = 1)
        text!(ax, (edges[k] + edges[k+1]) / 2, 0.5, text = string(w), align = (:center, :center), fontsize = 12)
    end
end

function plot_flux_spectrum(hN; nmax = 2.0, npoints = 150, area_ratio = 0.5, nev = 40,
        solver = ES.ShiftInvert(ES.Arpack(nev = 4 + nev), 0), savepath = nothing, params...)
    Δ = NamedTuple(params).Δ
    ns, jumps, _ = flux_grid(nmax, npoints, area_ratio)
    b = bands(hN, ns; mapping = n -> ftuple(; params..., n, n0 = n * area_ratio), solver)

    fig = Figure(size = (600, 550))
    bar_specs = (("Inner ring", round.(Int, ns .* area_ratio)),
                 ("Outer ring", round.(Int, ns)))
    bar_axes = map(enumerate(bar_specs)) do (row, (label, windings))
        axb = Axis(fig[row, 1], ylabel = label, ylabelrotation = 0,
            title = row == 1 ? L"\Delta_0=\Delta_1" : "", titlealign = :right)
        fluxoid_bar!(axb, ns, windings)
        hidexdecorations!(axb)
        hideydecorations!(axb, label = false)
        hidespines!(axb)
        limits!(axb, 0, nmax, 0, 1)
        rowsize!(fig.layout, row, Fixed(22))
        axb
    end

    # BdG spectrum vs flux (style of figure 2(c) in arXiv_2506.15502/figures.jl)
    ax = Axis(fig[3, 1];
        xlabel = L"\Phi/\Phi_0", ylabel = L"\epsilon/(\Delta_0+\Delta_1)",
        yticks = (range(-2Δ, 2Δ, 3), ["-1", "0", "1"]),
        limits = ((0, nmax), (-2.5Δ, 2.5Δ)), xgridvisible = false,
        xlabelsize = 16, ylabelsize = 16)
    qplot!(b; color = (ψ, ϵ, ϕs) -> ifelse(abs(ϵ) < 0.12, 0, 1), colormap = ColorScheme([colorant"red", colorant"grey"]), hide = :nodes)
    vlines!(ax, jumps; color = :gray, linewidth = 0.5)
    foreach(axb -> linkxaxes!(ax, axb), bar_axes)
    rowgap!(fig.layout, 4)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, ns, b, jumps)
end

function plot_free_energy_current(hN, cases; n0 = 0, npoints = 128, solver = ES.LinearAlgebra(),
        maxscale = 1000, savepath = nothing, params...)
    # Periodic grid without the duplicate endpoint (needed for the periodic central difference)
    phases = collect(range(0, 2π, length = npoints + 1)[1:end-1])
    N = length(phases)
    Δϕ = 2π / N

    results = map(cases) do case
        (; Δn, Vg) = case
        overrides = Base.structdiff(case, NamedTuple{(:Δn, :scale)})
        F_vals = [free_energy(hN(; params..., overrides..., n0, n = n0 + Δn, ϕ = ϕ_val), solver) for ϕ_val in phases]
        I_vals = [2.0 * (F_vals[mod1(i + 1, N)] - F_vals[mod1(i - 1, N)]) / (2.0 * Δϕ) for i in 1:N]
        (; case, F_vals, I_vals)
    end

    # Common normalization: largest free-energy variation and largest current over all cases
    Fref = maximum(r -> maximum(r.F_vals) - minimum(r.F_vals), results)
    Iref = maximum(r -> maximum(abs, r.I_vals), results)

    # Magnification of each case: case.scale if given, otherwise the power of 10 that brings
    # its current close to Iref, capped at maxscale so cases that are ~0 do not blow up noise
    function case_scale(r)
        haskey(r.case, :scale) && return r.case.scale
        Imax = maximum(abs, r.I_vals)
        Imax > 0 || return 1
        return min(10.0^floor(log10(Iref / Imax)), maxscale)
    end
    scales = map(case_scale, results)

    common = (; xlabelsize = 16, ylabelsize = 16, xgridvisible = false)
    fig = Figure(size = (600, 550))
    ax1 = Axis(fig[1, 1];
        title = L"\Delta_0=\Delta_1", titlealign = :right,
        ylabel = L"(F - F_\text{min}) / \Delta F_\text{ref}", common...)
    ax2 = Axis(fig[2, 1];
        xlabel = L"\varphi", ylabel = L"I / I_\text{ref}",
        xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]), common...)

    # Close the curves at ϕ = 2π for plotting
    closed(v) = [v; first(v)]
    results = map(results, scales) do r, scale
        (; Δn, Vg) = r.case
        label = @sprintf("Δn = %g, Vg = %g meV", Δn, Vg) * (scale == 1 ? "" : @sprintf(" (×%g)", scale))
        lines!(ax1, [phases; 2π], closed(scale .* (r.F_vals .- minimum(r.F_vals)) ./ Fref); linewidth = 2, label)
        lines!(ax2, [phases; 2π], closed(scale .* r.I_vals ./ Iref); linewidth = 2, label)
        (; r..., label, scale)
    end

    hlines!(ax2, 0; color = :black, linewidth = 0.5)
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, 2π)
    hidexdecorations!(ax1, grid = false)
    Legend(fig[3, 1], ax1; orientation = :horizontal, nbanks = 2, labelsize = 11, framevisible = false)
    rowgap!(fig.layout, 5)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, phases, results, Fref, Iref)
end

# -------------------------------------------------------------
# 3. Execution
# -------------------------------------------------------------

base_params = (;
    ħ2ome = 76.2,
    N = 250,
    m = 0.065,
    µ = 2,
    R = 500,
    seed = 1)


solver = ES.LinearAlgebra()


params = merge(base_params,(; R = 500, ϕ = 0, n = 1,
    N = 250, Δ = 0.2, α = 0,β = 0 ,Vg = 1.5,
    σs = 0.0000, σa = 0.000))
(; hN, ϵ1) = build(; params...)


results = detailed_fourier_analysis(hN;npoints = 51,max_harmonics = 8,solver = solver,params...)

#heatmaps = sweep_vg_mu_harmonic_amplitudes(hN, (; Δn = 1); Vg_values = range(0, 3, 7), µ_values = range(1, 5, 7),
#    npoints = 33, max_harmonics = 8, solver, savepath = joinpath(@__DIR__, "figures", "harmonics_Vg_mu.png"), params...)

flux_params = (; nmax = 4.0, npoints = 250, area_ratio = 0.4)


spectrum_plot = plot_flux_spectrum(hN; flux_params..., nev = 40,
    savepath = joinpath(@__DIR__, "figures", "spectrum.png"), params...)

    
Vg_on = 0.5
cases = [(; Δn = 0, Vg = Vg_on),
         (; Δn = 1, Vg = 0.0),
         (; Δn = 1, Vg = Vg_on)]
FI_plot = plot_free_energy_current(hN, cases; n0 = 0, npoints = 51, solver = ES.LinearAlgebra(),
    savepath = joinpath(@__DIR__, "figures", "free_energy_current.png"), params...)
print("done")
