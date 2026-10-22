# Meshes

Hofstadter butterflies of mesoscopic networks (meshes) of 1D wires, normal and superconducting, built with [Quantica.jl](https://github.com/pablosanjose/Quantica.jl).

Each mesh is a finite lattice whose bonds are 1D tight-binding wires: every nearest-neighbour bond of the underlying lattice is subdivided into `N - 1` segments. A perpendicular magnetic field enters through Peierls phases, and the spectrum is computed as a function of the flux per plaquette `ϕ/ϕ₀`.

## Contents

All scripts are in [`exercises/`](exercises):

| Script | What it does |
|---|---|
| `exercise1.jl` | Warm-up: bands of a 1D chain |
| `exercise2.jl` | A single square plaquette made of wires; spectrum against flux |
| `exercise3.jl` | Square mesh (`M × M` plaquettes); butterfly with bulk/edge weighting |
| `exercise3_honeycomb.jl` | Honeycomb mesh |
| `exercise3_dice.jl` | Dice (T₃) mesh, with separate `:S`, `:E`, `:Ins` sublattices |
| `super_butterfly.jl` | Superconducting (BdG) versions of the square, honeycomb and dice meshes, with pairing `Δ` and a pairing-phase gradient `ϕ` |

Parameters you will usually change:

- `N`: number of sites per wire (bond subdivision)
- `M`: system size in plaquettes
- `t`, `e` / `ϵ`: hopping and onsite energy
- `del`, `ϕ`: pairing amplitude and phase (superconducting scripts only)
- `flux_ratios`: the range of `ϕ/ϕ₀` to sweep

## Requirements

Julia 1.13 (the `Manifest.toml` was generated with it; older versions may need `] resolve`). The exact package versions are recorded in `Project.toml` and `Manifest.toml`. To install them, run this from the repo folder:

```
julia --project
] instantiate
```

## Usage

The scripts save figures with relative paths such as `Figures/metal/...` and `Figures/superc/...`, so run them from the repo folder:

```bash
mkdir -p Figures/metal Figures/superc
julia --project exercises/exercise3.jl
```

Or open a script in VS Code with the Julia extension and run it interactively.

The `Figures/` folders hold generated output and are not tracked by git.
