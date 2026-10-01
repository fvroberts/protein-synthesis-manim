# The Navier–Stokes equations, in Manim

A ~3½ minute animated explainer of the incompressible Navier–Stokes equations,

$$\rho\Big(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\Big) = -\nabla p + \mu\nabla^2\mathbf{u} + \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0,$$

built with [Manim Community](https://www.manim.community/). The last two scenes
aren't hand-animated. They come from real numerical solutions of these equations,
computed by the small solvers in `fluid_solver.py`.

| Scene | What it shows |
| --- | --- |
| `Title` | Title card over animated streamlines |
| `EquationAnatomy` | From `F = ma` to the full equations; every term colour-coded and labelled |
| `Convection` | `(u·∇)u`: a steady nozzle flow where parcels still accelerate |
| `Pressure` | `−∇p`: pressure pushing on a parcel's faces; only the gradient matters |
| `Viscosity` | `μ∇²u`: a jet's velocity profile diffusing (exact `erf` solution) |
| `Incompressibility` | `∇·u = 0`: an expanding flow vs. an area-preserving one |
| `KelvinHelmholtz` | Simulated double shear layer rolling up into vortices that merge |
| `VortexStreet` | Simulated flow past a cylinder at Re = 150: a von Kármán vortex street |
| `Outro` | The open 3-D existence-and-smoothness Millennium Prize problem |
| `NavierStokesFull` | Every scene above, back to back |

## Rendering

```bash
pip install -r requirements.txt
cd navier_stokes

manim -qh navier_stokes.py NavierStokesFull   # whole video, 1080p60
manim -pql navier_stokes.py VortexStreet      # one scene, quick low-quality preview
```

Manim needs a LaTeX distribution (for the equations) and, on Linux, the Cairo and
Pango development headers. See the
[Manim installation guide](https://docs.manim.community/en/stable/installation.html).
On Debian/Ubuntu:

```bash
sudo apt install libcairo2-dev libpango1.0-dev texlive-latex-extra texlive-fonts-recommended texlive-science dvisvgm
```

The first render of `KelvinHelmholtz` and `VortexStreet` runs their simulations,
about 1½ and 4 minutes on a 4-core machine. The results are cached in
`sim_cache/`, so later renders start straight away. Delete that folder after
changing simulation parameters.

## The simulations

**Kelvin–Helmholtz** (`SpectralNS2D`): Fourier pseudo-spectral method on a
periodic 512×256 grid, at viscosity ν = 10⁻⁴. The nonlinear term is evaluated in
rotational form and dealiased with the 2/3 rule. Pressure is eliminated by
projecting every right-hand side onto divergence-free fields in Fourier space.
Viscosity is integrated exactly with an integrating factor, and the rest uses
3rd-order SSP Runge–Kutta.

**Vortex street** (`ChannelNS2D`): finite-difference projection method on a
staggered (MAC) grid with 24 cells per cylinder diameter. Each step advances
momentum with Adams–Bashforth 2 and central differences, then solves a pressure
Poisson equation (sparse LU, factorised once) that makes the velocity
divergence-free to machine precision. The cylinder is a set of blocked cells, so
it is exactly impermeable. Boundaries are uniform inflow, zero-gradient outflow
and free-slip walls. Shedding saturates by t ≈ 20 at a Strouhal number of about
0.20. The unconfined value at Re = 150 is ≈ 0.18, and the channel's 1:8 blockage
raises it slightly.

Both scenes colour the flow by **vorticity** ω = ∂v/∂x − ∂u/∂y, the local spin of
the fluid: blue is clockwise, orange counter-clockwise.
