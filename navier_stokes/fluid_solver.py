"""Two small solvers for the 2-D incompressible Navier-Stokes equations,

    du/dt + (u . grad) u = -grad p + nu lap u,      div u = 0,

used to drive the simulation scenes in ``navier_stokes.py``.

``SpectralNS2D``
    Fourier pseudo-spectral method on a doubly periodic box (Kelvin-Helmholtz).
    The nonlinear term is evaluated in rotational form, (v w, -u w) with
    w = dv/dx - du/dy, and dealiased with the 2/3 rule. Pressure is never formed:
    every right-hand side is projected onto divergence-free fields,
    u_hat <- u_hat - k (k . u_hat) / |k|^2. Viscosity is integrated exactly with an
    integrating factor, and the rest uses 3rd-order SSP Runge-Kutta.

``ChannelNS2D``
    Finite-difference projection method on a staggered (MAC) grid, for flow past
    an obstacle. Uniform inflow on the left, zero-gradient outflow on the right,
    free-slip walls at top and bottom. The obstacle is a set of blocked cells, so
    it is exactly impermeable. Each step advances momentum with 2nd-order
    Adams-Bashforth and central differences, then solves a pressure Poisson
    equation (sparse LU, factorised once) to make the velocity divergence-free.

Simulation results are cached to ``sim_cache/*.npz`` so re-rendering skips them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import fft, sparse
from scipy.sparse.linalg import splu

CACHE_DIR = Path(__file__).resolve().parent / "sim_cache"


class SpectralNS2D:
    """Incompressible Navier-Stokes on a periodic ``lx`` x ``ly`` box.

    Arrays are indexed ``[iy, ix]``, so ``y`` runs down the rows.
    """

    def __init__(self, nx: int, ny: int, lx: float, ly: float, nu: float, dt: float):
        self.nx, self.ny, self.lx, self.ly = nx, ny, lx, ly
        self.nu, self.dt, self.t = nu, dt, 0.0

        x = np.arange(nx) * lx / nx
        y = np.arange(ny) * ly / ny
        self.x, self.y = np.meshgrid(x, y)

        kx = 2 * np.pi * fft.rfftfreq(nx, d=lx / nx)
        ky = 2 * np.pi * fft.fftfreq(ny, d=ly / ny)
        self.kx, self.ky = np.meshgrid(kx, ky)
        self.k2 = self.kx**2 + self.ky**2
        self._k2_safe = np.where(self.k2 == 0, 1.0, self.k2)

        kx_cut = (2 / 3) * np.abs(kx).max()
        ky_cut = (2 / 3) * np.abs(ky).max()
        self._dealias = (np.abs(self.kx) < kx_cut) & (np.abs(self.ky) < ky_cut)

        # Exact viscous decay over a full and a half time step (and its inverse).
        self._e_full = np.exp(-nu * self.k2 * dt)
        self._e_half = np.exp(-nu * self.k2 * dt / 2)
        self._e_mhalf = 1 / self._e_half

        self.uh = np.zeros_like(self.k2, dtype=complex)
        self.vh = np.zeros_like(self.uh)

    def _fwd(self, a):
        return fft.rfft2(a, workers=-1)

    def _inv(self, ah):
        return fft.irfft2(ah, s=(self.ny, self.nx), workers=-1)

    def _project(self, ah, bh):
        k_dot = (self.kx * ah + self.ky * bh) / self._k2_safe
        return ah - self.kx * k_dot, bh - self.ky * k_dot

    def set_velocity(self, u, v):
        self.uh, self.vh = self._project(self._fwd(u), self._fwd(v))

    @property
    def velocity(self):
        return self._inv(self.uh), self._inv(self.vh)

    @property
    def vorticity(self):
        return self._inv(1j * (self.kx * self.vh - self.ky * self.uh))

    def _rhs(self, uh, vh):
        u, v = self._inv(uh), self._inv(vh)
        w = self._inv(1j * (self.kx * vh - self.ky * uh))
        return self._project(self._fwd(v * w) * self._dealias, self._fwd(-u * w) * self._dealias)

    def step(self):
        dt, ef, eh, emh = self.dt, self._e_full, self._e_half, self._e_mhalf
        u0, v0 = self.uh, self.vh

        # Integrating-factor SSP-RK3 (Shu-Osher form).
        nu_, nv_ = self._rhs(u0, v0)
        u1, v1 = ef * (u0 + dt * nu_), ef * (v0 + dt * nv_)
        nu_, nv_ = self._rhs(u1, v1)
        u2 = 0.75 * eh * u0 + 0.25 * emh * (u1 + dt * nu_)
        v2 = 0.75 * eh * v0 + 0.25 * emh * (v1 + dt * nv_)
        nu_, nv_ = self._rhs(u2, v2)
        self.uh = ef * u0 / 3 + (2 / 3) * eh * (u2 + dt * nu_)
        self.vh = ef * v0 / 3 + (2 / 3) * eh * (v2 + dt * nv_)
        self.t += dt


class ChannelNS2D:
    """Channel flow past an obstacle on a staggered grid with spacing ``h``.

    ``solid`` is a boolean ``(nx, ny)`` array of blocked cells. Velocities carry one
    ring of ghost cells: ``u[i, j]`` sits on the right face of cell ``(i, j)``,
    ``v[i, j]`` on its top face, with interior cells numbered from 1.
    """

    def __init__(self, solid: np.ndarray, h: float, nu: float, dt: float, u_inf: float = 1.0):
        nx, ny = solid.shape
        self.nx, self.ny, self.h, self.nu, self.dt, self.u_inf = nx, ny, h, nu, dt, u_inf
        self.t = 0.0

        fluid = np.zeros((nx + 2, ny + 2), dtype=bool)
        fluid[1:-1, 1:-1] = ~solid
        blocked = np.zeros_like(fluid)
        blocked[1:-1, 1:-1] = solid
        self.fluid = fluid

        # Faces between two fluid cells carry flow; faces touching the obstacle do not.
        self.u_open = fluid[:-1, :] & fluid[1:, :]
        self.v_open = fluid[:, :-1] & fluid[:, 1:]
        self.u_blocked = blocked[:-1, :] | blocked[1:, :]
        self.v_blocked = blocked[:, :-1] | blocked[:, 1:]
        # Faces buried in the obstacle mirror their open neighbour, putting the
        # no-slip wall exactly on the cell boundary.
        u_buried = blocked[:-1, :] & blocked[1:, :]
        v_buried = blocked[:, :-1] & blocked[:, 1:]
        self._u_mirror_n = u_buried & np.roll(self.u_open, -1, axis=1)
        self._u_mirror_s = u_buried & np.roll(self.u_open, 1, axis=1) & ~self._u_mirror_n
        self._v_mirror_e = v_buried & np.roll(self.v_open, -1, axis=0)
        self._v_mirror_w = v_buried & np.roll(self.v_open, 1, axis=0) & ~self._v_mirror_e
        self._outflow = fluid[nx, :]

        self.u = np.full((nx + 1, ny + 2), float(u_inf))
        self.v = np.zeros((nx + 2, ny + 1))
        self._previous_rhs = None
        self._factorise_poisson()

    def _factorise_poisson(self):
        """Laplacian over fluid cells: Neumann at walls and obstacle, p = 0 at the outlet."""
        n = int(self.fluid.sum())
        index = -np.ones(self.fluid.shape, dtype=int)
        index[self.fluid] = np.arange(n)
        a = np.concatenate([index[:-1, :][self.u_open], index[:, :-1][self.v_open]])
        b = np.concatenate([index[1:, :][self.u_open], index[:, 1:][self.v_open]])
        diag = -(np.bincount(a, minlength=n) + np.bincount(b, minlength=n)).astype(float)
        diag[index[self.nx, :][self._outflow]] -= 2
        ones = np.ones(len(a))
        diagonal = np.arange(n)
        matrix = sparse.coo_matrix(
            (np.concatenate([ones, ones, diag]), (np.concatenate([a, b, diagonal]), np.concatenate([b, a, diagonal]))),
            shape=(n, n),
        )
        self._solve = splu(matrix.tocsc()).solve

    def _apply_boundary_conditions(self, u, v):
        u[0, :] = self.u_inf  # inflow
        v[0, :] = -v[1, :]
        u[-1, :] = u[-2, :]  # zero-gradient outflow
        v[-1, :] = v[-2, :]
        u[:, 0], u[:, -1] = u[:, 1], u[:, -2]  # free-slip walls
        v[:, 0] = v[:, -1] = 0
        u[self.u_blocked] = 0
        v[self.v_blocked] = 0
        u[self._u_mirror_n] = -np.roll(u, -1, axis=1)[self._u_mirror_n]
        u[self._u_mirror_s] = -np.roll(u, 1, axis=1)[self._u_mirror_s]
        v[self._v_mirror_e] = -np.roll(v, -1, axis=0)[self._v_mirror_e]
        v[self._v_mirror_w] = -np.roll(v, 1, axis=0)[self._v_mirror_w]

    def _rhs(self, u, v):
        h, nu = self.h, self.nu

        uc = u[1:-1, 1:-1]
        ue, uw, un, us = u[2:, 1:-1], u[:-2, 1:-1], u[1:-1, 2:], u[1:-1, :-2]
        v_top = (v[1:-2, 1:] + v[2:-1, 1:]) / 2
        v_bottom = (v[1:-2, :-1] + v[2:-1, :-1]) / 2
        du2_dx = (((uc + ue) / 2) ** 2 - ((uw + uc) / 2) ** 2) / h
        duv_dy = (v_top * (uc + un) - v_bottom * (us + uc)) / (2 * h)
        ru = nu * (ue + uw + un + us - 4 * uc) / h**2 - du2_dx - duv_dy

        vc = v[1:-1, 1:-1]
        ve, vw, vn, vs = v[2:, 1:-1], v[:-2, 1:-1], v[1:-1, 2:], v[1:-1, :-2]
        u_right = (u[1:, 1:-2] + u[1:, 2:-1]) / 2
        u_left = (u[:-1, 1:-2] + u[:-1, 2:-1]) / 2
        duv_dx = (u_right * (vc + ve) - u_left * (vw + vc)) / (2 * h)
        dv2_dy = (((vc + vn) / 2) ** 2 - ((vs + vc) / 2) ** 2) / h
        rv = nu * (ve + vw + vn + vs - 4 * vc) / h**2 - duv_dx - dv2_dy
        return ru, rv

    def step(self):
        h, dt = self.h, self.dt
        u, v = self.u, self.v
        self._apply_boundary_conditions(u, v)

        ru, rv = self._rhs(u, v)
        if self._previous_rhs is None:
            au, av = ru, rv
        else:
            au, av = 1.5 * ru - 0.5 * self._previous_rhs[0], 1.5 * rv - 0.5 * self._previous_rhs[1]
        self._previous_rhs = (ru, rv)

        fu, fv = u.copy(), v.copy()
        fu[1:-1, 1:-1] += dt * au
        fv[1:-1, 1:-1] += dt * av
        fu[self.u_blocked] = 0
        fv[self.v_blocked] = 0
        fu[-1, :] = fu[-2, :]

        # Pressure projection: lap p = div(F) / dt, then u = F - dt grad p.
        div = (fu[1:, 1:-1] - fu[:-1, 1:-1] + fv[1:-1, 1:] - fv[1:-1, :-1]) / h
        p = np.zeros(self.fluid.shape)
        p[self.fluid] = self._solve(div[self.fluid[1:-1, 1:-1]] * h * h / dt)
        fu[self.u_open] -= dt * ((p[1:, :] - p[:-1, :]) / h)[self.u_open]
        fv[self.v_open] -= dt * ((p[:, 1:] - p[:, :-1]) / h)[self.v_open]
        fu[self.nx, self._outflow] += 2 * dt * p[self.nx, self._outflow] / h

        self.u, self.v = fu, fv
        self.t += dt

    @property
    def vorticity(self):
        """Vorticity at cell corners, indexed ``[iy, ix]``."""
        u, v, h = self.u, self.v, self.h
        w = (v[1:, :] - v[:-1, :]) / h - (u[:, 1:] - u[:, :-1]) / h
        return w.T

    @property
    def divergence(self):
        u, v, h = self.u, self.v, self.h
        div = (u[1:, 1:-1] - u[:-1, 1:-1] + v[1:-1, 1:] - v[1:-1, :-1]) / h
        return div[self.fluid[1:-1, 1:-1]]


def _run(solver, n_frames, steps_per_frame, label):
    frames, times = [], np.empty(n_frames)
    for i in range(n_frames):
        frames.append(solver.vorticity.astype(np.float32))
        times[i] = solver.t
        if not np.isfinite(frames[-1]).all():
            raise FloatingPointError(f"{label}: simulation blew up at t={solver.t:.2f}")
        for _ in range(steps_per_frame):
            solver.step()
        if i % 50 == 0:
            print(f"  [{label}] frame {i}/{n_frames}  t={solver.t:.2f}", flush=True)
    return np.stack(frames), times


def _cached(name, build):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    print(f"Simulating {name} (cached afterwards in {path}) ...", flush=True)
    result = build()
    np.savez_compressed(path, **result)
    return result


def kelvin_helmholtz(nx=512, ny=256, n_frames=600, steps_per_frame=6):
    """Double shear layer on [0, 2] x [0, 1]: counter-flowing streams roll up into vortices."""

    def build():
        lx, ly, nu, dt = 2.0, 1.0, 1.0e-4, 1.0e-3
        s = SpectralNS2D(nx, ny, lx, ly, nu=nu, dt=dt)
        rho = 30.0  # inverse shear-layer thickness
        y, x = s.y, s.x
        u = np.where(y <= 0.5, np.tanh(rho * (y - 0.25)), np.tanh(rho * (0.75 - y)))
        # Seed the most unstable wavelength (~0.5) plus weak low modes that later
        # make neighbouring vortices pair up.
        rng = np.random.default_rng(7)
        v = 0.05 * np.sin(2 * np.pi * x / 0.5)
        for m in (1, 2, 3):
            v += 0.01 * np.sin(2 * np.pi * m * x / lx + rng.uniform(0, 2 * np.pi))
        s.set_velocity(u, v)
        frames, times = _run(s, n_frames, steps_per_frame, "kelvin-helmholtz")
        return {"vorticity": frames.astype(np.float16), "times": times, "lx": lx, "ly": ly}

    return _cached(f"kelvin_helmholtz_{nx}x{ny}_f{n_frames}_s{steps_per_frame}", build)


def cylinder_wake(cells_per_diameter=24, n_frames=720, frame_dt=0.125, reynolds=150):
    """Uniform flow past a cylinder of diameter 1: the von Karman vortex street."""

    def build():
        lx, ly, d, u_inf = 15.0, 8.0, 1.0, 1.0
        cx, cy = 3.0, ly / 2
        h = d / cells_per_diameter
        nx, ny = round(lx / h), round(ly / h)
        xc = (np.arange(nx) + 0.5) * h
        yc = (np.arange(ny) + 0.5) * h
        solid = np.hypot(xc[:, None] - cx, yc[None, :] - cy) < d / 2

        steps_per_frame = int(np.ceil(frame_dt / (0.15 * h / u_inf)))
        s = ChannelNS2D(solid, h, nu=u_inf * d / reynolds, dt=frame_dt / steps_per_frame, u_inf=u_inf)
        # A brief sideways nudge in the near wake breaks the symmetry so that
        # shedding starts promptly instead of waiting for round-off to grow.
        xv, yv = (np.arange(nx + 2) - 0.5) * h, np.arange(ny + 1) * h
        s.v += 0.5 * np.exp(-((xv[:, None] - cx - 1.2) ** 2 + (yv[None, :] - cy - 0.25) ** 2) / 0.2)

        frames, times = _run(s, n_frames, steps_per_frame, "cylinder")
        return {
            "vorticity": frames.astype(np.float16),
            "times": times,
            "lx": lx,
            "ly": ly,
            "cx": cx,
            "cy": cy,
            "d": d,
            "reynolds": float(reynolds),
        }

    return _cached(f"cylinder_re{reynolds}_c{cells_per_diameter}_f{n_frames}", build)
