"""
D2Q9 Lattice Boltzmann solver for 2-D incompressible flow past an obstacle
==========================================================================

A compact single-relaxation-time (BGK) lattice Boltzmann method. The
obstacle is imposed with half-way bounce-back; the inlet uses a Zou/He
velocity boundary condition and the outlet a zero-gradient condition.
The same solver is used to (a) validate against the circular-cylinder
benchmark and (b) simulate the inclined flat plate.

Lattice layout follows the standard convention (Latt): distributions are
stored as f[9, nx, ny].
"""

import numpy as np

# D2Q9 velocity set and weights
V = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
T = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
COL1 = np.array([0, 1, 2])     # vx = +1  (unknown at inlet)
COL2 = np.array([3, 4, 5])     # vx =  0
COL3 = np.array([6, 7, 8])     # vx = -1  (unknown at outlet)
OPP = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])   # opposite directions

_DT = np.float32
_VX = V[:, 0].astype(_DT).reshape(9, 1, 1)
_VY = V[:, 1].astype(_DT).reshape(9, 1, 1)
_T = T.astype(_DT).reshape(9, 1, 1)


def equilibrium(rho, u):
    cu = 3.0 * (_VX * u[0] + _VY * u[1])
    usqr = 1.5 * (u[0]**2 + u[1]**2)
    return (rho * _T * (1 + cu + 0.5 * cu**2 - usqr)).astype(_DT)


def macroscopic(fin):
    rho = fin.sum(axis=0)
    u = np.empty((2, *rho.shape), dtype=_DT)
    u[0] = (_VX * fin).sum(axis=0) / rho
    u[1] = (_VY * fin).sum(axis=0) / rho
    return rho, u


def cylinder_mask(nx, ny, cx, cy, r):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return (x - cx)**2 + (y - cy)**2 < r**2


def plate_mask(nx, ny, cx, cy, length, angle_deg, thickness=1.4):
    """Solid mask for a thin flat plate of given length/angle (deg from +x)."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])          # along-plate unit vector
    n = np.array([-np.sin(a), np.cos(a)])         # plate normal
    dx, dy = x - cx, y - cy
    along = dx * d[0] + dy * d[1]
    perp = dx * n[0] + dy * n[1]
    return (np.abs(perp) <= thickness) & (np.abs(along) <= length / 2)


def run(obstacle, nx, ny, uLB, Re, Lchar, n_steps,
        record_from=0, force_every=25, perturb=True, verbose=True,
        fin0=None, start_step=0):
    """Run the LBM. Re is based on the characteristic length Lchar (lattice).

    Pass fin0/start_step to resume from a checkpoint; the returned dict
    contains 'fin' so the run can be continued in another call.
    """
    nu = uLB * Lchar / Re
    omega = _DT(1.0 / (3.0 * nu + 0.5))

    # inlet / initial velocity field (tiny sinusoidal perturbation to seed shedding)
    vel = np.zeros((2, nx, ny), dtype=_DT)
    yy = np.arange(ny)
    pert = 1e-4 * np.sin(2 * np.pi * yy / ny) if perturb else 0.0
    vel[0] = (uLB * (1.0 + pert)[None, :]).astype(_DT)

    fin = fin0.copy() if fin0 is not None else equilibrium(np.ones((nx, ny), dtype=_DT), vel)

    # precompute static boundary links once (fluid node -> solid via direction i)
    links = []
    for i in range(9):
        shifted = np.roll(np.roll(obstacle, V[i, 0], axis=0), V[i, 1], axis=1)
        link = shifted & (~obstacle)
        links.append(link if link.any() else None)

    forces = []
    for local in range(n_steps):
        step = start_step + local
        fin[COL3, -1, :] = fin[COL3, -2, :]                      # outlet
        rho, u = macroscopic(fin)

        u[:, 0, :] = vel[:, 0, :]                                # inlet Zou/He
        rho[0, :] = (1.0 / (1.0 - u[0, 0, :])) * (
            fin[COL2, 0, :].sum(axis=0) + 2.0 * fin[COL3, 0, :].sum(axis=0))

        feq = equilibrium(rho, u)
        fin[COL1, 0, :] = feq[COL1, 0, :] + fin[COL3[::-1], 0, :] - feq[COL3[::-1], 0, :]

        fout = fin - omega * (fin - feq)                         # BGK collision

        for i in range(9):                                       # bounce-back
            fout[i, obstacle] = fin[OPP[i], obstacle]

        if step >= record_from and step % force_every == 0:      # momentum exchange
            Fx = Fy = 0.0
            for i in range(9):
                lk = links[i]
                if lk is not None:
                    amt = (fout[i][lk] + fin[OPP[i]][lk]).sum()
                    Fx += V[i, 0] * amt
                    Fy += V[i, 1] * amt
            forces.append((step, float(Fx), float(Fy)))

        for i in range(9):                                       # streaming
            fin[i] = np.roll(np.roll(fout[i], V[i, 0], axis=0), V[i, 1], axis=1)

        if verbose and step % 5000 == 0:
            umax = float(np.sqrt(u[0]**2 + u[1]**2).max())
            print(f"  step {step:6d}   umax={umax:.4f}", flush=True)

    rho, u = macroscopic(fin)
    u[0][obstacle] = 0.0
    u[1][obstacle] = 0.0
    return {"u": u, "rho": rho, "omega": omega, "nu": nu,
            "forces": np.array(forces), "fin": fin, "end_step": start_step + n_steps}


def vorticity(u):
    dudy = np.gradient(u[0], axis=1)
    dvdx = np.gradient(u[1], axis=0)
    return dvdx - dudy


def strouhal(forces, D, U, dt=1):
    """Estimate Strouhal number from the lift (Fy) oscillation via FFT."""
    if len(forces) < 64:
        return None, None, None
    steps = forces[:, 0]
    fy = forces[:, 2] - forces[:, 2].mean()
    n = len(fy)
    freqs = np.fft.rfftfreq(n, d=(steps[1] - steps[0]))
    amp = np.abs(np.fft.rfft(fy * np.hanning(n)))
    amp[0] = 0.0
    fpk = freqs[np.argmax(amp)]           # cycles per step
    St = fpk * D / U
    return St, fpk, (freqs, amp)
