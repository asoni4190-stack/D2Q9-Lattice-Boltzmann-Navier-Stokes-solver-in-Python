"""
Inclined Flat Plate Aerodynamics, SINGLE-FILE version
it runs the whole project (Part A + cylinder validation + plate simulation + all figures) from one file.

  QUICK = True   -> coarse grids / few steps, ~1 minute  
  QUICK = False  -> full resolution, ~6-8 minutes        
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

QUICK = False     # <-- set to False for full-resolution report figures and True for quick analysis

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})


# PART A - empirical force analysis at the real condition (35 m/s, 45 deg)

def part_A():
    U, chord, span, alpha_deg = 35.0, 1.0, 1.0, 45.0
    rho, mu, a_sound = 1.225, 1.802e-5, 340.3      # air, 15 C ISA
    nu = mu / rho
    alpha = np.radians(alpha_deg)
    A = chord * span
    q = 0.5 * rho * U**2
    Re = U * chord / nu
    Mach = U / a_sound

    C_N = 1.0 / (0.222 + 0.283 / np.sin(alpha))     # Hoerner, separated flow
    Cd, Cl = C_N * np.sin(alpha), C_N * np.cos(alpha)
    N = C_N * q * A
    Drag, Lift = Cd * q * A, Cl * q * A
    Cf = 0.074 / Re**0.2
    D_fric = Cf * q * A

    print("=" * 60)
    print("PART A  -  FORCE ANALYSIS AT 35 m/s, 45 deg")
    print("=" * 60)
    print(f"  dynamic pressure q = {q:8.1f} Pa")
    print(f"  Reynolds number Re = {Re:8.3e}   (turbulent, separated)")
    print(f"  Mach number      M = {Mach:8.3f}   (incompressible)")
    print(f"  C_N = {C_N:.3f}   Cd = {Cd:.3f}   Cl = {Cl:.3f}")
    print(f"  normal force N = {N:7.1f} N")
    print(f"  drag         D = {Drag:7.1f} N")
    print(f"  lift         L = {Lift:7.1f} N")
    print(f"  skin friction  = {D_fric:6.1f} N  ({100*D_fric/Drag:.1f}% of drag)")
    print("  (2-D per-unit-span; a finite square plate is ~40% lower)")

    aa = np.radians(np.linspace(1, 90, 200))
    cn = 1.0 / (0.222 + 0.283 / np.sin(aa))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(np.degrees(aa), cn, label="$C_N$", lw=2)
    ax.plot(np.degrees(aa), cn * np.sin(aa), label="$C_d$", lw=2)
    ax.plot(np.degrees(aa), cn * np.cos(aa), label="$C_l$", lw=2)
    ax.axvline(45, color="grey", ls="--", lw=1)
    ax.plot(45, Cd, "ko"); ax.plot(45, Cl, "ko")
    ax.set(xlabel="angle of attack [deg]", ylabel="coefficient",
           title="Flat-plate force coefficients (Hoerner, 2-D)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/coefficients_vs_angle.png"); plt.show()



# PART B - D2Q9 lattice Boltzmann solver

V = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
T = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
COL1, COL2, COL3 = np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])
OPP = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
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


def plate_mask(nx, ny, cx, cy, length, angle_deg, thickness=1.5):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-np.sin(a), np.cos(a)])
    dx, dy = x - cx, y - cy
    along = dx * d[0] + dy * d[1]
    perp = dx * n[0] + dy * n[1]
    return (np.abs(perp) <= thickness) & (np.abs(along) <= length / 2)


def run(obstacle, nx, ny, uLB, Re, Lchar, n_steps,
        record_from, force_every=20, label=""):
    nu = uLB * Lchar / Re
    omega = _DT(1.0 / (3.0 * nu + 0.5))
    vel = np.zeros((2, nx, ny), dtype=_DT)
    pert = 1e-4 * np.sin(2 * np.pi * np.arange(ny) / ny)
    vel[0] = (uLB * (1.0 + pert)[None, :]).astype(_DT)
    fin = equilibrium(np.ones((nx, ny), dtype=_DT), vel)

    links = []
    for i in range(9):
        sh = np.roll(np.roll(obstacle, V[i, 0], axis=0), V[i, 1], axis=1)
        lk = sh & (~obstacle)
        links.append(lk if lk.any() else None)

    forces, t0 = [], time.time()
    for step in range(n_steps):
        fin[COL3, -1, :] = fin[COL3, -2, :]
        rho, u = macroscopic(fin)
        u[:, 0, :] = vel[:, 0, :]
        rho[0, :] = (1.0 / (1.0 - u[0, 0, :])) * (
            fin[COL2, 0, :].sum(axis=0) + 2.0 * fin[COL3, 0, :].sum(axis=0))
        feq = equilibrium(rho, u)
        fin[COL1, 0, :] = feq[COL1, 0, :] + fin[COL3[::-1], 0, :] - feq[COL3[::-1], 0, :]
        fout = fin - omega * (fin - feq)
        for i in range(9):
            fout[i, obstacle] = fin[OPP[i], obstacle]
        if step >= record_from and step % force_every == 0:
            Fx = Fy = 0.0
            for i in range(9):
                lk = links[i]
                if lk is not None:
                    amt = (fout[i][lk] + fin[OPP[i]][lk]).sum()
                    Fx += V[i, 0] * amt; Fy += V[i, 1] * amt
            forces.append((step, float(Fx), float(Fy)))
        for i in range(9):
            fin[i] = np.roll(np.roll(fout[i], V[i, 0], axis=0), V[i, 1], axis=1)
        if step % 3000 == 0:
            umax = float(np.sqrt(u[0]**2 + u[1]**2).max())
            print(f"  [{label}] step {step:6d}/{n_steps}  umax={umax:.3f}", flush=True)

    rho, u = macroscopic(fin)
    u[0][obstacle] = 0.0; u[1][obstacle] = 0.0
    print(f"  [{label}] finished in {time.time()-t0:.0f}s")
    return u, np.array(forces)


def vorticity(u):
    return np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)


def strouhal(forces, Lref, U):
    steps = forces[:, 0]
    fy = forces[:, 2] - forces[:, 2].mean()
    n = len(fy); N = 1 << 18; dt = steps[1] - steps[0]
    amp = np.abs(np.fft.rfft(fy * np.hanning(n), N)); amp[0] = 0
    freqs = np.fft.rfftfreq(N, d=dt); k = int(np.argmax(amp))
    a, b, c = amp[k-1], amp[k], amp[k+1]
    dk = 0.5 * (a - c) / (a - 2*b + c) if (a - 2*b + c) != 0 else 0
    fpk = (k + dk) / (N * dt)
    return fpk * Lref / U, freqs, amp


# run configurations 
if QUICK:
    CYL = dict(nx=240, ny=120, cx=70, cy=60, r=10, uLB=0.10, Re=100,
               steps=6000, record_from=2500)
    PLATE = dict(nx=300, ny=130, cx=95, cy=65, L=30, ang=45, uLB=0.12, Re=300,
                 steps=6000, record_from=2500)
else:
    CYL = dict(nx=360, ny=180, cx=90, cy=90, r=14, uLB=0.10, Re=100,
               steps=18000, record_from=6000)
    PLATE = dict(nx=420, ny=180, cx=140, cy=90, L=40, ang=45, uLB=0.12, Re=300,
                 steps=18000, record_from=6000)


def part_B():
    #cylinder validation
    print("\n" + "=" * 60)
    print("PART B.1  -  CYLINDER VALIDATION (Re=100)")
    print("=" * 60)
    c = CYL
    obs = cylinder_mask(c["nx"], c["ny"], c["cx"], c["cy"], c["r"])
    u, F = run(obs, c["nx"], c["ny"], c["uLB"], c["Re"], 2*c["r"],
               c["steps"], c["record_from"], label="cyl")
    D, U = 2*c["r"], c["uLB"]
    St, _, _ = strouhal(F, D, U)
    m = F[:, 0] > c["record_from"] + 2000
    Cd = -F[m, 1].mean() / (0.5 * U**2 * D)
    print(f"  RESULT  St = {St:.3f} (benchmark 0.164)   Cd = {Cd:.3f} (benchmark ~1.35)")

    w = np.ma.array(vorticity(u), mask=obs)
    lim = np.abs(w).max() * 0.6
    plt.figure(figsize=(9, 4.2))
    plt.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    plt.title("Cylinder wake vorticity (Re=100) - validation"); plt.axis("off")
    plt.tight_layout(); plt.savefig("figures/cyl_vorticity.png"); plt.show()

    #inclined plate 
    print("\n" + "=" * 60)
    print("PART B.2  -  INCLINED PLATE (45 deg, Re=300)")
    print("=" * 60)
    p = PLATE
    obs = plate_mask(p["nx"], p["ny"], p["cx"], p["cy"], p["L"], p["ang"])
    u, F = run(obs, p["nx"], p["ny"], p["uLB"], p["Re"], p["L"],
               p["steps"], p["record_from"], force_every=15, label="plate")
    U, L = p["uLB"], p["L"]
    h = L * np.sin(np.radians(p["ang"]))          # frontal projected height
    Fx, Fy = -F[:, 1], -F[:, 2]                   # body force = -(fluid force)
    m = F[:, 0] > p["record_from"] + 2000
    qref = 0.5 * U**2 * L
    Cd, Cl = Fx[m].mean() / qref, Fy[m].mean() / qref
    Clf = np.sqrt(((Fy[m] - Fy[m].mean())**2).mean()) / qref
    St, freqs, amp = strouhal(F, h, U)
    print(f"  RESULT  Cd={Cd:.2f}  |Cl|={abs(Cl):.2f}  Cl'rms={Clf:.2f}  St={St:.3f}")

    nx, ny = p["nx"], p["ny"]
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    w = np.ma.array(vorticity(u), mask=obs)
    speed = np.ma.array(np.sqrt(u[0]**2 + u[1]**2), mask=obs)

    lim = np.abs(w).max() * 0.6
    plt.figure(figsize=(9, 4.2))
    plt.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    plt.title("Inclined plate (45 deg, Re=300) - vorticity"); plt.axis("off")
    plt.tight_layout(); plt.savefig("figures/plate_vorticity.png"); plt.show()

    plt.figure(figsize=(9, 4.2))
    plt.contourf(X, Y, speed / U, levels=30, cmap="turbo")
    plt.colorbar(label="|U|/U_inf", shrink=0.8)
    plt.streamplot(np.arange(nx), np.arange(ny), u[0].T, u[1].T,
                   density=1.4, color="k", linewidth=0.4, arrowsize=0.6)
    plt.title("Inclined plate (45 deg, Re=300) - speed & streamlines")
    plt.xlim(0, nx); plt.ylim(0, ny); plt.gca().set_aspect("equal"); plt.axis("off")
    plt.tight_layout(); plt.savefig("figures/plate_streamlines.png"); plt.show()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(F[:, 0], Fx / qref, label="$C_d$", lw=1)
    a1.plot(F[:, 0], Fy / qref, label="$C_l$", lw=1)
    a1.set(xlabel="time step", ylabel="coefficient", title="Force history")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(freqs * h / U, amp / amp.max(), lw=1.2); a2.axvline(St, color="r", ls="--")
    a2.set(xlim=(0, 1), xlabel="Strouhal $fh/U$", ylabel="lift spectrum",
           title=f"Shedding peak St={St:.3f}"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/plate_forces.png"); plt.show()


if __name__ == "__main__":
    part_A()
    part_B()
    print("\nDone. Figures also saved in ./figures/")
