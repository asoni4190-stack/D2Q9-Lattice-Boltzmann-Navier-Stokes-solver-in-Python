import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lbm

d = np.load("result_plate.npz")
u, obs, F = d["u"], d["obs"], d["forces"]
nx, ny = int(d["nx"]), int(d["ny"])
U = float(d["U"]); L = 40.0; alpha = np.radians(45)
h = L * np.sin(alpha)                     # frontal projected height
steps = F[:, 0]

#forces: body force = -(momentum imparted to fluid); ref = chord L 
Fx = -F[:, 1]; Fy = -F[:, 2]
m = steps > 8000
qref = 0.5 * U**2 * L
Cd = Fx[m].mean() / qref
Cl = Fy[m].mean() / qref
Cd_f = np.sqrt(((Fx[m]-Fx[m].mean())**2).mean()) / qref
Cl_f = np.sqrt(((Fy[m]-Fy[m].mean())**2).mean()) / qref

# --- refined Strouhal from lift signal (ref = frontal height h) ---
fy = Fy - Fy.mean(); n = len(fy); N = 1 << 18; dt = steps[1]-steps[0]
amp = np.abs(np.fft.rfft(fy*np.hanning(n), N)); amp[0] = 0
freqs = np.fft.rfftfreq(N, d=dt); k = np.argmax(amp)
a, b, c = amp[k-1], amp[k], amp[k+1]; dk = 0.5*(a-c)/(a-2*b+c)
fpk = (k+dk)/(N*dt); St = fpk * h / U

print(f"Re=300 plate @45deg  (reference = chord, per unit span)")
print(f"  mean Cd = {Cd:.3f}   mean Cl = {Cl:.3f}")
print(f"  Cd' rms = {Cd_f:.3f}   Cl' rms = {Cl_f:.3f}")
print(f"  Strouhal (frontal height) St = {St:.3f}")

# figures 
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
w = np.ma.array(lbm.vorticity(u), mask=obs)
speed = np.ma.array(np.sqrt(u[0]**2+u[1]**2), mask=obs)

fig, ax = plt.subplots(figsize=(9, 4.2))
lim = 0.03
im = ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
ax.set_title("Inclined thin plate (1.0 m chord \u00d7 0.03 m thick, 45\u00b0, Re=300) \u2014 vorticity"); ax.axis("off")
fig.tight_layout(); fig.savefig("figures/plate_vorticity.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
cf = ax.contourf(X, Y, speed/U, levels=30, cmap="turbo")
st = 9
ax.streamplot(np.arange(nx), np.arange(ny), u[0].T, u[1].T,
              density=1.4, color="k", linewidth=0.4, arrowsize=0.6)
fig.colorbar(cf, ax=ax, label="|U|/U\u221e", shrink=0.8)
ax.set_title("Inclined thin plate (1.0 m chord \u00d7 0.03 m thick, 45\u00b0, Re=300) \u2014 speed & streamlines")
ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig("figures/plate_streamlines.png", dpi=130); plt.close(fig)

# force history + spectrum
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.plot(steps, Fx/qref, label="$C_d$", lw=1)
a1.plot(steps, Fy/qref, label="$C_l$", lw=1)
a1.set(xlabel="time step", ylabel="force coefficient",
       title="Force history (Re=300)"); a1.legend(); a1.grid(alpha=0.3)
sf = freqs*h/U
a2.plot(sf, amp/amp.max(), lw=1.2)
a2.set(xlim=(0, 1.0), xlabel="Strouhal number $fh/U$", ylabel="lift spectrum (norm.)",
       title=f"Shedding peak  St={St:.3f}"); a2.grid(alpha=0.3)
a2.axvline(St, color="r", ls="--", lw=1)
fig.tight_layout(); fig.savefig("figures/plate_forces.png", dpi=130); plt.close(fig)
print("saved plate figures")
