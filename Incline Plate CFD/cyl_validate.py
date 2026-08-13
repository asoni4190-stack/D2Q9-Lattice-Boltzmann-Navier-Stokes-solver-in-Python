import numpy as np, lbm

nx, ny = 360, 180
cx, cy, r = 90, 90, 14
D, U = 2 * r, 0.1
obs = lbm.cylinder_mask(nx, ny, cx, cy, r)

res = lbm.run(obs, nx, ny, uLB=U, Re=100, Lchar=D,
              n_steps=18000, record_from=6000, force_every=20, verbose=True)

F = res["forces"]
St, fpk, spec = lbm.strouhal(F, D, U)
m = F[:, 0] > 10000
Cd = F[m, 1].mean() / (0.5 * U**2 * D)
Cl_rms = np.sqrt((F[m, 2]**2).mean()) / (0.5 * U**2 * D)

print(f"RESULT St={St:.4f} Cd={Cd:.3f} Cl_rms={Cl_rms:.3f}")
np.savez("cyl_result.npz", u=res["u"], forces=F, obs=obs,
         St=St, Cd=Cd, D=D, U=U)
print("saved cyl_result.npz")
