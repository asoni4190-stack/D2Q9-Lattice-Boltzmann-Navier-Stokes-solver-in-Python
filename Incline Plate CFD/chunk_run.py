"""
Run an LBM case in resumable chunks so each invocation stays under the
shell time limit.

    python chunk_run.py <case> <chunk_steps>

<case> is 'cyl' or 'plate'. State is checkpointed to state_<case>.npz and
force history is accumulated. When the target step count is reached the
final field + forces are written to result_<case>.npz.
"""
import sys, os, numpy as np, lbm

# Physical mapping of the plate case (see report):
#   chord c = 1.0 m is resolved by 40 lattice cells  ->  1 cell = 2.5 cm.
#   plate thickness = 0.03 m (t/c = 0.03) is ~1 cell, below grid resolution,
#   so the plate is imposed as a THIN BARRIER (plate_mask thickness=1.5, i.e.
#   a few cells) - the minimum that seals the 45-deg bounce-back. Air at 15 C,
#   U = 35 m/s in physical units; the lattice runs at Re = 300 (resolvable).
CASES = {
    "cyl":   dict(nx=360, ny=180, uLB=0.10, Re=100, target=18000,
                  record_from=6000, obstacle=("cyl", 90, 90, 14), Lchar=28),
    "plate": dict(nx=420, ny=180, uLB=0.12, Re=300, target=18000,
                  record_from=6000, obstacle=("plate", 140, 90, 40, 45), Lchar=40),
}

def build_obstacle(cfg):
    o = cfg["obstacle"]
    if o[0] == "cyl":
        return lbm.cylinder_mask(cfg["nx"], cfg["ny"], o[1], o[2], o[3])
    return lbm.plate_mask(cfg["nx"], cfg["ny"], o[1], o[2], o[3], o[4], thickness=1.5)

def main():
    case = sys.argv[1]
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    cfg = CASES[case]
    obs = build_obstacle(cfg)
    state = f"state_{case}.npz"

    if os.path.exists(state):
        d = np.load(state)
        fin0, start = d["fin"], int(d["step"])
        forces = list(map(tuple, d["forces"])) if d["forces"].size else []
    else:
        fin0, start, forces = None, 0, []
        print(f"[{case}] obstacle nodes = {int(obs.sum())}", flush=True)

    steps = min(chunk, cfg["target"] - start)
    if steps <= 0:
        print(f"[{case}] already at target {cfg['target']}"); return

    res = lbm.run(obs, cfg["nx"], cfg["ny"], cfg["uLB"], cfg["Re"], cfg["Lchar"],
                  n_steps=steps, record_from=cfg["record_from"], force_every=20,
                  verbose=True, fin0=fin0, start_step=start)

    forces += list(map(tuple, res["forces"]))
    np.savez(state, fin=res["fin"], step=res["end_step"],
             forces=np.array(forces) if forces else np.zeros((0, 3)))
    print(f"[{case}] now at step {res['end_step']} / {cfg['target']}", flush=True)

    if res["end_step"] >= cfg["target"]:
        F = np.array(forces)
        D, U = cfg["Lchar"], cfg["uLB"]
        St, fpk, _ = lbm.strouhal(F, D, U)
        m = F[:, 0] > (cfg["record_from"] + 4000)
        Cd = F[m, 1].mean() / (0.5 * U**2 * D)
        Clrms = np.sqrt((F[m, 2]**2).mean()) / (0.5 * U**2 * D)
        np.savez(f"result_{case}.npz", u=res["u"], rho=res["rho"],
                 forces=F, obs=obs, St=St, Cd=Cd, D=D, U=U,
                 nx=cfg["nx"], ny=cfg["ny"])
        print(f"[{case}] DONE  St={St:.4f}  Cd={Cd:.3f}  Cl_rms={Clrms:.3f}", flush=True)

if __name__ == "__main__":
    main()
