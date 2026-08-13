"""
Inclined Flat Plate — Aerodynamic Force Analysis (design condition)
===================================================================

Analytical / empirical aerodynamic analysis of a flat sheet in a uniform
air stream, evaluated at the ACTUAL operating condition:

    plate size      1.0 m  x  1.0 m
    angle of attack 45 deg  (angle between the plate chord and the flow)
    air speed       35 m/s

At this condition the Reynolds number is ~2.4e6, so the flow is fully
turbulent and massively separated. In this regime the aerodynamic force
on a flat plate is pressure-dominated and is estimated with well
established empirical relations rather than attached-flow (thin-airfoil)
theory, which is invalid past stall.

Empirical normal-force coefficient for a flat plate in fully separated
flow (Hoerner, "Fluid-Dynamic Lift/Drag"), 2-D / per-unit-span:

    C_N(alpha) = 1 / (0.222 + 0.283 / sin(alpha))

    - alpha = 90 deg (normal plate) -> C_N = 1.98  (classic 2-D value)
    - the resultant force acts normal to the plate; it is resolved into
      lift (perpendicular to the flow) and drag (parallel to the flow):
          Cd = C_N * sin(alpha)
          Cl = C_N * cos(alpha)
"""

import numpy as np

# ----------------------------- inputs ---------------------------------
U = 35.0            # free-stream speed [m/s]
chord = 1.0         # plate chord (streamwise dimension when at alpha) [m]
span = 1.0          # plate span [m]
alpha_deg = 45.0    # angle of attack [deg]

# air at 15 C, sea level (ISA)
rho = 1.225         # density [kg/m^3]
mu = 1.802e-5       # dynamic viscosity [Pa.s]
a_sound = 340.3     # speed of sound [m/s]

# --------------------------- derived ----------------------------------
nu = mu / rho
alpha = np.radians(alpha_deg)
A = chord * span                     # plate planform area [m^2]
q = 0.5 * rho * U**2                 # dynamic pressure [Pa]
Re = U * chord / nu
Mach = U / a_sound

# Hoerner separated-flow normal-force coefficient (2-D)
C_N = 1.0 / (0.222 + 0.283 / np.sin(alpha))
Cd = C_N * np.sin(alpha)
Cl = C_N * np.cos(alpha)

# forces (per given 1 m span; A = 1 m^2)
N_force = C_N * q * A                 # normal to plate [N]
Drag = Cd * q * A                     # along flow [N]
Lift = Cl * q * A                     # perpendicular to flow [N]

# turbulent skin-friction drag (flat plate, Prandtl) for comparison
Cf = 0.074 / Re**0.2
D_friction = Cf * q * A               # friction acts on wetted area ~A

# ----------------------------- report ---------------------------------
def line(): print("-" * 60)

print("INCLINED FLAT PLATE - AERODYNAMIC ANALYSIS")
line()
print(f"  plate               : {chord:.2f} m x {span:.2f} m  (A = {A:.2f} m^2)")
print(f"  angle of attack     : {alpha_deg:.1f} deg")
print(f"  free-stream speed   : {U:.1f} m/s")
print(f"  air (15 C, ISA)     : rho={rho} kg/m^3, nu={nu:.3e} m^2/s")
line()
print("  Flow regime")
print(f"    dynamic pressure  q = {q:8.1f} Pa")
print(f"    Reynolds number  Re = {Re:8.3e}   -> turbulent, separated")
print(f"    Mach number       M = {Mach:8.3f}   -> incompressible (M<0.3)")
line()
print("  Force coefficients (Hoerner, 2-D separated flow)")
print(f"    normal force  C_N = {C_N:6.3f}")
print(f"    drag          C_d = {Cd:6.3f}")
print(f"    lift          C_l = {Cl:6.3f}")
line()
print("  Forces on the plate")
print(f"    normal force   N = {N_force:8.1f} N")
print(f"    drag           D = {Drag:8.1f} N")
print(f"    lift           L = {Lift:8.1f} N")
print(f"    (skin friction D_f = {D_friction:6.1f} N  -> {100*D_friction/Drag:.1f}% of drag)")
line()
print("  Note: values are 2-D (per unit span). A finite square plate")
print("  (aspect ratio 1) sheds flow around all four edges, lowering the")
print("  coefficients by roughly 40% (e.g. normal-plate Cd ~1.17 vs 1.98).")

# ------------------- coefficient vs angle (for a plot) ----------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aa = np.radians(np.linspace(1, 90, 200))
    cn = 1.0 / (0.222 + 0.283 / np.sin(aa))
    cd = cn * np.sin(aa)
    cl = cn * np.cos(aa)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(np.degrees(aa), cn, label="$C_N$ (normal)", lw=2)
    ax.plot(np.degrees(aa), cd, label="$C_d$ (drag)", lw=2)
    ax.plot(np.degrees(aa), cl, label="$C_l$ (lift)", lw=2)
    ax.axvline(45, color="grey", ls="--", lw=1)
    ax.plot(45, Cd, "o", color="k"); ax.plot(45, Cl, "o", color="k")
    ax.annotate(f"  design point 45 deg\n  $C_d$={Cd:.2f}, $C_l$={Cl:.2f}",
                (45, Cd), fontsize=9)
    ax.set(xlabel="angle of attack [deg]", ylabel="coefficient",
           title="Flat-plate force coefficients vs. angle (Hoerner, 2-D)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/coefficients_vs_angle.png", dpi=130)
    print("\nsaved figures/coefficients_vs_angle.png")
