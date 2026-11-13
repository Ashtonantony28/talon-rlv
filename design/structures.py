"""Thin-wall structural sizing of tank barrels and dry bays.

Load cases (limit loads, multiplied by the ultimate factor of safety FS):
  A. max axial acceleration (booster near burnout): compression = m_above * n_axial * g0
  B. max-q: compression = m_above * n_maxq * g0 plus bending from the aerodynamic normal force
     N = q S CN_alpha alpha acting at the nose centre of pressure (no inertial relief: conservative)
  P. internal pressure (tanks only): hoop stress p R / t

Failure modes: yield (von Mises not needed for the governing uniaxial/hoop states) and axial
buckling of an unstiffened cylinder with the NASA SP-8007 knockdown factor
    gamma = 1 - 0.901 (1 - exp(-phi)),   phi = sqrt(R/t) / 16
    sigma_cr = gamma * E / sqrt(3 (1 - nu^2)) * t / R      (0.605 E t/R for nu = 0.3)
Pressure stabilisation is ignored (conservative).
"""
import numpy as np
from .spec import G0


def knockdown(R, t):
    phi = np.sqrt(R / t) / 16.0
    return 1.0 - 0.901 * (1.0 - np.exp(-phi))


def sigma_cr(R, t, E, nu):
    return knockdown(R, t) * E / np.sqrt(3.0 * (1.0 - nu * nu)) * t / R


def hoop_stress(p, R, t):
    return p * R / t


def compressive_stress(R, t, axial_force, bending_moment):
    """Peak compressive stress in a thin cylinder: P/(2 pi R t) + M/(pi R^2 t)."""
    return axial_force / (2 * np.pi * R * t) + bending_moment / (np.pi * R * R * t)


def margins(R, t, mat, FS, axial_force, bending_moment, pressure=0.0):
    """Ratio capacity / (FS * applied) for each mode; >= 1 means positive margin at FS."""
    sc = compressive_stress(R, t, axial_force, bending_moment)
    out = {
        "buckling": sigma_cr(R, t, mat["E"], mat["nu"]) / (FS * sc) if sc > 0 else np.inf,
        "yield_compression": mat["yield_pa"] / (FS * sc) if sc > 0 else np.inf,
    }
    if pressure > 0:
        out["yield_hoop"] = mat["yield_pa"] / (FS * hoop_stress(pressure, R, t))
    return out


def required_thickness(R, mat, FS, axial_force, bending_moment, pressure, t_min):
    """Smallest t >= t_min with every margin ratio >= 1 (bisection; margins rise monotonically in t)."""
    def ok(t):
        return min(margins(R, t, mat, FS, axial_force, bending_moment, pressure).values()) >= 1.0
    lo, hi = t_min, t_min
    if ok(lo):
        return lo
    while not ok(hi):
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def station_loads(spec, m_above, x_from_nose, x_cp_from_nose):
    """Limit axial force and bending moment at a station, for cases A and B; returns list of (P, M)."""
    st = spec["structures"]
    S = np.pi * (spec["diameter_m"] / 2) ** 2
    N = st["design_max_q_pa"] * S * st["cn_alpha_per_rad"] * np.radians(st["design_alpha_maxq_deg"])
    arm = max(x_from_nose - x_cp_from_nose, 0.0)
    return [
        (m_above * st["max_axial_g"] * G0, 0.0),
        (m_above * st["maxq_axial_g"] * G0, N * arm),
    ]
