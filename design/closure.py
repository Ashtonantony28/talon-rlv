"""Design closure loop: sizing <-> structures <-> CAD (M1).

A structural-fraction guess gives the first sizing. After that the loop iterates on absolute
stage dry mass: size the vehicle for the current dry masses, build the CAD, re-derive wall
thicknesses from the load cases using masses taken from the CAD solids, and take the CAD dry mass
as the next estimate. CAD dry mass depends only weakly on propellant load (barrel length, wall
thickness), so this fixed point contracts; iterating on structural fraction instead oscillates
because small stages are dominated by fixed masses (fairing, avionics, minimum gauge).
"""
from dataclasses import dataclass
import numpy as np
from . import sizing as sz, structures as st
from .spec import G0
from cad import vehicle as cv


@dataclass
class Design:
    sizing: sz.Sizing
    parts: list
    info: dict
    thickness: dict
    dry: dict
    eps: dict
    margins: dict
    iterations: int
    history: list


def _masses(parts):
    return {p.name: p.props() for p in parts}


def station_margins(spec, s, parts, info, props, thickness=None):
    """Required thickness and margin ratios for every barrel, over all load cases."""
    mat, S_ = spec["material"], spec["structures"]
    FS = S_["factor_of_safety"]
    lump = spec["lumped"]
    z_top, z_sep = info["z_top"], info["z_sep"]
    x_cp = spec["layout"]["nose_cp_fraction"] * spec["layout"]["fairing"]["nose_m"]
    struct = {stg: sum(props[p.name][0] for p in parts if p.stage == stg and p.kind == "structure")
              for stg in ("booster", "upper")}
    stage_lump = {stg: lump[f"{stg}_avionics_kg"] + lump[f"{stg}_misc_fraction"] * struct[stg]
                  for stg in ("booster", "upper")}
    stage_cz = {}
    for stg in stage_lump:
        sel = [props[p.name] for p in parts if p.stage == stg and p.kind != "propellant"]
        stage_cz[stg] = cv.combine(sel)[1][2]
    dry_u = sum(props[p.name][0] for p in parts if p.stage == "upper" and p.kind != "propellant") + stage_lump["upper"]
    n_upper_bo = spec["engines"]["upper"]["count"] * spec["engines"]["upper"]["thrust_vac_n"] / (
        (dry_u + spec["mission"]["payload_kg"]) * G0)
    N = S_["design_max_q_pa"] * np.pi * (spec["diameter_m"] / 2) ** 2 * S_["cn_alpha_per_rad"] * np.radians(
        S_["design_alpha_maxq_deg"])

    def m_above(z0, prop_full):
        m = 0.0
        for p in parts:
            mp, c, _ = props[p.name]
            if p.kind == "propellant":
                if prop_full[p.stage] and z0 < p.station["z_bottom"]:
                    m += mp
            elif c[2] > z0:
                m += mp
        for stg, ml in stage_lump.items():
            if stage_cz[stg] > z0:
                m += ml
        return m

    out = {}
    for p in parts:
        if not p.station.get("R") or p.kind != "structure":
            continue
        z0 = p.station["z_bottom"]
        cases = [
            ("A1_booster_burnout", spec["structures"]["max_axial_g"], {"booster": False, "upper": True}, 0.0),
            ("B_max_q", S_["maxq_axial_g"], {"booster": True, "upper": True}, N * max(z_top - z0 - x_cp, 0.0)),
        ]
        if p.stage == "upper":
            cases.append(("A2_upper_burnout", n_upper_bo, {"booster": False, "upper": False}, 0.0))
        R = p.station["R"]
        t_req = spec["structures"]["min_gauge_m"]
        loads = []
        for name, n, full, M in cases:
            P = m_above(z0, full) * n * G0
            loads.append((name, P, M))
            t_req = max(t_req, st.required_thickness(R, mat, FS, P, M, p.station["pressure"], S_["min_gauge_m"]))
        t_now = p.station["t"]
        ratios = {}
        for name, P, M in loads:
            for mode, r in st.margins(R, t_now, mat, FS, P, M, p.station["pressure"]).items():
                ratios[f"{name}/{mode}"] = r
        gov = min(ratios, key=ratios.get)
        out[p.name] = {"t": t_now, "t_req": t_req, "min_ratio": ratios[gov], "governing": gov,
                       "loads": loads}
    return out, n_upper_bo


def close(spec, max_iter=30, verbose=False):
    guess = spec["sizing"]["struct_fraction_guess"]
    eps = {"booster": guess["booster"], "upper": guess["upper"]}
    tol = spec["sizing"]["closure_tol"]
    thickness = {}
    history = []
    dry_est = None
    for it in range(1, max_iter + 1):
        s = sz.size(spec, eps["booster"], eps["upper"]) if dry_est is None else sz.size(spec, dry=dry_est)
        # thickness sub-iteration at fixed sizing (geometry depends weakly on t)
        for _ in range(4):
            parts, info = cv.build(spec, s, thickness)
            props = _masses(parts)
            mg, _ = station_margins(spec, s, parts, info, props)
            step = spec["structures"]["gauge_step_m"]
            new_t = {k: float(np.ceil(v["t_req"] / step - 1e-9) * step) for k, v in mg.items()}
            if all(abs(new_t[k] - thickness.get(k, -1)) < 1e-5 for k in new_t):
                break
            thickness = new_t
        parts, info = cv.build(spec, s, thickness)
        props = _masses(parts)
        dry = {stg: cv.stage_dry_mass(spec, parts, stg) for stg in ("booster", "upper")}
        stg_s = {"booster": s.booster, "upper": s.upper}
        err = {k: (dry[k] - stg_s[k].dry) / stg_s[k].dry for k in dry}
        history.append({"iter": it, "glow": s.glow, "eps": dict(eps), "dry_cad": dict(dry),
                        "dry_sizing": {k: stg_s[k].dry for k in dry}, "err": err})
        if verbose:
            print(history[-1])
        if max(abs(e) for e in err.values()) < tol:
            mg, _ = station_margins(spec, s, parts, info, props)
            eps = {k: stg_s[k].dry / (stg_s[k].dry + stg_s[k].prop) for k in dry}
            return Design(s, parts, info, thickness, dry, eps, mg, it, history)
        dry_est = dict(dry)
        eps = {k: dry[k] / (dry[k] + stg_s[k].prop) for k in dry}
    raise RuntimeError(f"design did not close in {max_iter} iterations: {history[-1]}")
