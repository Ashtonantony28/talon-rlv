"""3-DOF ascent simulation to orbit (M2).

Point mass in an Earth-centred inertial frame aligned with ECEF at t = 0: J2 gravity, rotating
US76 atmosphere, Mach-dependent drag along the air-relative velocity (zero-alpha flight), thrust
corrected for nozzle back-pressure. Altitude is measured above a sphere of the WGS-84 equatorial
radius (documented simplification; consistent for launch and orbit).

Attitude is assumed to track the guidance direction exactly; the 6-DOF TVC loop that makes that
true is M3's job.
"""
from dataclasses import dataclass, field
import numpy as np
from design.spec import G0
from . import gravity, atmosphere
from aero.coeffs import Aero
from gnc.ascent_guidance import BoosterGuidance, UpperGuidance, AscentParams

OMEGA_E = 7.2921159e-5


@dataclass
class Vehicle:
    """Masses and engines for the ascent, taken from the closed M1 design."""
    payload: float
    booster_dry: float
    booster_prop_ascent: float
    booster_prop_recovery: float
    upper_dry: float          # includes fairing
    upper_prop: float
    fairing: float
    b_n: int
    b_thrust_sl: float
    b_isp_sl: float
    b_isp_vac: float
    b_throttle_min: float
    u_thrust_vac: float
    u_isp_vac: float
    u_throttle_min: float

    @classmethod
    def from_design(cls, spec, d):
        e = spec["engines"]
        fairing = sum(p.props()[0] for p in d.parts if p.kind == "fairing")
        return cls(payload=d.sizing.payload, booster_dry=d.dry["booster"],
                   booster_prop_ascent=d.sizing.booster.prop_ascent,
                   booster_prop_recovery=d.sizing.booster.prop_recovery,
                   upper_dry=d.dry["upper"], upper_prop=d.sizing.upper.prop, fairing=fairing,
                   b_n=e["booster"]["count"], b_thrust_sl=e["booster"]["thrust_sl_n"],
                   b_isp_sl=e["booster"]["isp_sl_s"], b_isp_vac=e["booster"]["isp_vac_s"],
                   b_throttle_min=e["booster"]["throttle_min"],
                   u_thrust_vac=e["upper"]["count"] * e["upper"]["thrust_vac_n"], u_isp_vac=e["upper"]["isp_vac_s"],
                   u_throttle_min=e["upper"]["throttle_min"])

    @property
    def b_mdot(self):            # per engine at full throttle
        return self.b_thrust_sl / (self.b_isp_sl * G0)

    @property
    def b_exit_area(self):       # implied by the SL / vacuum Isp pair
        return self.b_mdot * G0 * (self.b_isp_vac - self.b_isp_sl) / atmosphere.properties(0.0)[1]

    def b_thrust(self, p_amb, throttle):
        return self.b_n * (throttle * self.b_mdot * self.b_isp_vac * G0 - p_amb * self.b_exit_area)


def orbit_elements(r, v):
    """(apogee alt, perigee alt) in m above the reference sphere, for a Keplerian orbit."""
    mu = gravity.MU
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    eps = vn * vn / 2 - mu / rn
    h = np.linalg.norm(np.cross(r, v))
    if eps >= 0:
        return np.inf, -np.inf
    a = -mu / (2 * eps)
    e = np.sqrt(max(0.0, 1 + 2 * eps * h * h / (mu * mu)))
    return a * (1 + e) - gravity.R_EQ, a * (1 - e) - gravity.R_EQ


@dataclass
class AscentResult:
    ok: bool
    reason: str
    t: np.ndarray
    X: np.ndarray                     # [r, v, m]
    log: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    log_max_alt_km: float = 0.0


def launch_state(cfg):
    ls = cfg["launch_site"]
    lat, lon = np.radians(ls["lat_deg"]), np.radians(ls["lon_deg"])
    R = gravity.R_EQ + ls["alt_m"]
    r = R * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    v = np.cross([0, 0, OMEGA_E], r)
    return r, v


def simulate(spec, veh: Vehicle, p: AscentParams, dt=None, record=True, allow_overrun=False):
    """Fly the ascent. `allow_overrun` lets the upper stage burn past depletion (propellant left goes
    negative) so the optimiser sees a smooth objective even for an under-performing vehicle."""
    cfg = spec["ascent"]
    dt = dt or cfg["dt_s"]
    aero = Aero(spec)
    st = spec["structures"]
    q_bucket = cfg["maxq_bucket_frac"] * st["design_max_q_pa"]
    n_max = st["max_axial_g"]
    target_apo = cfg["target_apogee_km"] * 1e3
    az = np.radians(cfg["azimuth_deg"])
    bg, ug = BoosterGuidance(cfg, p, az), UpperGuidance(p)
    w_e = np.array([0, 0, OMEGA_E])

    r, v = launch_state(cfg)
    m = veh.payload + veh.upper_dry + veh.upper_prop + veh.booster_dry + veh.booster_prop_ascent + veh.booster_prop_recovery
    b_prop_left = veh.booster_prop_ascent
    u_prop_left = veh.upper_prop
    phase = "booster"
    t = 0.0
    t_stage = None
    fairing_on = True
    ev = {}
    max_alt = 0.0
    L = {k: [] for k in ("t", "alt", "vel", "q", "mach", "accel_g", "throttle", "phase")}
    T_rec, X_rec = [], []

    def env(r, v):
        alt = np.linalg.norm(r) - gravity.R_EQ
        _, p_amb, rho, a = atmosphere.properties(alt)
        v_rel = v - np.cross(w_e, r)
        return alt, p_amb, rho, a, v_rel

    def guidance(t, r, v, m, phase, base_throttle):
        """(thrust direction, throttle) as a continuous function of time and state."""
        v_rel = v - np.cross(w_e, r)
        if phase == "booster":
            d = bg.direction(t, r, v_rel)
            _, p_amb, *_ = atmosphere.properties(np.linalg.norm(r) - gravity.R_EQ)
            thr = base_throttle
            T_full = veh.b_thrust(p_amb, thr)
            if T_full / m > n_max * G0:
                thr = max(veh.b_throttle_min, thr * n_max * G0 * m / T_full)
            return d, thr
        if phase == "upper":
            thr = 1.0
            if veh.u_thrust_vac / m > n_max * G0:
                thr = max(veh.u_throttle_min, n_max * G0 * m / veh.u_thrust_vac)
            return ug.direction(t, r, v), thr
        return v_rel / max(np.linalg.norm(v_rel), 1e-9), 0.0

    def accel(t, r, v, m, phase, base_throttle):
        thr_dir, throttle = guidance(t, r, v, m, phase, base_throttle)
        alt, p_amb, rho, a, v_rel = env(r, v)
        V = np.linalg.norm(v_rel)
        q = 0.5 * rho * V * V
        engines_on = phase in ("booster", "upper")
        drag = -q * aero.S * aero.CA(V / a, engines_on) * (v_rel / V) if V > 1e-6 else np.zeros(3)
        if phase == "booster":
            T, mdot = veh.b_thrust(p_amb, throttle), veh.b_n * veh.b_mdot * throttle
        elif phase == "upper":
            T, mdot = veh.u_thrust_vac * throttle, veh.u_thrust_vac * throttle / (veh.u_isp_vac * G0)
        else:
            T, mdot = 0.0, 0.0
        acc = gravity.j2(r) + (T * thr_dir + drag) / m
        return acc, mdot, q, V / a, T, throttle

    while True:
        alt, p_amb, rho, a, v_rel = env(r, v)
        max_alt = max(max_alt, alt)
        if alt < -1.0:
            return AscentResult(False, "impact", np.array(T_rec), np.array(X_rec), L, ev, max_alt / 1e3)
        # guidance (evaluated continuously inside the integrator) + throttle (per step, except the
        # acceleration limit, which is continuous)
        if phase == "booster":
            bg.update(t, r, v_rel)
            q_now = 0.5 * rho * np.dot(v_rel, v_rel)
            base_throttle = cfg["maxq_throttle"] if q_now > q_bucket else 1.0
        elif phase == "upper":
            ug.start(t)
            base_throttle = 1.0
        else:
            base_throttle = 0.0

        # RK4 on (r, v, m)
        def f(tt, rr, vv, mm):
            acc, mdot, *_ = accel(tt, rr, vv, mm, phase, base_throttle)
            return vv, acc, -mdot

        def rk4(h):
            k1 = f(t, r, v, m)
            k2 = f(t + h / 2, r + h / 2 * k1[0], v + h / 2 * k1[1], m + h / 2 * k1[2])
            k3 = f(t + h / 2, r + h / 2 * k2[0], v + h / 2 * k2[1], m + h / 2 * k2[2])
            k4 = f(t + h, r + h * k3[0], v + h * k3[1], m + h * k3[2])
            return (r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                    v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                    h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))

        def event_crossed(rn, vn, dmn):
            if phase == "booster":
                return b_prop_left + dmn <= 0
            if phase == "upper":
                return orbit_elements(rn, vn)[0] >= target_apo
            return False

        if record:
            acc, _, q, mach, T, throttle = accel(t, r, v, m, phase, base_throttle)
            ng = np.linalg.norm(acc - gravity.j2(r)) / G0
            for k, val in zip(L, (t, alt, np.linalg.norm(v_rel), q, mach, ng, throttle, phase)):
                L[k].append(val)
            T_rec.append(t)
            X_rec.append(np.concatenate([r, v, [m]]))
        h = dt
        rn, vn, dm = rk4(h)
        if event_crossed(rn, vn, dm):
            # locate MECO / SECO inside the step (bisection on the step length) so events do not
            # depend on dt
            lo, hi = 0.0, dt
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if event_crossed(*rk4(mid)) else (mid, hi)
            h = hi
            rn, vn, dm = rk4(h)
        r, v, m = rn, vn, m + dm
        t += h

        if phase == "booster":
            b_prop_left += dm
            if b_prop_left <= 0:
                alt_, *_rest, v_rel_ = env(r, v)
                ev["meco"] = {"t": t, "alt": alt_, "v_rel": float(np.linalg.norm(v_rel_)), "r": r.copy(), "v": v.copy(),
                              "flight_path_deg": float(np.degrees(np.arcsin(np.dot(v_rel_, r) / np.linalg.norm(v_rel_) / np.linalg.norm(r))))}
                m -= veh.booster_dry + veh.booster_prop_recovery + b_prop_left
                phase, t_stage = "coast", t
        elif phase == "coast":
            if t - t_stage >= cfg["staging_coast_s"]:
                phase = "upper"
                ev["upper_ignition"] = {"t": t}
        elif phase == "upper":
            u_prop_left += dm
            if fairing_on and 0.5 * rho * np.dot(v_rel, v_rel) < cfg["fairing_jettison_q_pa"]:
                m -= veh.fairing
                fairing_on = False
                ev["fairing_sep"] = {"t": t, "alt": alt}
            apo, peri = orbit_elements(r, v)
            if apo >= target_apo:
                ev["seco"] = {"t": t, "alt": alt, "apogee": apo, "perigee": peri, "prop_left": u_prop_left,
                              "m": m, "r": r.copy(), "v": v.copy()}
                if fairing_on:
                    return AscentResult(False, "fairing still on at SECO", np.array(T_rec), np.array(X_rec), L, ev)
                return AscentResult(True, "seco", np.array(T_rec), np.array(X_rec), L, ev)
            if u_prop_left <= 0 and not allow_overrun:
                ev["depletion"] = {"t": t, "apogee": apo, "perigee": peri, "r": r.copy(), "v": v.copy()}
                return AscentResult(False, "upper stage depleted", np.array(T_rec), np.array(X_rec), L, ev)
        if t > 2000:
            return AscentResult(False, "timeout", np.array(T_rec), np.array(X_rec), L, ev, max_alt / 1e3)


def circularization(spec, veh, res):
    """Impulsive apogee burn to circularise; returns dv, propellant needed and remaining margin."""
    s = res.events["seco"]
    mu = gravity.MU
    ra, rp = gravity.R_EQ + s["apogee"], gravity.R_EQ + s["perigee"]
    a = 0.5 * (ra + rp)
    v_apo = np.sqrt(mu * (2 / ra - 1 / a))
    dv = np.sqrt(mu / ra) - v_apo
    ve = veh.u_isp_vac * G0
    prop_needed = s["m"] * (1 - np.exp(-dv / ve))
    return {"dv": dv, "prop_needed": prop_needed, "margin_kg": s["prop_left"] - prop_needed,
            "final_alt_km": s["apogee"] / 1e3}
