"""Two-stage sizing by the staged rocket equation.

Unknown: how much of the orbital dv each stage provides. Booster additionally carries
propellant for its own recovery (boostback/entry/landing), burned after separation with only
the empty booster. Structural fraction eps = dry / (dry + usable propellant) per stage.

For a fixed dv split the stage masses follow in closed form from the top down; the split that
minimises gross lift-off mass is found by a bounded 1-D search.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize_scalar
from .spec import G0


@dataclass
class Stage:
    gross: float          # wet mass of this stage alone
    dry: float
    prop_ascent: float
    prop_recovery: float
    dv_ascent: float
    isp: float

    @property
    def prop(self):
        return self.prop_ascent + self.prop_recovery


@dataclass
class Sizing:
    glow: float
    payload: float
    booster: Stage
    upper: Stage
    dv_split_booster: float

    def as_dict(self):
        return {"glow": self.glow, "payload": self.payload, "dv_split_booster": self.dv_split_booster,
                "booster": asdict(self.booster), "upper": asdict(self.upper)}


def _stage_on_top_of(m_above, dv, isp, eps, dv_recovery=0.0):
    """Size one stage carrying m_above. Returns Stage or None if infeasible.

    Ascent: (m_above + dry + p_rec + p_asc) / (m_above + dry + p_rec) = exp(dv/ve)
    Recovery: (dry + p_rec) / dry = exp(dv_rec/ve)
    eps: dry = eps * (dry + p_asc + p_rec)
    """
    ve = isp * G0
    Ra = np.exp(dv / ve)
    Rr = np.exp(dv_recovery / ve)
    # unknown dry d: p_rec = d (Rr - 1); p_asc = (Ra - 1)(m_above + d Rr)
    # eps (d + p_asc + p_rec) = d  ->  d [1 - eps (Rr + (Ra - 1) Rr)] = eps (Ra - 1) m_above
    denom = 1.0 - eps * Rr * Ra
    if denom <= 0:
        return None
    d = eps * (Ra - 1.0) * m_above / denom
    p_rec = d * (Rr - 1.0)
    p_asc = (Ra - 1.0) * (m_above + d * Rr)
    return Stage(gross=d + p_rec + p_asc, dry=d, prop_ascent=p_asc, prop_recovery=p_rec,
                 dv_ascent=dv, isp=isp)


def _stage_with_dry(m_above, dv, isp, dry, dv_recovery=0.0):
    """Size one stage whose dry mass is known (closure mode): always feasible, closed form."""
    ve = isp * G0
    Rr = np.exp(dv_recovery / ve)
    p_rec = dry * (Rr - 1.0)
    p_asc = (np.exp(dv / ve) - 1.0) * (m_above + dry * Rr)
    return Stage(gross=dry + p_rec + p_asc, dry=dry, prop_ascent=p_asc, prop_recovery=p_rec,
                 dv_ascent=dv, isp=isp)


def size(spec, eps_booster=None, eps_upper=None, dry=None):
    """Optimal two-stage sizing. Give structural fractions (eps_*) for a first estimate, or
    absolute stage dry masses `dry={"booster": kg, "upper": kg}` for the CAD closure loop."""
    mis, eng = spec["mission"], spec["engines"]
    payload, dv_tot = mis["payload_kg"], mis["dv_to_orbit_mps"]
    # booster ascent Isp: simple trajectory-averaged value between SL and vac (refined in M2)
    isp_b = 0.5 * (eng["booster"]["isp_sl_s"] + eng["booster"]["isp_vac_s"])
    isp_u = eng["upper"]["isp_vac_s"]

    def build(dv_b):
        if dry is not None:
            up = _stage_with_dry(payload, dv_tot - dv_b, isp_u, dry["upper"])
            bo = _stage_with_dry(payload + up.gross, dv_b, isp_b, dry["booster"], mis["booster_recovery_dv_mps"])
            return Sizing(glow=payload + up.gross + bo.gross, payload=payload, booster=bo, upper=up,
                          dv_split_booster=dv_b)
        up = _stage_on_top_of(payload, dv_tot - dv_b, isp_u, eps_upper)
        if up is None:
            return None
        bo = _stage_on_top_of(payload + up.gross, dv_b, isp_b, eps_booster, mis["booster_recovery_dv_mps"])
        if bo is None:
            return None
        return Sizing(glow=payload + up.gross + bo.gross, payload=payload, booster=bo, upper=up,
                      dv_split_booster=dv_b)

    def cost(dv_b):
        s = build(dv_b)
        return s.glow if s else 1e12

    lo = max(0.15 * dv_tot, mis.get("booster_dv_min_mps", 0.0))
    res = minimize_scalar(cost, bounds=(lo, 0.75 * dv_tot), method="bounded",
                          options={"xatol": 0.1})
    s = build(res.x)
    if s is None:
        raise ValueError("no feasible dv split for these structural fractions")
    return s


def engine_check(spec, s):
    """Thrust-to-weight checks against the engine set in the spec."""
    eng, mis = spec["engines"], spec["mission"]
    tw_liftoff = eng["booster"]["count"] * eng["booster"]["thrust_sl_n"] / (s.glow * G0)
    m_upper_ign = s.payload + s.upper.gross
    tw_upper = eng["upper"]["count"] * eng["upper"]["thrust_vac_n"] / (m_upper_ign * G0)
    return {"tw_liftoff": tw_liftoff, "tw_upper": tw_upper,
            "liftoff_ok": tw_liftoff >= mis["liftoff_tw"],
            "upper_ok": tw_upper >= mis["upper_stage_initial_tw"]}
