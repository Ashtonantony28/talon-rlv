"""M2: aero analytic checks, ascent-sim consistency checks and the optimal ascent itself."""
import numpy as np
import pytest
from design import spec as dspec
from aero.coeffs import Aero
from sim import gravity, atmosphere
from sim.ascent import Vehicle, simulate, circularization, orbit_elements
from gnc import ascent_opt

SPEC = dspec.load()


def test_slender_body_normal_force_and_cp():
    a = Aero(SPEC)
    # small-alpha slope is 2 per radian; CN(alpha) = 2 sin a cos a
    assert np.isclose(a.CN(1e-6) / 1e-6, 2.0, rtol=1e-9)
    assert np.isclose(a.CN(np.radians(30)), 2 * np.sin(np.radians(30)) * np.cos(np.radians(30)))
    # cone CP at 2/3 of nose length from the tip
    assert np.isclose(a.x_cp_from_tip, 2 / 3 * SPEC["layout"]["fairing"]["nose_m"], rtol=1e-3)


def test_aero_force_directions():
    a = Aero(SPEC)
    _, _, rho, snd = atmosphere.properties(5000.0)
    F, q = a.forces(np.array([300.0, 0.0, 0.0]), rho, snd)
    assert F[0] < 0 and np.allclose(F[1:], 0)                      # pure drag at zero alpha
    assert np.isclose(-F[0], q * a.S * a.CA(300.0 / snd), rtol=1e-12)
    F2, _ = a.forces(np.array([300.0, 30.0, 0.0]), rho, snd)
    assert F2[1] < 0                                                # normal force opposes crossflow


def test_booster_thrust_matches_spec_at_sea_level_and_vacuum(design):
    veh = Vehicle.from_design(SPEC, design)
    p0 = atmosphere.properties(0.0)[1]
    e = SPEC["engines"]["booster"]
    assert np.isclose(veh.b_thrust(p0, 1.0), e["count"] * e["thrust_sl_n"], rtol=1e-12)
    assert np.isclose(veh.b_thrust(0.0, 1.0) / (e["count"] * veh.b_mdot * 9.80665), e["isp_vac_s"], rtol=1e-12)


def test_orbit_elements_circular_and_elliptic():
    r0 = gravity.R_EQ + 400e3
    vc = np.sqrt(gravity.MU / r0)
    apo, peri = orbit_elements(np.array([r0, 0, 0]), np.array([0, vc, 0]))
    assert np.isclose(apo, 400e3, atol=1e-3) and np.isclose(peri, 400e3, atol=1e-3)
    ra = gravity.R_EQ + 800e3
    a = 0.5 * (r0 + ra)
    vp = np.sqrt(gravity.MU * (2 / r0 - 1 / a))
    apo, peri = orbit_elements(np.array([r0, 0, 0]), np.array([0, vp, 0]))
    assert np.isclose(apo, 800e3, atol=1e-2) and np.isclose(peri, 400e3, atol=1e-2)


@pytest.fixture(scope="module")
def ascent(design):
    veh = Vehicle.from_design(SPEC, design)
    p = ascent_opt.load()
    res = simulate(SPEC, veh, p)
    return veh, p, res


def test_reaches_target_orbit_within_1pct(ascent):
    veh, _, res = ascent
    assert res.ok, res.reason
    c = circularization(SPEC, veh, res)
    target = SPEC["ascent"]["target_apogee_km"]
    assert abs(c["final_alt_km"] - target) / target < 0.01
    assert res.events["seco"]["perigee"] / 1e3 >= SPEC["ascent"]["min_perigee_km"] - 0.5   # 0.5 km: soft-penalty floor
    assert c["margin_kg"] > 0


def test_loads_within_design_limits(ascent):
    _, _, res = ascent
    st = SPEC["structures"]
    assert max(res.log["q"]) <= st["design_max_q_pa"]
    assert max(res.log["accel_g"]) <= st["max_axial_g"] + 1e-6


def test_committed_params_are_locally_optimal(design):
    veh = Vehicle.from_design(SPEC, design)
    p = ascent_opt.load()
    x = np.array([p.kick_deg, p.theta0_deg, p.c_per_s])
    c0 = ascent_opt.cost(SPEC, veh, x, 0.2)
    for i, h in enumerate([0.1, 0.5, 5e-5]):
        for sgn in (-1, 1):
            xx = x.copy()
            xx[i] += sgn * h
            assert ascent_opt.cost(SPEC, veh, xx, 0.2) >= c0 - 0.5   # 0.5 kg: optimiser fatol + dt noise
