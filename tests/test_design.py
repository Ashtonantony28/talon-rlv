"""M1: analytic checks of sizing/structures/mass-property pipeline, then design closure."""
import numpy as np
import pytest
from build123d import Cylinder
from design import spec as dspec, sizing, structures as st
from design.spec import G0
from cad import vehicle as cv

SPEC = dspec.load()


def test_stage_satisfies_rocket_equation():
    s = sizing._stage_on_top_of(500.0, 4000.0, 320.0, 0.09, dv_recovery=1500.0)
    ve = 320.0 * G0
    m0 = 500.0 + s.gross
    m1 = 500.0 + s.dry + s.prop_recovery
    assert np.isclose(ve * np.log(m0 / m1), 4000.0, rtol=1e-12)
    assert np.isclose(ve * np.log((s.dry + s.prop_recovery) / s.dry), 1500.0, rtol=1e-12)
    assert np.isclose(s.dry / (s.dry + s.prop), 0.09, rtol=1e-12)


def test_two_stage_split_is_optimal_and_meets_dv():
    s = sizing.size(SPEC, 0.08, 0.10)
    assert np.isclose(s.booster.dv_ascent + s.upper.dv_ascent, SPEC["mission"]["dv_to_orbit_mps"])
    # perturbing the split must not reduce GLOW (only upward when the staging floor is active)
    floor = SPEC["mission"]["booster_dv_min_mps"]
    assert s.booster.dv_ascent >= floor - 1.0
    steps = (50.0,) if s.booster.dv_ascent < floor + 1.0 else (-50.0, 50.0)
    for d in steps:
        up = sizing._stage_on_top_of(s.payload, s.upper.dv_ascent - d, s.upper.isp, 0.10)
        bo = sizing._stage_on_top_of(s.payload + up.gross, s.booster.dv_ascent + d, s.booster.isp, 0.08,
                                     SPEC["mission"]["booster_recovery_dv_mps"])
        assert s.payload + up.gross + bo.gross >= s.glow - 1e-6


def test_infeasible_structural_fraction_is_rejected():
    assert sizing._stage_on_top_of(100.0, 9000.0, 300.0, 0.2) is None


def test_sp8007_knockdown_and_classical_buckling():
    # R/t = 1024 -> phi = 2 -> gamma = 1 - 0.901 (1 - e^-2)
    assert np.isclose(st.knockdown(1.024, 1e-3), 1 - 0.901 * (1 - np.exp(-2.0)), rtol=1e-12)
    # classical (gamma -> 1 for very thick) equals 0.6052 E t / R at nu = 0.3
    sc = st.sigma_cr(1.0, 1e-3, 70e9, 0.3) / st.knockdown(1.0, 1e-3)
    assert np.isclose(sc, 70e9 * 1e-3 / np.sqrt(3 * 0.91), rtol=1e-12)


def test_required_thickness_is_tight():
    mat = SPEC["material"]
    t = st.required_thickness(0.9, mat, 1.4, 2.0e6, 3.0e5, 3e5, 1e-4)
    m = st.margins(0.9, t, mat, 1.4, 2.0e6, 3.0e5, 3e5)
    assert min(m.values()) >= 1.0
    assert min(st.margins(0.9, t * 0.999, mat, 1.4, 2.0e6, 3.0e5, 3e5).values()) < 1.0


def test_cad_mass_properties_match_hollow_cylinder():
    R, t, L, rho = 0.9, 0.004, 3.0, 2710.0
    p = cv.Part("tube", cv.tube(R, t, L, 1.0), rho, "booster", "structure")
    m, c, J = p.props()
    Ri = R - t
    m_ref = rho * np.pi * (R * R - Ri * Ri) * L
    assert np.isclose(m, m_ref, rtol=1e-9)
    assert np.allclose(c, [0, 0, 1.0 + L / 2], atol=1e-9)
    Izz = 0.5 * m_ref * (R * R + Ri * Ri)
    Ixx = m_ref * (3 * (R * R + Ri * Ri) + L * L) / 12
    assert np.allclose(np.diag(J), [Ixx, Ixx, Izz], rtol=1e-9)


def test_parallel_axis_combination():
    a = cv.Part("a", cv.Pos(0, 0, 0) * Cylinder(0.5, 1.0), 1000.0, "booster", "structure").props()
    b = cv.Part("b", cv.Pos(0, 0, 2.0) * Cylinder(0.5, 1.0), 1000.0, "booster", "structure").props()
    m, c, J = cv.combine([a, b])
    whole = cv.Part("w", cv.Pos(0, 0, 0) * Cylinder(0.5, 1.0) + cv.Pos(0, 0, 2.0) * Cylinder(0.5, 1.0),
                    1000.0, "booster", "structure").props()
    assert np.isclose(m, whole[0]) and np.allclose(c, whole[1]) and np.allclose(J, whole[2], rtol=1e-9)


def test_tank_volume_formula_matches_cad():
    R, h, L = 0.9, 0.636, 2.0
    liq = cv.liquid_solid(R, h, L, 0.0)
    assert np.isclose(liq.volume, cv.tank_volume(R, h, L), rtol=1e-9)
    assert np.isclose(cv.liquid_solid(R, h, 0.0, 0.0).volume, cv.tank_volume(R, h, 0.0), rtol=1e-9)


@pytest.fixture(scope="module")
def design():
    from design import closure
    return closure.close(SPEC)


def test_design_closes_cad_dry_mass_within_2pct(design):
    for stg in ("booster", "upper"):
        sizing_dry = getattr(design.sizing, stg).dry
        assert abs(design.dry[stg] - sizing_dry) / sizing_dry < 0.02


def test_all_structural_margins_positive_at_fs(design):
    # every ratio >= 1 means capacity >= FS (1.4) x limit load, i.e. >= 1.25 required by PLAN M1
    for name, m in design.margins.items():
        assert m["min_ratio"] >= 1.0 - 1e-9, (name, m)


def test_thrust_to_weight(design):
    chk = sizing.engine_check(SPEC, design.sizing)
    assert chk["liftoff_ok"] and chk["upper_ok"]


def test_tanks_hold_their_propellant(design):
    # liquid solids are sized from the sizing propellant; their CAD mass must equal it
    liq = sum(p.props()[0] for p in design.parts if p.kind == "propellant" and p.stage == "booster")
    assert np.isclose(liq, design.sizing.booster.prop, rtol=1e-9)
