"""M0 analytic regression suite for the 6-DOF core."""
import numpy as np
import pytest
from sim import quat, atmosphere, gravity
from sim.rigidbody import RigidBody, Loads, pack, integrate, R, V, Q, W, M

Z3 = np.zeros(3)
Q0 = np.array([1.0, 0, 0, 0])


def free_loads(t, x):
    return Loads(Z3.copy(), Z3.copy())


def test_quat_dcm_orthonormal_and_rotation():
    q = quat.from_axis_angle([1, 2, 3], 0.7)
    C = quat.to_dcm(q)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(C), 1.0)
    # 90 deg about z maps body x to inertial y
    qz = quat.from_axis_angle([0, 0, 1], np.pi / 2)
    assert np.allclose(quat.rotate(qz, [1, 0, 0]), [0, 1, 0], atol=1e-15)
    # composition matches matrix product
    q2 = quat.from_axis_angle([0, 1, 0], -0.3)
    assert np.allclose(quat.to_dcm(quat.mul(q, q2)), C @ quat.to_dcm(q2), atol=1e-14)


def test_vacuum_ballistic_exact():
    g = 9.80665
    body = RigidBody(inertia=lambda m: np.eye(3), loads=free_loads, gravity=gravity.uniform(g))
    v0 = np.array([30.0, 0.0, 100.0])
    ts, X = integrate(body, pack(Z3, v0, Q0, Z3, 10.0), 15.0, 0.05)
    t = ts[-1]
    expect = v0 * t + 0.5 * np.array([0, 0, -g]) * t * t
    assert np.allclose(X[-1, R], expect, rtol=1e-12, atol=1e-9)


def test_rocket_equation():
    isp, g0, thrust, m0, mdot = 300.0, 9.80665, 5000.0, 100.0, -5000.0 / (300.0 * 9.80665)
    def loads(t, x):
        return Loads(quat.rotate(x[Q], [1.0, 0, 0]) * thrust, Z3.copy(), mdot)
    body = RigidBody(inertia=lambda m: np.eye(3), loads=loads, gravity=lambda r: Z3)
    tb = 20.0
    ts, X = integrate(body, pack(Z3, Z3, Q0, Z3, m0), tb, 0.01)
    mf = m0 + mdot * tb
    assert np.isclose(X[-1, M], mf, rtol=1e-12)
    dv = isp * g0 * np.log(m0 / mf)
    assert np.isclose(X[-1, V][0], dv, rtol=1e-9)


def test_torque_free_conserves_energy_and_momentum():
    J = np.diag([2.0, 3.0, 5.0])
    body = RigidBody(inertia=lambda m: J, loads=free_loads, gravity=lambda r: Z3)
    w0 = np.array([0.3, 1.5, 0.2])  # near unstable intermediate axis: a hard case
    ts, X = integrate(body, pack(Z3, Z3, Q0, w0, 1.0), 20.0, 1e-3)
    def inv(x):
        w = x[W]
        H_inertial = quat.rotate(x[Q], J @ w)
        return 0.5 * w @ J @ w, H_inertial
    E0, H0 = inv(X[0])
    E1, H1 = inv(X[-1])
    assert abs(E1 - E0) / E0 < 1e-9
    assert np.linalg.norm(H1 - H0) / np.linalg.norm(H0) < 1e-9


def test_axisymmetric_precession_rate():
    It, Iz, wz = 4.0, 1.0, 10.0
    J = np.diag([It, It, Iz])
    body = RigidBody(inertia=lambda m: J, loads=free_loads, gravity=lambda r: Z3)
    ts, X = integrate(body, pack(Z3, Z3, Q0, [0.5, 0.0, wz], 1.0), 3.0, 1e-3)
    lam = (Iz - It) / It * wz  # body-frame precession of transverse rate
    t = ts[-1]
    expect = 0.5 * np.array([np.cos(lam * t), np.sin(lam * t)])
    assert np.allclose(X[-1, W][:2], expect, atol=1e-9)


def test_circular_orbit_radius_held():
    r0 = gravity.R_EQ + 400e3
    vc = np.sqrt(gravity.MU / r0)
    period = 2 * np.pi * np.sqrt(r0 ** 3 / gravity.MU)
    body = RigidBody(inertia=lambda m: np.eye(3), loads=free_loads, gravity=gravity.point_mass)
    ts, X = integrate(body, pack([r0, 0, 0], [0, vc, 0], Q0, Z3, 1.0), period, period / 5000)
    radii = np.linalg.norm(X[:, R], axis=1)
    assert np.max(np.abs(radii - r0)) / r0 < 1e-9
    assert np.linalg.norm(X[-1, R] - [r0, 0, 0]) < 1.0  # back where it started, < 1 m


def test_j2_reduces_to_point_mass_scale():
    r = np.array([gravity.R_EQ, 0, 0])
    rel = np.linalg.norm(gravity.j2(r) - gravity.point_mass(r)) / np.linalg.norm(gravity.point_mass(r))
    assert np.isclose(rel, 1.5 * gravity.J2, rtol=1e-12)  # equatorial J2 correction


@pytest.mark.parametrize("h_geopot,T,p", [
    (0.0, 288.15, 101325.0),
    (11000.0, 216.65, 22632.06),
    (20000.0, 216.65, 5474.889),
    (32000.0, 228.65, 868.0187),
    (47000.0, 270.65, 110.9063),
])
def test_us76_reference_points(h_geopot, T, p):
    z = atmosphere.R_EARTH_GEOPOT * h_geopot / (atmosphere.R_EARTH_GEOPOT - h_geopot)
    Tm, pm, rho, a = atmosphere.properties(z)
    assert np.isclose(Tm, T, rtol=1e-9)
    assert np.isclose(pm, p, rtol=2e-6)
    if h_geopot == 0.0:
        assert np.isclose(rho, 1.225, rtol=1e-4)
        assert np.isclose(a, 340.294, rtol=1e-5)
