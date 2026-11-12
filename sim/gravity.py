"""Gravity models in an Earth-centred frame: point mass and J2 (WGS-84 constants)."""
import numpy as np

MU = 3.986004418e14   # m^3/s^2
R_EQ = 6378137.0      # m
J2 = 1.08262668e-3


def point_mass(r):
    rn = np.linalg.norm(r)
    return -MU * r / rn ** 3


def j2(r):
    x, y, z = r
    rn = np.linalg.norm(r)
    k = 1.5 * J2 * MU * R_EQ ** 2 / rn ** 5
    zr2 = 5.0 * z * z / (rn * rn)
    return point_mass(r) + k * np.array([x * (zr2 - 1), y * (zr2 - 1), z * (zr2 - 3)])


def uniform(g=9.80665):
    """Flat-Earth gravity along -z (local ENU-up), for analytic tests and short hops."""
    vec = np.array([0.0, 0.0, -g])
    return lambda r: vec
