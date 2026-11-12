"""Quaternion utilities. Scalar-first [w, x, y, z], Hamilton product, body-to-inertial (ADR 0001)."""
import numpy as np


def mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q):
    return q / np.linalg.norm(q)


def to_dcm(q):
    """Rotation matrix C such that v_inertial = C @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), *(axis * s)])


def rotate(q, v):
    return to_dcm(q) @ v


def deriv(q, omega_body):
    """dq/dt for body angular rate omega_body (rad/s)."""
    return 0.5 * mul(q, np.array([0.0, *omega_body]))
