"""Variable-mass 6-DOF rigid body.

State x = [r(3), v(3), q(4), w(3), m(1)]: inertial position/velocity, body-to-inertial quaternion
(scalar-first), body angular rate, mass. Forces are given in the inertial frame, torques in the body
frame. Inertia is supplied as a function of mass so propellant depletion shifts it (ADR 0001).
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from . import quat

R, V, Q, W, M = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), 13
NX = 14


def pack(r, v, q, w, m):
    return np.concatenate([r, v, q, w, [m]]).astype(float)


@dataclass
class Loads:
    force_inertial: np.ndarray
    torque_body: np.ndarray
    mdot: float = 0.0  # kg/s, negative while burning


@dataclass
class RigidBody:
    inertia: Callable[[float], np.ndarray]   # m -> 3x3 body inertia about CoM
    loads: Callable[[float, np.ndarray], Loads]
    gravity: Callable[[np.ndarray], np.ndarray]
    inertia_dot: Callable[[float, float], np.ndarray] | None = None  # (m, mdot) -> dJ/dt

    def deriv(self, t, x):
        r, v, q, w, m = x[R], x[V], x[Q], x[W], x[M]
        ld = self.loads(t, x)
        J = self.inertia(m)
        Jdot = self.inertia_dot(m, ld.mdot) if self.inertia_dot else np.zeros((3, 3))
        dx = np.empty(NX)
        dx[R] = v
        dx[V] = ld.force_inertial / m + self.gravity(r)
        dx[Q] = quat.deriv(q, w)
        dx[W] = np.linalg.solve(J, ld.torque_body - np.cross(w, J @ w) - Jdot @ w)
        dx[M] = ld.mdot
        return dx


def rk4_step(f, t, x, dt):
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[Q] = quat.normalize(xn[Q])
    return xn


def integrate(body, x0, t_end, dt, stop=None):
    """Fixed-step RK4. Returns (t array, X array). `stop(t, x)` ends the run early when True."""
    n = int(round(t_end / dt))
    ts = np.empty(n + 1)
    X = np.empty((n + 1, NX))
    ts[0], X[0] = 0.0, x0
    x = x0.copy()
    for i in range(n):
        x = rk4_step(body.deriv, ts[i], x, dt)
        ts[i + 1], X[i + 1] = ts[i] + dt, x
        if stop is not None and stop(ts[i + 1], x):
            return ts[: i + 2], X[: i + 2]
    return ts, X
