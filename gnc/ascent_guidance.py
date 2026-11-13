"""Open-loop-parameterised ascent guidance (M2).

Booster: vertical rise, pitch kick toward the launch azimuth, attitude hold until the air-relative
velocity has tilted by the kick angle, then a zero-alpha gravity turn (thrust along air-relative
velocity, which keeps aerodynamic normal loads near zero through max-q).
Upper stage: linear-tangent steering, tan(theta) = tan(theta0) - c * t, theta measured from the
local horizontal in the orbit plane. This is the optimal steering law for flat-Earth vacuum flight
and close to optimal for a short upper-stage burn.
"""
from dataclasses import dataclass
import numpy as np


def local_frame(r, azimuth):
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    downrange = np.sin(azimuth) * east + np.cos(azimuth) * north
    return up, downrange


@dataclass
class AscentParams:
    kick_deg: float
    theta0_deg: float      # upper stage initial pitch above local horizontal
    c_per_s: float         # linear-tangent rate


class BoosterGuidance:
    def __init__(self, cfg, p: AscentParams, azimuth):
        self.t_v = cfg["vertical_rise_s"]
        self.t_k = cfg["kick_duration_s"]
        self.kick = np.radians(p.kick_deg)
        self.az = azimuth
        self.gravity_turn = False

    def update(self, t, r, v_rel):
        """Mode logic, called once per step: switch to the gravity turn after the kick once the
        air-relative velocity has tilted by the kick angle."""
        if self.gravity_turn or t < self.t_v + self.t_k:
            return
        up = r / np.linalg.norm(r)
        vr = np.linalg.norm(v_rel)
        tilt = np.arccos(np.clip(np.dot(v_rel, up) / vr, -1, 1)) if vr > 1 else 0.0
        if tilt >= self.kick:
            self.gravity_turn = True

    def direction(self, t, r, v_rel):
        """Thrust direction (pure function of time and state, given the current mode)."""
        if self.gravity_turn:
            return v_rel / np.linalg.norm(v_rel)
        up, dr = local_frame(r, self.az)
        if t < self.t_v:
            return up
        ang = self.kick * min((t - self.t_v) / self.t_k, 1.0)
        return np.cos(ang) * up + np.sin(ang) * dr


class UpperGuidance:
    def __init__(self, p: AscentParams):
        self.tan0 = np.tan(np.radians(p.theta0_deg))
        self.c = p.c_per_s
        self.t0 = None

    def start(self, t):
        if self.t0 is None:
            self.t0 = t

    def direction(self, t, r, v):
        up = r / np.linalg.norm(r)
        h = np.cross(r, v)
        horiz = np.cross(h, r)
        horiz /= np.linalg.norm(horiz)
        th = np.arctan(self.tan0 - self.c * (t - self.t0))
        return np.cos(th) * horiz + np.sin(th) * up
