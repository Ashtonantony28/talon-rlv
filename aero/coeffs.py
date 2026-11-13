"""Ascent aerodynamics of the Talon stack.

Axial force coefficient CA(M) is a semi-empirical table for a slender cone-nosed body (spec),
linearly interpolated, plus base drag when the engines are off. Normal force follows slender-body
theory: CN_alpha = 2 per radian for any pointed nose, acting at the nose centre of pressure
(2/3 of a cone's length from the tip). Coefficients refer to the body cross-section.
"""
import numpy as np


class Aero:
    def __init__(self, spec):
        a = spec["aero"]
        self.mach = np.asarray(a["mach"], float)
        self.ca = np.asarray(a["ca_on"], float)
        self.ca_base_off = a["ca_base_off"]
        self.cn_alpha = a["cn_alpha_nose_per_rad"]
        self.S = np.pi * (spec["diameter_m"] / 2) ** 2
        lay = spec["layout"]
        self.x_cp_from_tip = lay["nose_cp_fraction"] * lay["fairing"]["nose_m"]

    def CA(self, mach, engines_on=True):
        c = np.interp(mach, self.mach, self.ca)
        return c if engines_on else c + self.ca_base_off

    def CN(self, alpha):
        return self.cn_alpha * np.sin(alpha) * np.cos(alpha)   # slender-body, valid to large alpha

    def forces(self, v_rel_body, rho, a_sound, engines_on=True):
        """Aerodynamic force in body axes (x along the nose) from air-relative velocity in body axes."""
        V = np.linalg.norm(v_rel_body)
        if V < 1e-6:
            return np.zeros(3), 0.0
        q = 0.5 * rho * V * V
        mach = V / a_sound
        u = v_rel_body / V
        alpha = np.arccos(np.clip(u[0], -1.0, 1.0))
        axial = -q * self.S * self.CA(mach, engines_on) * np.sign(u[0] if u[0] != 0 else 1.0)
        lat = u[1:]
        ln = np.linalg.norm(lat)
        normal = np.zeros(2) if ln < 1e-12 else -q * self.S * self.CN(alpha) * lat / ln
        return np.array([axial, normal[0], normal[1]]), q
