"""US Standard Atmosphere 1976, 0-86 km geopotential layers."""
import numpy as np

G0 = 9.80665          # m/s^2
R_AIR = 8314.32 / 28.9644  # J/(kg K), R*/M0 exactly as defined in US76
GAMMA = 1.4
R_EARTH_GEOPOT = 6356766.0  # m, used for geometric -> geopotential altitude

# base geopotential altitude (m), base temperature (K), lapse rate (K/m), base pressure (Pa)
_LAYERS = []
_h = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
_L = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]
_T, _P = 288.15, 101325.0
for i, L in enumerate(_L):
    _LAYERS.append((_h[i], _T, L, _P))
    dh = _h[i + 1] - _h[i]
    if L == 0.0:
        _P = _P * np.exp(-G0 * dh / (R_AIR * _T))
    else:
        T1 = _T + L * dh
        _P = _P * (T1 / _T) ** (-G0 / (R_AIR * L))
        _T = T1
H_MAX = _h[-1]


def properties(alt_geometric):
    """Return (T [K], p [Pa], rho [kg/m^3], a [m/s]) at geometric altitude (m).

    Above 84.852 km geopotential, density decays exponentially from the last layer (adequate
    for drag; documented approximation)."""
    z = max(float(alt_geometric), 0.0)
    h = R_EARTH_GEOPOT * z / (R_EARTH_GEOPOT + z)
    if h >= H_MAX:
        hb, Tb, L, pb = _LAYERS[-1]
        T = Tb + L * (H_MAX - hb)
        p_top = pb * (T / Tb) ** (-G0 / (R_AIR * L))
        p = p_top * np.exp(-G0 * (h - H_MAX) / (R_AIR * T))
    else:
        for hb, Tb, L, pb in reversed(_LAYERS):
            if h >= hb:
                break
        dh = h - hb
        if L == 0.0:
            T = Tb
            p = pb * np.exp(-G0 * dh / (R_AIR * Tb))
        else:
            T = Tb + L * dh
            p = pb * (T / Tb) ** (-G0 / (R_AIR * L))
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA * R_AIR * T)
    return T, p, rho, a
