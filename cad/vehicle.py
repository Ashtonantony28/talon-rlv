"""Parametric build123d model of Talon-1 and mass properties straight from the CAD solids.

Vehicle axis is +Z (body x in ADR 0001 terms), z = 0 at the booster skirt base.
Every part carries a density, a stage tag and, for barrels, the structural station data used by
design/structures.py. Engines are mass-calibrated solids (their density is set so the solid
matches the spec mass); everything else gets its physical material density.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from build123d import (Box, Cone, Cylinder, Pos, Rot, Compound, export_step, export_stl, Align, BuildPart,
                       BuildSketch, Plane, Ellipse, Rectangle, Mode, revolve, Axis)
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

CENTER_BOTTOM = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Part:
    name: str
    solid: object
    rho: float
    stage: str                 # "booster" | "upper" | "payload"
    kind: str                  # "structure" | "engine" | "leg" | "fin" | "fairing" | "payload" | "propellant"
    station: dict = field(default_factory=dict)   # barrels: {"R", "t", "z_bottom", "pressure", "tank"}
    mass_override: float | None = None             # engines: calibrated mass

    def props(self):
        """(mass, centre of mass (3,), inertia about own CoM (3x3)); cached per part."""
        if getattr(self, "_props", None) is None:
            self._props = self._compute_props()
        return self._props

    def _compute_props(self):
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(self.solid.wrapped, g)
        vol = g.Mass()
        rho = self.rho if self.mass_override is None else self.mass_override / vol
        c = g.CentreOfMass()
        mi = g.MatrixOfInertia()
        J = np.array([[mi.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]) * rho
        return vol * rho, np.array([c.X(), c.Y(), c.Z()]), J


def combine(props):
    """Sum mass properties with the parallel-axis theorem."""
    m = sum(p[0] for p in props)
    c = sum(p[0] * p[1] for p in props) / m
    J = np.zeros((3, 3))
    for mi, ci, Ji in props:
        d = ci - c
        J += Ji + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return m, c, J


# ---------- primitive builders ----------

def tube(R, t, L, z0):
    return Pos(0, 0, z0) * (Cylinder(R, L, align=CENTER_BOTTOM) - Cylinder(R - t, L, align=CENTER_BOTTOM))


def half_ellipsoid(R, h, up, z0):
    """Solid half-ellipsoid of radius R and height h, base at z0, bulging up (+1) or down (-1).

    Built by revolving a quarter ellipse, so the surface is exact (a non-uniformly scaled sphere
    becomes a B-spline approximation and loses ~0.2 % of its volume)."""
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            Ellipse(R, h)
            Rectangle(R, h, align=(Align.MIN, Align.MIN if up > 0 else Align.MAX), mode=Mode.INTERSECT)
        revolve(axis=Axis.Z)
    return Pos(0, 0, z0) * bp.part


def dome_shell(R, t, h, up, z0):
    return half_ellipsoid(R, h, up, z0) - half_ellipsoid(R - t, h - t, up, z0)


def tank_volume(R, h, L):
    """Interior volume of barrel length L with two ellipsoidal end domes."""
    return np.pi * R * R * L + 2 * (2.0 / 3.0) * np.pi * R * R * h


def barrel_length_for(V, R, h):
    return max((V - 2 * (2.0 / 3.0) * np.pi * R * R * h) / (np.pi * R * R), 0.0)


def liquid_solid(R_in, h_in, L, z_barrel_bottom):
    """Full-tank liquid volume as a solid: aft dome + barrel + forward dome interior."""
    s = half_ellipsoid(R_in, h_in, -1, z_barrel_bottom)
    if L > 0:
        s = s + Pos(0, 0, z_barrel_bottom) * Cylinder(R_in, L, align=CENTER_BOTTOM)
    return s + half_ellipsoid(R_in, h_in, +1, z_barrel_bottom + L)


@lru_cache(maxsize=4)
def grid_fin(span, depth, chord, web, cells):
    """Lattice grid fin centred on the origin: outer frame with cells x cells square openings."""
    cell_w = (span - web) / cells
    cell_h = (chord - web) / cells
    holes = [Pos(-span / 2 + web / 2 + cell_w * (ix + 0.5), 0, -chord / 2 + web / 2 + cell_h * (iz + 0.5))
             * Box(cell_w - web, depth + 0.01, cell_h - web) for ix in range(cells) for iz in range(cells)]
    return Box(span, depth, chord) - Compound(children=holes)


# ---------- the vehicle ----------

def propellant_split(spec, prop_mass):
    mr = spec["propellants"]["mixture_ratio"]
    return prop_mass * mr / (1 + mr), prop_mass / (1 + mr)


def build(spec, sizing, thickness):
    """Build all parts. `thickness` maps barrel name -> wall thickness (m); missing -> min gauge."""
    lay, pr, mat, eng = spec["layout"], spec["propellants"], spec["material"], spec["engines"]
    R = spec["diameter_m"] / 2
    a = lay["dome_aspect"]
    tmin = spec["structures"]["min_gauge_m"]
    p_tank = spec["structures"]["tank_pressure_pa"]
    rho_al = mat["rho"]
    parts = []

    def t_of(name):
        return thickness.get(name, tmin)

    def add_barrel(name, stage, L, z0, tank=None):
        t = t_of(name)
        parts.append(Part(name, tube(R, t, L, z0), rho_al, stage, "structure",
                          {"R": R, "t": t, "z_bottom": z0, "L": L,
                           "pressure": p_tank if tank else 0.0, "tank": tank}))
        return z0 + L

    def add_tank(name, stage, prop_kg, rho_liq, z0):
        """Tank barrel starting at z0 plus its domes and its full liquid. Returns barrel top z."""
        V = prop_kg / rho_liq * (1 + pr["ullage_fraction"])
        h = a * R
        L = barrel_length_for(V, R, h)
        z1 = add_barrel(name, stage, L, z0, tank=name)
        t = t_of(name)
        parts.append(Part(name + "_aft_dome", dome_shell(R, t, h, -1, z0), rho_al, stage, "structure"))
        parts.append(Part(name + "_fwd_dome", dome_shell(R, t, h, +1, z1), rho_al, stage, "structure"))
        liq = liquid_solid(R - t, h - t, L, z0)
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(liq.wrapped, g)
        parts.append(Part(name + "_liquid", liq, prop_kg / g.Mass(), stage, "propellant",
                          {"tank": name, "z_bottom": z0}))
        return z1

    h = a * R
    ox_b, fu_b = propellant_split(spec, sizing.booster.prop)
    ox_u, fu_u = propellant_split(spec, sizing.upper.prop)

    # --- booster ---
    eb = eng["booster"]
    n = eb["count"]
    ring = 0.62 * R
    for i in range(n):
        xy = (0.0, 0.0) if i == 0 else (ring * np.cos(2 * np.pi * (i - 1) / (n - 1)),
                                        ring * np.sin(2 * np.pi * (i - 1) / (n - 1)))
        nozzle = Pos(xy[0], xy[1], 0.05) * Cone(eb["exit_diameter_m"] / 2, 0.12, eb["length_m"], align=CENTER_BOTTOM)
        parts.append(Part(f"b_engine{i}", nozzle, 1.0, "booster", "engine", mass_override=eb["mass_kg"]))
    z = add_barrel("b_skirt", "booster", lay["booster_skirt_m"], 0.0)
    z = add_tank("b_fuel", "booster", fu_b, pr["rho_fuel"], z)
    z = add_barrel("b_intertank", "booster", 2 * h + lay["bay_gap_m"], z)
    z = add_tank("b_ox", "booster", ox_b, pr["rho_ox"], z)
    eu = eng["upper"]
    z_sep = add_barrel("b_interstage", "booster", h + eu["length_m"] + lay["interstage_extra_m"], z)

    gf = lay["grid_fins"]
    frame = grid_fin(gf["span_m"], gf["depth_m"], gf["chord_m"], gf["web_m"], gf["cells"])
    for i in range(gf["count"]):
        ang = 360.0 * i / gf["count"]
        fin = Rot(0, 0, ang) * Pos(R + gf["span_m"] / 2, 0, z_sep - gf["chord_m"] / 2 - 0.1) * frame
        parts.append(Part(f"b_gridfin{i}", fin, gf["rho"], "booster", "fin"))

    lg = lay["legs"]
    for i in range(lg["count"]):
        ang = 360.0 * i / lg["count"] + 45.0
        strut = tube(lg["tube_d_m"] / 2, lg["tube_t_m"], lg["length_m"], 0.0)
        foot = Cylinder(lg["foot_d_m"] / 2, 0.05, align=CENTER_BOTTOM)
        leg = Rot(0, 0, ang) * Pos(R + lg["tube_d_m"] / 2, 0, 0.0) * (strut + foot)
        parts.append(Part(f"b_leg{i}", leg, lg["rho"], "booster", "leg"))
        act = Rot(0, 0, ang) * Pos(R + lg["tube_d_m"] / 2, 0, lg["length_m"] - 0.3) * Box(0.08, 0.08, 0.3, align=CENTER_BOTTOM)
        parts.append(Part(f"b_leg{i}_actuator", act, 1.0, "booster", "leg", mass_override=lg["actuator_kg"]))

    # --- upper stage (aft skirt holds the aft dome; the engine hangs into the interstage) ---
    z = add_barrel("u_aft_skirt", "upper", h + 0.05, z_sep)
    upper_nozzle = Pos(0, 0, z - h - eu["length_m"]) * Cone(eu["exit_diameter_m"] / 2, 0.14, eu["length_m"], align=CENTER_BOTTOM)
    parts.append(Part("u_engine", upper_nozzle, 1.0, "upper", "engine", mass_override=eu["mass_kg"]))
    z = add_tank("u_fuel", "upper", fu_u, pr["rho_fuel"], z)
    z = add_barrel("u_intertank", "upper", 2 * h + lay["bay_gap_m"], z)
    z = add_tank("u_ox", "upper", ox_u, pr["rho_ox"], z)
    z = add_barrel("u_fwd_skirt", "upper", h + lay["upper_fwd_skirt_extra_m"], z)
    pa = Pos(0, 0, z) * (Cone(R, 0.45 * R, lay["payload_adapter_m"], align=CENTER_BOTTOM)
                         - Cone(R - tmin * 2, 0.45 * R - tmin * 2, lay["payload_adapter_m"], align=CENTER_BOTTOM))
    parts.append(Part("u_payload_adapter", pa, rho_al, "upper", "structure"))
    pe = lay["payload_envelope"]
    pay = Pos(0, 0, z + lay["payload_adapter_m"]) * Cylinder(pe["diameter_m"] / 2, pe["height_m"], align=CENTER_BOTTOM)
    parts.append(Part("payload", pay, 1.0, "payload", "payload", mass_override=spec["mission"]["payload_kg"]))
    fr = lay["fairing"]
    ft = fr["thickness_m"]
    fcyl = tube(R, ft, fr["cyl_m"], z)
    nose = Pos(0, 0, z + fr["cyl_m"]) * (Cone(R, 0.0, fr["nose_m"], align=CENTER_BOTTOM)
                                         - Cone(R - ft, 0.0, fr["nose_m"] - ft * fr["nose_m"] / R, align=CENTER_BOTTOM))
    parts.append(Part("u_fairing_cyl", fcyl, fr["rho"], "upper", "fairing"))
    parts.append(Part("u_fairing_nose", nose, fr["rho"], "upper", "fairing"))
    z_top = z + fr["cyl_m"] + fr["nose_m"]
    return parts, {"z_sep": z_sep, "z_top": z_top, "length": z_top}


def stage_dry_mass(spec, parts, stage):
    """Dry mass of a stage from CAD: structure + engines + legs + fins + fairing + lumped items."""
    lump = spec["lumped"]
    total = struct = 0.0
    for p in parts:
        if p.stage != stage or p.kind in ("propellant", "payload"):
            continue
        m = p.props()[0]
        total += m
        if p.kind == "structure":
            struct += m
    total += lump[f"{stage}_avionics_kg"] + lump[f"{stage}_misc_fraction"] * struct
    return total


def mass_properties(spec, parts, include=lambda p: True):
    """Combined mass properties of selected parts, with each stage's lumped masses smeared at its CoM."""
    lump = spec["lumped"]
    sel = [p for p in parts if include(p)]
    props = [p.props() for p in sel]
    for stage in ("booster", "upper"):
        sp = [pp for p, pp in zip(sel, props) if p.stage == stage and p.kind != "propellant"]
        if not sp:
            continue
        struct = sum(pp[0] for p, pp in zip(sel, props) if p.stage == stage and p.kind == "structure")
        m_l = lump[f"{stage}_avionics_kg"] + lump[f"{stage}_misc_fraction"] * struct
        _, c, _ = combine(sp)
        props.append((m_l, c, np.zeros((3, 3))))
    return combine(props)


def export(parts, out_dir):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solids = [p.solid for p in parts if p.kind != "propellant"]
    comp = Compound(children=solids)
    export_step(comp, str(out / "talon1.step"))
    export_stl(comp, str(out / "talon1.stl"), tolerance=0.01)
    return out
