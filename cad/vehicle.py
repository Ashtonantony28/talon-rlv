"""Parametric build123d model of Talon-1 and mass properties straight from the CAD solids.

Vehicle axis is +Z (body x in ADR 0001 terms), z = 0 at the booster skirt base. Every part
carries a density, a stage tag and, for barrels, the structural station data used by
design/structures.py. Engines are mass-calibrated solids (density set so the solid matches the
spec mass); everything else gets its physical material density.

Configurations: ``state="ascent"`` (legs and grid fins stowed) or ``state="landing"`` (deployed).
``sweeps()`` returns the swept envelopes of every moving part at full travel for cad/audit.py.

Geometry rules that keep the model interference-free (checked by cad/audit.py):
  * tank domes sit inside the barrel/bay walls: dome outer radius = R - max(adjacent wall t),
    the sub-millimetre land between dome equator and the thinner wall is the weld land;
  * the LOX downcomer passes through a tunnel in the fuel tank (cut from both fuel domes and the
    fuel liquid), so it never shares volume with propellant or structure;
  * bolted joints meet on shared faces (zero-volume contact), never by overlap.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from build123d import (Box, Cone, Cylinder, Pos, Rot, Compound, export_step, export_stl, Align, BuildPart,
                       BuildSketch, BuildLine, Plane, Ellipse, Rectangle, Mode, revolve, Axis, Spline, Polyline,
                       Line, make_face, Solid, Vector, Sphere, ThreePointArc)
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

CB = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Part:
    name: str
    solid: object
    rho: float
    stage: str                 # "booster" | "upper" | "payload"
    kind: str                  # structure | secondary | engine | leg | fin | fairing | payload | propellant
    station: dict = field(default_factory=dict)   # barrels: {"R", "t", "z_bottom", "pressure", "tank"}
    mass_override: float | None = None
    moving: str | None = None                     # "gimbal" | "leg" | "fin" for swept checks

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


# ---------------------------------------------------------------- primitives

def tube(R, t, L, z0):
    return Pos(0, 0, z0) * (Cylinder(R, L, align=CB) - Cylinder(R - t, L, align=CB))


def half_ellipsoid(R, h, up, z0):
    """Solid half-ellipsoid (radius R, height h), base at z0, bulging up (+1) or down (-1).

    Revolved from a quarter ellipse so the surface is exact (a scaled sphere becomes a B-spline
    approximation and loses ~0.2 % of its volume)."""
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            Ellipse(R, h)
            Rectangle(R, h, align=(Align.MIN, Align.MIN if up > 0 else Align.MAX), mode=Mode.INTERSECT)
        revolve(axis=Axis.Z)
    return Pos(0, 0, z0) * bp.part


def dome_shell(R, t, h, up, z0):
    return half_ellipsoid(R, h, up, z0) - half_ellipsoid(R - t, h - t, up, z0)


def tank_volume(R, h, L):
    return np.pi * R * R * L + 2 * (2.0 / 3.0) * np.pi * R * R * h


def barrel_length_for(V, R, h):
    return max((V - 2 * (2.0 / 3.0) * np.pi * R * R * h) / (np.pi * R * R), 0.0)


def liquid_solid(R_in, h_in, L, z_barrel_bottom):
    """Full-tank liquid: aft dome + barrel + forward dome interior (barrel radius R_in)."""
    s = half_ellipsoid(R_in, h_in, -1, z_barrel_bottom)
    if L > 0:
        s = s + Pos(0, 0, z_barrel_bottom) * Cylinder(R_in, L, align=CB)
    return s + half_ellipsoid(R_in, h_in, +1, z_barrel_bottom + L)


def revolve_rz(points, smooth=True):
    """Solid of revolution about Z of the region between the axis and a curve through (r, z)
    points ordered by z. smooth=True uses a spline (no facets)."""
    pts = [(float(r), float(z)) for r, z in points]
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                if smooth:
                    Spline(*pts)
                else:
                    Polyline(*pts)
                Polyline(pts[-1], (0.0, pts[-1][1]), (0.0, pts[0][1]), pts[0])
            make_face()
        revolve(axis=Axis.Z)
    return bp.part


def revolved_shell(points, t, smooth=True):
    """Thin shell of revolution: outer curve minus the curve moved radially inward by t."""
    inner = [(max(r - t, 1e-4), z) for r, z in points]
    return revolve_rz(points, smooth) - revolve_rz(inner, smooth)


def annular_sector(r_in, r_out, z0, height, width, azimuth_deg):
    """Body-conformal block: annulus r_in..r_out, tangential chord width, at an azimuth."""
    ring = Pos(0, 0, z0) * (Cylinder(r_out, height, align=CB) - Cylinder(r_in, height, align=CB))
    wedge = Pos(r_out / 2 + 0.01, 0, z0 + height / 2) * Box(r_out + 0.02, width, height)
    return Rot(0, 0, azimuth_deg) * (ring & wedge)


def _ogive_geometry(R, L, rn):
    """Spherically blunted tangent ogive (x from the sharp apex toward the base at x = L)."""
    rho = (R * R + L * L) / (2 * R)
    x0 = L - np.sqrt((rho - rn) ** 2 - (rho - R) ** 2)
    yt = rn * (rho - R) / (rho - rn)
    xt = x0 - np.sqrt(rn * rn - yt * yt)
    xa = x0 - rn
    return rho, x0, xt, yt, xa


def blunted_ogive_length(R, L, rn):
    _, _, _, _, xa = _ogive_geometry(R, L, rn)
    return L - xa


def blunted_ogive(R, L, rn, z_base):
    """Solid nose: tangent ogive from the shoulder (radius R at z_base) to the tangency point,
    then an exact spherical cap of radius rn to the apex."""
    rho, x0, xt, yt, xa = _ogive_geometry(R, L, rn)
    xs = xt + (L - xt) * (1 - np.linspace(0, 1, 16) ** 1.5)          # base -> tangency
    og = [(float(np.sqrt(rho ** 2 - (L - x) ** 2) + R - rho), float(z_base + L - x)) for x in xs]
    og[-1] = (float(yt), float(z_base + L - xt))
    xm = 0.5 * (xa + xt)                                            # arc midpoint
    pm = (float(np.sqrt(rn * rn - (xm - x0) ** 2)), float(z_base + L - xm))
    apex = (0.0, float(z_base + L - xa))
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Spline(*og)
                ThreePointArc(og[-1], pm, apex)
                Polyline(apex, (0.0, float(z_base)), og[0])
            make_face()
        revolve(axis=Axis.Z)
    return bp.part


def hollow_sector(r_in, r_out, z0, height, width, azimuth_deg, wall):
    """Closed fairing/housing shell conforming to the body: a body-conformal block with a pocket
    `wall` thick on the outer, side and end faces (open toward the skin it is bonded to)."""
    outer = annular_sector(r_in, r_out, z0, height, width, azimuth_deg)
    pocket = annular_sector(r_in, r_out - wall, z0 + wall, height - 2 * wall, width - 2 * wall, azimuth_deg)
    return outer - pocket


def leg_fitting(R, lg, azimuth_deg):
    """Leg hinge fitting: skin doubler, two clevis lugs outboard of the side struts and the hinge pin."""
    br = lg["bracket"]
    H = (R + lg["hinge_offset_m"], 0.0, lg["hinge_z_m"])
    y_lug = lg["side_spread_m"] + lg["side_d_m"] / 2 + 0.03
    dbl = annular_sector(R, R + 0.006, lg["hinge_z_m"] - br["height_m"] / 2, br["height_m"], br["width_m"], 0.0)
    f = dbl
    for sgn in (1, -1):
        lug = Pos(R + 0.006 + (H[0] + 0.04 - R - 0.006) / 2, sgn * y_lug, lg["hinge_z_m"] - 0.05) * Box(
            H[0] + 0.04 - R - 0.006, 0.02, 0.20)
        f = f + lug
    pin = strut((H[0], -y_lug - 0.012, H[2]), (H[0], y_lug + 0.012, H[2]), 0.02)
    return Rot(0, 0, azimuth_deg) * (f + pin)


def strut(p0, p1, r_out, t=None):
    """Tube (or rod when t is None) between two points."""
    p0, p1 = Vector(*p0), Vector(*p1)
    d = p1 - p0
    L = d.length
    pl = Plane(origin=p0, z_dir=d.normalized())
    s = Solid.make_cylinder(r_out, L, pl)
    if t is not None:
        s = s - Solid.make_cylinder(r_out - t, L, pl)
    return s


def frame_ring(R_inner_wall, z, spec):
    fr = spec["layout"]["frame"]
    return Pos(0, 0, z) * (Cylinder(R_inner_wall, fr["height_m"], align=CB)
                           - Cylinder(R_inner_wall - fr["depth_m"], fr["height_m"], align=CB))


# ---------------------------------------------------------------- sub-assemblies

def engine_profile(length, exit_r, kind="booster"):
    """(r, z) outer contour from injector face (z = length) to nozzle exit (z = 0): chamber,
    converging section, throat and a Rao-like bell, as one smooth spline."""
    Lc = 0.28 * length
    rc = 0.34 * exit_r if kind == "booster" else 0.22 * exit_r
    rt = 0.55 * rc
    z_t = length - Lc - 0.10 * length
    pts = [(rc, length), (rc, length - 0.6 * Lc), (0.92 * rc, length - Lc),
           (rt * 1.15, z_t + 0.04 * length), (rt, z_t)]
    # bell: theta_n ~ 30 deg at throat exit, theta_e ~ 8 deg at exit (quadratic Bezier fit)
    for f in (0.15, 0.35, 0.55, 0.75, 1.0):
        zb = z_t * (1 - f)
        rb = rt + (exit_r - rt) * (1 - (1 - f) ** 1.8)
        pts.append((rb, zb))
    return list(reversed(pts))


@lru_cache(maxsize=4)
def engine_solid(length, exit_r, kind):
    """Engine at the origin, nozzle exit at z = 0: bell shell + chamber + powerhead + gimbal block."""
    prof = engine_profile(length, exit_r, kind)
    wall = 0.004 if kind == "booster" else 0.003
    bell = revolved_shell(prof, wall)
    rc = prof[-1][0]
    head = Pos(0, 0, length) * Cylinder(rc * 0.9, 0.06, align=CB)
    pump = Pos(0, 0, length + 0.06) * Cone(rc * 0.9, rc * 0.55, 0.08, align=CB)
    return bell + head + pump, length + 0.14


@lru_cache(maxsize=4)
def grid_fin_solid(span, width, depth, web, nx, ny):
    """Lattice fin in its deployed pose at the origin: span along +x from the hinge line, width
    along y, depth along z (flow passes through the cells along the vehicle axis)."""
    cw = (span - web) / nx
    ch = (width - web) / ny
    holes = [Pos(web / 2 + cw * (i + 0.5), -width / 2 + web / 2 + ch * (j + 0.5), 0) * Box(cw - web, ch - web, depth + 0.01)
             for i in range(nx) for j in range(ny)]
    plate = Pos(span / 2, 0, 0) * Box(span, width, depth)
    root = Pos(-0.02, 0, 0) * Box(0.04, width * 0.5, depth)          # hinge lug to the hinge line
    return plate - Compound(children=holes) + root


def fin_envelope(span, width, depth):
    return Pos(span / 2, 0, 0) * Box(span, width, depth) + Pos(-0.02, 0, 0) * Box(0.04, width * 0.5, depth)


def leg_parts(lg, R, deploy_deg, envelope=False):
    """One A-frame leg at azimuth 0 (radial = +x), built deployed then rotated about the hinge
    axis (tangential, through H) by deploy_deg - full_deploy. Returns list of (name, solid, rho)."""
    H = np.array([R + lg["hinge_offset_m"], 0.0, lg["hinge_z_m"]])
    full = np.radians(lg["deploy_deg"])
    u = np.array([np.sin(full), 0.0, np.cos(full)])             # deployed direction (down and out)
    L = lg["length_m"]
    foot = H + L * u
    rm, rs = lg["main_d_m"] / 2, lg["side_d_m"] / 2
    s0 = 0.06                                                   # strut ends stop short of the hinge lugs
    if envelope:
        main = strut(H + s0 * u, foot, rm)
    else:
        main = strut(H + s0 * u, foot, rm, lg["main_t_m"])
    parts = [("main", main, lg["rho"])]
    for sgn, nm in ((1, "side_l"), (-1, "side_r")):
        Hs = H + np.array([0.0, sgn * lg["side_spread_m"], 0.0])
        tip = foot - 0.25 * u
        d = (tip - Hs) / np.linalg.norm(tip - Hs)
        parts.append((nm, strut(Hs + s0 * d, tip, rs, None if envelope else lg["side_t_m"]), lg["rho"]))
    pad = Pos(*(foot + np.array([0, 0, -rm - lg["pad_h_m"]]))) * (
        Cylinder(lg["pad_d_m"] / 2, lg["pad_h_m"], align=CB)
        + Pos(0, 0, lg["pad_h_m"]) * Cone(lg["pad_d_m"] / 2 * 0.6, rm * 0.9, rm, align=CB))
    parts.append(("pad", pad, lg["pad_rho"]))
    ang = deploy_deg - lg["deploy_deg"]
    if abs(ang) > 1e-9:
        ax = Axis(tuple(H), (0, 1, 0))
        parts = [(n, s.rotate(ax, ang), r) for n, s, r in parts]
    return parts


def fin_pose(gf, R, z_hinge, deploy_deg, solid):
    """Place a deployed-pose fin at its hinge (azimuth 0) folded by (deploy_deg - full)."""
    H = (R + gf["hinge_offset_m"], 0.0, z_hinge)
    s = Pos(*H) * solid
    ang = deploy_deg - gf["deploy_deg"]
    if abs(ang) > 1e-9:
        s = s.rotate(Axis(H, (0, 1, 0)), ang)
    return s


# ---------------------------------------------------------------- the vehicle

def propellant_split(spec, prop_mass):
    mr = spec["propellants"]["mixture_ratio"]
    return prop_mass * mr / (1 + mr), prop_mass / (1 + mr)


def build(spec, sizing, thickness, state="ascent"):
    """Build all parts. `thickness` maps barrel name -> wall thickness (m); missing -> min gauge."""
    assert state in ("ascent", "landing")
    lay, pr, mat, eng = spec["layout"], spec["propellants"], spec["material"], spec["engines"]
    R = spec["diameter_m"] / 2
    a = lay["dome_aspect"]
    tmin = spec["structures"]["min_gauge_m"]
    p_tank = spec["structures"]["tank_pressure_pa"]
    rho_al = mat["rho"]
    h = a * R
    parts = []
    info = {}

    def t_of(name):
        return thickness.get(name, tmin)

    def add(name, solid, rho, stage, kind, **kw):
        parts.append(Part(name, solid, rho, stage, kind, **kw))

    def add_barrel(name, stage, L, z0, tank=None):
        t = t_of(name)
        add(name, tube(R, t, L, z0), rho_al, stage, "structure",
            station={"R": R, "t": t, "z_bottom": z0, "L": L, "pressure": p_tank if tank else 0.0, "tank": tank})
        return z0 + L

    # tank barrels are laid out first so every dome knows both neighbouring wall thicknesses
    ox_b, fu_b = propellant_split(spec, sizing.booster.prop)
    ox_u, fu_u = propellant_split(spec, sizing.upper.prop)

    def add_tank(name, stage, prop_kg, rho_liq, z0, below, above, tunnel=None, tunnel_through=False):
        """Tank barrel from z0 with domes inset inside the walls below/above; liquid follows the
        dome/barrel interior. tunnel: radius of a central tunnel cut through (fuel tank)."""
        t = t_of(name)
        Rd = R - max(t, t_of(below), t_of(above))        # dome outer radius (weld land <= 1 mm)
        td = t
        # liquid barrel radius R - t; liquid domes (Rd - td, h - td)
        V = prop_kg / rho_liq * (1 + pr["ullage_fraction"])
        Vd = 2 * (2.0 / 3.0) * np.pi * (Rd - td) ** 2 * (h - td)
        if tunnel:
            Vd -= 2 * np.pi * tunnel ** 2 * (h - td) * 0.98   # tunnel through both domes (approx)
        # a short barrel is kept even when the domes alone would hold the load (extra ullage)
        L = max((V - Vd) / (np.pi * ((R - t) ** 2 - (tunnel or 0) ** 2)), lay["min_barrel_m"])
        z1 = add_barrel(name, stage, L, z0, tank=name)
        aft = dome_shell(Rd, td, h, -1, z0)
        fwd = dome_shell(Rd, td, h, +1, z1)
        liq = half_ellipsoid(Rd - td, h - td, -1, z0) + Pos(0, 0, z0) * Cylinder(R - t, L, align=CB) \
            + half_ellipsoid(Rd - td, h - td, +1, z1)
        if tunnel:
            cut = Pos(0, 0, z0 - h - 0.1) * Cylinder(tunnel, L + 2 * h + 0.2, align=CB)
            aft, fwd, liq = aft - cut, fwd - cut, liq - cut
        add(name + "_aft_dome", aft, rho_al, stage, "structure")
        add(name + "_fwd_dome", fwd, rho_al, stage, "structure")
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(liq.wrapped, g)
        add(name + "_liquid", liq, prop_kg / g.Mass(), stage, "propellant", station={"tank": name, "z_bottom": z0})
        info[name] = {"z0": z0, "z1": z1, "Rd": Rd, "td": td, "L": L}
        return z1

    eb, eu = eng["booster"], eng["upper"]
    bay = lay["engine_bay"]
    dc = lay["downcomer"]

    # ---------------- booster structure
    z = add_barrel("b_skirt", "booster", lay["booster_skirt_m"], 0.0)
    z = add_tank("b_fuel", "booster", fu_b, pr["rho_fuel"], z, "b_skirt", "b_intertank", tunnel=dc["radius_m"])
    z_it0 = z
    z = add_barrel("b_intertank", "booster", 2 * h + lay["bay_gap_m"], z)
    z = add_tank("b_ox", "booster", ox_b, pr["rho_ox"], z, "b_intertank", "b_interstage")
    z_is0 = z
    is_len = h + eu["length_m"] + lay["upper_engine"]["mount_drop_m"] + lay["interstage_extra_m"]
    z_sep = add_barrel("b_interstage", "booster", is_len, z)
    info["z_sep"] = z_sep

    # internal ring frames (where no dome reaches the wall)
    add("b_intertank_frame", frame_ring(R - t_of("b_intertank"), z_it0 + h + lay["bay_gap_m"] / 2
                                        - lay["frame"]["height_m"] / 2, spec), rho_al, "booster", "structure")
    add("b_interstage_frame", frame_ring(R - t_of("b_interstage"), z_is0 + h + 0.15, spec), rho_al, "booster", "structure")

    # ---------------- engine bay: thrust plate, thrust cone, engines, heat shield
    tp = bay["thrust_plate"]
    eng_len = eb["length_m"]
    z_exit = -bay["nozzle_below_skirt_m"]
    esolid, e_h = engine_solid(eng_len, eb["exit_diameter_m"] / 2, "booster")
    z_gimbal = z_exit + e_h                       # pivot at the top of the engine
    gb = bay["gimbal_block"]
    z_tp0 = z_gimbal + gb["height_m"]
    z_tp1 = z_tp0 + tp["thickness_m"]
    n = eb["count"]
    ring = bay["ring_radius_frac"] * R
    xy = [(0.0, 0.0)] + [(ring * np.cos(2 * np.pi * k / (n - 1)), ring * np.sin(2 * np.pi * k / (n - 1)))
                         for k in range(n - 1)]
    plate = Pos(0, 0, z_tp0) * Cylinder(tp["radius_m"], tp["thickness_m"], align=CB)
    for k in range(n - 1):                         # lightening holes between the outer engines
        ang = 2 * np.pi * (k + 0.5) / (n - 1)
        plate = plate - Pos(0.55 * ring * np.cos(ang), 0.55 * ring * np.sin(ang), z_tp0 - 0.01) * Cylinder(
            tp["lightening_hole_r_m"], tp["thickness_m"] + 0.02, align=CB)
    add("b_thrust_plate", plate, rho_al, "booster", "structure")
    # thrust cone: shell from the plate rim up to the skirt inner wall
    r_top = R - t_of("b_skirt")
    ct = bay["thrust_cone_t_m"]
    cone = Pos(0, 0, z_tp1) * (Cone(tp["radius_m"], r_top, bay["thrust_cone_top_m"] - z_tp1, align=CB)
                               - Cone(tp["radius_m"] - ct, r_top - ct, bay["thrust_cone_top_m"] - z_tp1, align=CB))
    add("b_thrust_cone", cone, rho_al, "booster", "structure")
    for i, (x, y) in enumerate(xy):
        add(f"b_engine{i}", Pos(x, y, z_exit) * esolid, 1.0, "booster", "engine", mass_override=eb["mass_kg"],
            moving="gimbal" if i == 0 else None)
        add(f"b_gimbal{i}", Pos(x, y, z_gimbal) * Cylinder(gb["radius_m"], gb["height_m"], align=CB), rho_al,
            "booster", "structure")
    info["engines_xy"] = xy
    info["z_gimbal"] = z_gimbal
    info["z_exit"] = z_exit
    # heat shield at the skirt base with holes sized for the nozzle (+ gimbal swing for the centre)
    hs = bay["heat_shield"]
    shield = Pos(0, 0, 0.0) * Cylinder(R - t_of("b_skirt"), hs["thickness_m"], align=CB)
    prof = engine_profile(eng_len, eb["exit_diameter_m"] / 2)
    r_at = float(np.interp(0.0 - z_exit + hs["thickness_m"], [p[1] for p in prof], [p[0] for p in prof]))
    swing = (z_gimbal - 0.0) * np.tan(np.radians(eb["gimbal_max_deg"]))
    for i, (x, y) in enumerate(xy):
        rh = r_at + hs["clearance_m"] + (swing if i == 0 else 0.0)
        shield = shield - Pos(x, y, -0.01) * Cylinder(rh, hs["thickness_m"] + 0.02, align=CB)
    add("b_heat_shield", shield, hs["rho"], "booster", "secondary")

    # ---------------- LOX downcomer through the fuel-tank tunnel to the centre-engine manifold
    ox_bottom = info["b_ox"]["z0"] - h
    dcomer = Pos(0, 0, z_tp1) * (Cylinder(dc["radius_m"] - 0.004, ox_bottom - z_tp1, align=CB)
                                 - Cylinder(dc["radius_m"] - 0.004 - dc["wall_m"], ox_bottom - z_tp1, align=CB))
    add("b_lox_downcomer", dcomer, rho_al, "booster", "secondary")

    # ---------------- raceway (booster + upper), RCS pods
    rw = lay["raceway"]
    add("b_raceway", hollow_sector(R, R + rw["height_m"], 0.55, z_sep - 0.95 - 0.55, rw["width_m"], rw["azimuth_deg"],
                                   rw["wall_m"]),
        rw["rho"], "booster", "secondary")
    rc = lay["rcs"]
    for k in range(rc["count"]):
        azr = rc["azimuth0_deg"] + 360.0 * k / rc["count"]
        add(f"b_rcs{k}", hollow_sector(R, R + rc["depth_m"], z_sep - 0.30, rc["height_m"], rc["width_m"], azr, 0.003),
            rho_al, "booster", "secondary")

    # ---------------- grid fins + actuator housings
    gf = lay["grid_fins"]
    z_fh = z_sep - gf["hinge_below_sep_m"]
    fin = grid_fin_solid(gf["span_m"], gf["width_m"], gf["depth_m"], gf["web_m"], *gf["cells"])
    fin_deg = gf["deploy_deg"] if state == "landing" else 0.0
    hz = gf["housing"]
    for k in range(gf["count"]):
        az = 360.0 * k / gf["count"]
        add(f"b_gridfin{k}", Rot(0, 0, az) * fin_pose(gf, R, z_fh, fin_deg, fin), gf["rho"], "booster", "fin",
            moving="fin")
        add(f"b_fin_housing{k}", hollow_sector(R, R + hz["depth_m"], z_fh - 0.06 - hz["height_m"], hz["height_m"],
                                               hz["width_m"], az, 0.004), rho_al, "booster", "secondary")
    info["z_fin_hinge"] = z_fh

    # ---------------- landing legs (A-frames) + hinge brackets
    lg = lay["legs"]
    leg_deg = lg["deploy_deg"] if state == "landing" else 0.0
    br = lg["bracket"]
    for k in range(lg["count"]):
        az = lg["azimuth0_deg"] + 360.0 * k / lg["count"]
        # struts and pad are one welded weldment: fuse them so their joints are not overlaps
        lp = leg_parts(lg, R, leg_deg)
        weld = lp[0][1] + lp[1][1] + lp[2][1]
        add(f"b_leg{k}", Rot(0, 0, az) * weld, lg["rho"], "booster", "leg", moving="leg")
        add(f"b_leg{k}_pad", Rot(0, 0, az) * (lp[3][1] - lp[0][1]), lg["pad_rho"], "booster", "leg", moving="leg")
        add(f"b_leg{k}_bracket", leg_fitting(R, lg, az), rho_al, "booster", "secondary")

    # ---------------- upper stage
    z = add_barrel("u_aft_skirt", "upper", h + 0.05, z_sep)
    um = lay["upper_engine"]
    usolid, ue_h = engine_solid(eu["length_m"], eu["exit_diameter_m"] / 2, "upper")
    z_ue_top = z_sep - um["mount_drop_m"]
    z_ue_exit = z_ue_top - ue_h
    add("u_engine", Pos(0, 0, z_ue_exit) * usolid, 1.0, "upper", "engine", mass_override=eu["mass_kg"], moving="gimbal")
    info["z_upper_gimbal"] = z_ue_top
    info["z_upper_exit"] = z_ue_exit
    # engine mount: conical shell from the aft-skirt wall (just above the separation plane) to the engine top
    r_m0 = R - t_of("u_aft_skirt")
    mt = um["mount_t_m"]
    z_m_top = z_sep + 0.04
    mount = Pos(0, 0, z_ue_top) * (Cone(0.16, r_m0, z_m_top - z_ue_top, align=CB)
                                   - Cone(0.16 - mt, r_m0 - mt, z_m_top - z_ue_top, align=CB))
    add("u_engine_mount", mount, rho_al, "upper", "structure")
    z = add_tank("u_fuel", "upper", fu_u, pr["rho_fuel"], z, "u_aft_skirt", "u_intertank", tunnel=dc["radius_m"])
    z_uit0 = z
    z = add_barrel("u_intertank", "upper", 2 * h + lay["bay_gap_m"], z)
    add("u_intertank_frame", frame_ring(R - t_of("u_intertank"), z_uit0 + h + lay["bay_gap_m"] / 2
                                        - lay["frame"]["height_m"] / 2, spec), rho_al, "upper", "structure")
    z = add_tank("u_ox", "upper", ox_u, pr["rho_ox"], z, "u_intertank", "u_fwd_skirt")
    z_fs0 = z
    z = add_barrel("u_fwd_skirt", "upper", h + lay["upper_fwd_skirt_extra_m"], z)
    add("u_fwd_skirt_frame", frame_ring(R - t_of("u_fwd_skirt"), z_fs0 + h + 0.05, spec), rho_al, "upper", "structure")
    # upper downcomer: LOX aft dome -> through the fuel tunnel -> engine powerhead
    u_ox_bottom = info["u_ox"]["z0"] - h
    udc = Pos(0, 0, z_ue_top) * (Cylinder(dc["radius_m"] - 0.004, u_ox_bottom - z_ue_top, align=CB)
                                 - Cylinder(dc["radius_m"] - 0.004 - dc["wall_m"], u_ox_bottom - z_ue_top, align=CB))
    add("u_lox_downcomer", udc, rho_al, "upper", "secondary")
    add("u_raceway", hollow_sector(R, R + rw["height_m"], z_sep + 0.10, z - z_sep - 0.15, rw["width_m"], rw["azimuth_deg"],
                                   rw["wall_m"]),
        rw["rho"], "upper", "secondary")

    # payload adapter (conical shell inside the fairing), payload, fairing
    fr = lay["fairing"]
    ft = fr["thickness_m"]
    r_pa0 = R - ft - 0.002
    r_pa1 = 0.45 * R
    pa_t = 2 * tmin
    pa = Pos(0, 0, z) * (Cone(r_pa0, r_pa1, lay["payload_adapter_m"], align=CB)
                         - Cone(r_pa0 - pa_t, r_pa1 - pa_t, lay["payload_adapter_m"], align=CB))
    add("u_payload_adapter", pa, rho_al, "upper", "structure")
    pl = lay["payload"]
    bx, by, bz = pl["bus_m"]
    zp = z + lay["payload_adapter_m"]
    sat = Pos(0, 0, zp) * Cylinder(r_pa1, 0.04, align=CB)                       # separation ring
    sat = sat + Pos(0, 0, zp + 0.04) * Box(bx, by, bz, align=CB)
    for sgn in (1, -1):
        sat = sat + Pos(sgn * (bx / 2 + pl["panel_t_m"] / 2 + 0.005), 0, zp + 0.04 + 0.05) * Box(
            pl["panel_t_m"], by * 0.9, bz - 0.1, align=CB)
    sat = sat + Pos(0, 0, zp + 0.04 + bz) * Cone(0.06, pl["dish_d_m"] / 2, 0.12, align=CB)
    add("payload", sat, 1.0, "payload", "payload", mass_override=spec["mission"]["payload_kg"])
    add("u_fairing_cyl", tube(R, ft, fr["cyl_m"], z), fr["rho"], "upper", "fairing")
    # spherically blunted tangent ogive (exact arc cap, tangent to a spline ogive)
    Ln, rn = fr["nose_m"], fr["nose_radius_m"]
    z0n = z + fr["cyl_m"]
    nose = blunted_ogive(R, Ln, rn, z0n) - blunted_ogive(R - ft, Ln - ft, rn - ft, z0n)
    z_last, rn_ = z0n + blunted_ogive_length(R, Ln, rn), rn
    add("u_fairing_nose", nose, fr["rho"], "upper", "fairing")
    z_top = z_last
    info.update({"z_top": z_top, "length": z_top - z_exit, "z_base": z_exit, "state": state})
    return parts, info


def lumped_mass(spec, stage, struct_mass):
    lump = spec["lumped"]
    return sum(lump[stage].values()) + lump[f"{stage}_misc_fraction"] * struct_mass


def stage_dry_mass(spec, parts, stage):
    """Dry mass of a stage from CAD: every non-propellant solid + lumped items."""
    total = struct = 0.0
    for p in parts:
        if p.stage != stage or p.kind in ("propellant", "payload"):
            continue
        m = p.props()[0]
        total += m
        if p.kind == "structure":
            struct += m
    return total + lumped_mass(spec, stage, struct)


def mass_properties(spec, parts, include=lambda p: True):
    """Combined mass properties of selected parts, with each stage's lumped masses at its CoM."""
    sel = [p for p in parts if include(p)]
    props = [p.props() for p in sel]
    for stage in ("booster", "upper"):
        sp = [pp for p, pp in zip(sel, props) if p.stage == stage and p.kind != "propellant"]
        if not sp:
            continue
        struct = sum(pp[0] for p, pp in zip(sel, props) if p.stage == stage and p.kind == "structure")
        _, c, _ = combine(sp)
        props.append((lumped_mass(spec, stage, struct), c, np.zeros((3, 3))))
    return combine(props)


def sweeps(spec, parts, info, n_steps=12):
    """Swept envelopes of moving parts at full travel, as (name, [solids], ignore-set) for the
    audit. Each envelope is a list of poses rather than one fused solid (fusing dozens of curved
    poses is fragile in OCCT and adds nothing to an interference test).

    Gimbals: the filled nozzle envelope at max deflection in 16 directions plus neutral.
    Legs / fins: solid envelopes (no lattice holes, solid tubes) at n_steps + 1 poses from stowed
    to deployed. Ignored: the moving part itself, its hinge joint and the other moving parts of the
    same kind (they move in unison; both end states are checked statically)."""
    lay, eng = spec["layout"], spec["engines"]
    R = spec["diameter_m"] / 2
    out = []
    for nm, pivot_z, max_deg, length, exit_d, kind, z_exit in (
            ("b_engine0", info["z_gimbal"], eng["booster"]["gimbal_max_deg"], eng["booster"]["length_m"],
             eng["booster"]["exit_diameter_m"], "booster", info["z_exit"]),
            ("u_engine", info["z_upper_gimbal"], lay["upper_engine"]["gimbal_max_deg"], eng["upper"]["length_m"],
             eng["upper"]["exit_diameter_m"], "upper", info["z_upper_exit"])):
        env = Pos(0, 0, z_exit) * revolve_rz(engine_profile(length, exit_d / 2, kind))
        poses = [env]
        for k in range(16):
            az = 2 * np.pi * k / 16
            poses.append(env.rotate(Axis((0, 0, pivot_z), (np.cos(az), np.sin(az), 0.0)), max_deg))
        ignore = {nm, "b_gimbal0", "u_engine_mount", "b_thrust_plate"}
        out.append((f"{nm}_gimbal_{max_deg:g}deg", poses, ignore))
    lg = lay["legs"]
    moving_legs = {p.name for p in parts if p.moving == "leg"}
    for k in range(lg["count"]):
        az = lg["azimuth0_deg"] + 360.0 * k / lg["count"]
        poses = [Rot(0, 0, az) * s for d in np.linspace(0.0, lg["deploy_deg"], n_steps + 1)
                 for _, s, _ in leg_parts(lg, R, d, envelope=True)]
        out.append((f"b_leg{k}_deploy", poses, moving_legs))
    gf = lay["grid_fins"]
    env = fin_envelope(gf["span_m"], gf["width_m"], gf["depth_m"])
    moving_fins = {p.name for p in parts if p.moving == "fin"}
    for k in range(gf["count"]):
        az = 360.0 * k / gf["count"]
        poses = [Rot(0, 0, az) * fin_pose(gf, R, info["z_fin_hinge"], d, env)
                 for d in np.linspace(0.0, gf["deploy_deg"], n_steps + 1)]
        out.append((f"b_gridfin{k}_deploy", poses, moving_fins))
    return out


def export(parts, out_dir, stem="talon1"):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solids = [p.solid for p in parts if p.kind != "propellant"]
    comp = Compound(children=solids)
    export_step(comp, str(out / f"{stem}.step"))
    export_stl(comp, str(out / f"{stem}.stl"), tolerance=0.002, angular_tolerance=0.1)
    return out
