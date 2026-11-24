"""Shaded multi-view renders of the closed Talon-1 design (pyvista, off-screen).

    python -m cad.render [--out docs/design/renders] [--quick]

Views: full stack (ascent), section through the axis, aft end from below, landing configuration
(legs + grid fins deployed), interstage detail and fairing/payload section. --quick renders from
an un-closed sizing (for iterating on geometry).
"""
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
from build123d import Box, Pos
from cad.build import ROOT

pv.OFF_SCREEN = True
COLORS = {
    "structure": "#c9ced6", "secondary": "#8d96a3", "engine": "#3b3b40", "leg": "#26272b", "fin": "#a8794a",
    "fairing": "#f4f4f2", "payload": "#d9a441", "propellant_ox": "#7fb3e6", "propellant_fuel": "#e8a15c",
}


def mesh(solid, tol=0.002, ang=0.15):
    verts, tris = solid.tessellate(tol, ang)
    if not tris:
        return None
    V = np.array([[v.X, v.Y, v.Z] for v in verts])
    F = np.hstack([np.full((len(tris), 1), 3), np.array(tris)]).ravel()
    return pv.PolyData(V, F)


def color(p):
    if p.kind == "propellant":
        return COLORS["propellant_ox"] if "_ox" in p.name else COLORS["propellant_fuel"]
    return COLORS[p.kind]


def add_parts(pl, parts, cut=None, kinds=None, skip=()):
    for p in parts:
        if (kinds and p.kind not in kinds) or p.name in skip:
            continue
        if p.kind == "propellant" and cut is None:
            continue
        s = p.solid
        if cut is not None:
            s = s & cut
            if s is None or s.volume <= 0:
                continue
        m = mesh(s)
        if m is None:
            continue
        pl.add_mesh(m, color=color(p), smooth_shading=True, specular=0.3, specular_power=20,
                    split_sharp_edges=True, opacity=0.35 if (p.kind == "fairing" and cut is None) else 1.0)


def plotter(size=(1100, 1400)):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    pl.add_light(pv.Light(position=(15, -20, 25), intensity=0.6))
    return pl


def shoot(pl, out, focal, position, up=(0, 0, 1), zoom=1.0):
    """Look from `position` toward `focal`, fit everything in view, then zoom."""
    d = np.array(position, float) - np.array(focal, float)
    pl.camera.focal_point = focal
    pl.camera.position = position
    pl.camera.up = up
    pl.reset_camera()
    pl.camera.focal_point = focal
    dist = np.linalg.norm(np.array(pl.camera.position) - np.array(pl.camera.focal_point))
    pl.camera.position = tuple(np.array(focal) + d / np.linalg.norm(d) * dist)
    pl.camera.zoom(zoom)
    pl.screenshot(str(out))
    pl.close()


def render_all(spec, parts, info, lparts, out):
    out.mkdir(parents=True, exist_ok=True)
    zc = 0.5 * (info["z_base"] + info["z_top"])
    files = []
    # 1 full stack
    pl = plotter((900, 1500))
    add_parts(pl, parts)
    shoot(pl, out / "01_full_stack.png", (0, 0, zc), (16, -22, zc + 6), zoom=1.15)
    files.append("01_full_stack.png")
    # 2 section (y < 0 half removed)
    half = Pos(0, 25, 10) * Box(50, 50, 60)               # keep y > 0
    pl = plotter((900, 1500))
    add_parts(pl, parts, cut=half)
    shoot(pl, out / "02_section.png", (0, 0, zc), (0.01, -30, zc), zoom=1.2)
    files.append("02_section.png")
    # 3 aft end from below
    pl = plotter((1100, 1100))
    add_parts(pl, [p for p in parts if p.kind != "propellant"], cut=Pos(0, 0, 0.6) * Box(10, 10, 4.0))
    shoot(pl, out / "03_aft_end.png", (0, 0, 0.0), (2.4, -3.0, -3.4), zoom=1.15)
    files.append("03_aft_end.png")
    # 4 landing configuration (booster only)
    pl = plotter((1200, 1300))
    add_parts(pl, [p for p in lparts if p.stage == "booster"])
    zb = 0.5 * (info["z_base"] + info["z_sep"])
    shoot(pl, out / "04_landing_config.png", (0, 0, zb), (19, -5, zb + 3.0), zoom=1.1)
    files.append("04_landing_config.png")
    # 5 interstage detail, section
    zs = info["z_sep"]
    win = Pos(0, 1.5, zs - 0.4) * Box(6, 3, 4.0)          # keep y > 0: we look from -y into the section
    pl = plotter((1200, 1100))
    add_parts(pl, [p for p in parts if p.kind not in ("fin",)], cut=win)
    add_parts(pl, [p for p in parts if p.kind == "fin"])
    shoot(pl, out / "05_interstage_detail.png", (0, 0, zs - 0.6), (2.8, -4.2, zs + 0.6), zoom=1.0)
    files.append("05_interstage_detail.png")
    # 6 fairing + payload section
    zt = info["z_top"]
    win = Pos(0, 1.5, zt - 2.4) * Box(6, 3, 5.5)
    pl = plotter((1000, 1200))
    add_parts(pl, parts, cut=win)
    shoot(pl, out / "06_fairing_payload.png", (0, 0, zt - 2.4), (3.2, -5.5, zt - 1.2), zoom=1.0)
    files.append("06_fairing_payload.png")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "design" / "renders"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from design import spec as dspec, closure, sizing
    from cad import vehicle as cv
    sp = dspec.load()
    if a.quick:
        s, th = sizing.size(sp, 0.095, 0.095), {}
    else:
        d = closure.close(sp)
        s, th = d.sizing, d.thickness
    parts, info = cv.build(sp, s, th, state="ascent")
    lparts, _ = cv.build(sp, s, th, state="landing")
    files = render_all(sp, parts, info, lparts, Path(a.out))
    print("\n".join(str(Path(a.out) / f) for f in files))


if __name__ == "__main__":
    main()
