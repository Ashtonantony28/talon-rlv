"""CAD turntables of the closed Talon-1 design for the README (pyvista, off-screen).

    python -m tools.turntable [--out docs/media] [--frames 72]

Writes turntable-ascent.{gif,mp4} (full stack, fairing translucent) and turntable-landing.{gif,mp4}
(booster with legs and grid fins deployed). Geometry is the same closed design `python -m cad.render`
draws; colours and lighting come from cad/render.py.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

from cad.build import ROOT
from cad.render import add_parts

pv.OFF_SCREEN = True


def spin(parts, focal, radius, elev, size, frames, out, stem, title, zoom=1.0):
    tmp = Path(tempfile.mkdtemp())
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("#ffffff", top="#dfe6ee")
    pl.enable_anti_aliasing("ssaa")
    pl.add_light(pv.Light(position=(15, -20, 25), intensity=0.6))
    add_parts(pl, parts)
    pl.add_text(title, position="upper_left", font_size=11, color="#24292f")
    pl.add_text("Talon-1  ·  build123d CAD, closed design", position="lower_left", font_size=8, color="#57606a")
    for k in range(frames):
        a = 2 * np.pi * k / frames - np.pi / 3
        pl.camera.focal_point = focal
        pl.camera.position = (focal[0] + radius * np.cos(a), focal[1] + radius * np.sin(a), focal[2] + elev)
        pl.camera.up = (0, 0, 1)
        pl.screenshot(str(tmp / f"f{k:03d}.png"))
    pl.close()
    mp4, gif = out / f"{stem}.mp4", out / f"{stem}.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(tmp / "f%03d.png"), "-c:v", "libx264",
                    "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    w = size[0] // 2 * 2 * 3 // 5 // 2 * 2
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "18", "-i", str(tmp / "f%03d.png"), "-vf",
                    f"scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
                    "-loop", "0", str(gif)], check=True)
    return mp4, gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--frames", type=int, default=72)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from design import spec as dspec, closure
    from cad import vehicle as cv
    sp = dspec.load()
    d = closure.close(sp)
    parts, info = cv.build(sp, d.sizing, d.thickness, state="ascent")
    lparts, _ = cv.build(sp, d.sizing, d.thickness, state="landing")
    zc = 0.5 * (info["z_base"] + info["z_top"])
    h = info["z_top"] - info["z_base"]
    files = spin(parts, (0, 0, zc), 2.15 * h, 0.10 * h, (560, 1200), a.frames, out, "turntable-ascent",
                 "Ascent configuration", zoom=2.75)
    booster = [p for p in lparts if p.stage == "booster"]
    zb = 0.5 * (info["z_base"] + info["z_sep"])
    hb = info["z_sep"] - info["z_base"]
    files += spin(booster, (0, 0, zb - 0.1 * hb), 2.5 * hb, 0.3 * hb, (900, 1000), a.frames, out, "turntable-landing",
                  "Booster, landing configuration", zoom=1.6)
    for f in files:
        print(f, f"{f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
