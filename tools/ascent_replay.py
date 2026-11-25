"""Animated replay of the simulated Talon-1 ascent (M2) for the README.

    python -m tools.ascent_replay [--out docs/media] [--seconds 14] [--fps 20]

Flies the committed guidance exactly as `python -m sim.fly --mission ascent` does, then draws the
trajectory (Earth-fixed downrange vs altitude) and the altitude / speed / dynamic pressure /
sensed-acceleration traces as they unfold. Writes ascent-replay.mp4 and ascent-replay.gif.
This is simulation output (3-DOF point mass, perfect attitude tracking), and the frame says so.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, GRID, TXT, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
C_BOOST, C_UPPER, C_COAST, ACC, WARN = "#58a6ff", "#f0883e", "#8b949e", "#3fb950", "#f85149"


def fly():
    from design import spec as dspec, closure
    from sim.fly import fly_ascent
    from sim.ascent import OMEGA_E
    from sim import gravity
    sp = dspec.load()
    veh, p, res, circ = fly_ascent(sp, closure.close(sp))
    assert res.ok, res.reason
    t = res.t
    r = res.X[:, :3]
    th = -OMEGA_E * t                                    # inertial -> Earth-fixed
    c, s = np.cos(th), np.sin(th)
    rf = np.stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]], 1)
    u = rf / np.linalg.norm(rf, axis=1, keepdims=True)
    ang = np.arccos(np.clip(u @ u[0], -1, 1))
    L = {k: np.asarray(v) for k, v in res.log.items() if k != "phase"}
    L["phase"] = np.array(res.log["phase"])
    L["down"] = gravity.R_EQ * ang
    return sp, res, circ, L


def draw(sp, res, circ, L, out, seconds, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.labelcolor": DIM,
                         "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": GRID})
    t, alt, down = L["t"], L["alt"] / 1e3, L["down"] / 1e3
    ph = L["phase"]
    col = np.where(ph == "booster", C_BOOST, np.where(ph == "upper", C_UPPER, C_COAST))
    ev = res.events
    i_q = int(np.argmax(L["q"]))
    marks = [(t[i_q], "Max-Q"), (ev["meco"]["t"], "MECO / staging"), (ev["upper_ignition"]["t"], "Upper stage ignition"),
             (ev["fairing_sep"]["t"], "Fairing separation"), (ev["seco"]["t"], "SECO")]
    n = int(seconds * fps)
    # ease: spend more frames on the dynamic first two minutes
    tau = np.linspace(0, 1, n) ** 1.6 * t[-1]
    hold = int(1.5 * fps)
    tau = np.concatenate([tau, np.full(hold, t[-1])])
    W, H, DPI = 12.8, 7.2, 100
    tmp = Path(tempfile.mkdtemp())
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.65, 1], left=0.06, right=0.97, top=0.86, bottom=0.08,
                          hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    sub = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    series = [(alt, "altitude  km", 1), (L["vel"], "air-relative speed  m/s", 1), (L["q"] / 1e3, "dynamic pressure  kPa", 1),
              (L["accel_g"], "sensed acceleration  g", 1)]
    for a in [ax] + sub:
        a.set_facecolor(PANEL)
        a.grid(color=GRID, lw=0.6)
        for s_ in a.spines.values():
            s_.set_color(GRID)
    ax.set_xlim(0, down[-1] * 1.04)
    ax.set_ylim(0, alt.max() * 1.12)
    ax.set_xlabel("downrange (Earth-fixed)  km")
    ax.set_ylabel("altitude  km")
    ax.plot(down, alt, color=GRID, lw=1.0, ls=(0, (2, 3)), zorder=1)
    segs = np.stack([np.c_[down[:-1], alt[:-1]], np.c_[down[1:], alt[1:]]], 1)
    lc = LineCollection([], linewidths=2.6, zorder=3)
    ax.add_collection(lc)
    glow, = ax.plot([], [], "o", ms=16, color="#ffffff", alpha=0.15, zorder=4)
    dot, = ax.plot([], [], "o", ms=7, color="#ffffff", zorder=5)
    hud = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="DejaVu Sans Mono", fontsize=12, color=TXT)
    evtxt = ax.text(0.97, 0.06, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color=ACC, weight="bold",
                    linespacing=1.5, bbox=dict(boxstyle="round,pad=0.5", fc=BG, ec=GRID, alpha=0.0))
    for k, (tm, lab) in enumerate(marks):
        j = int(np.searchsorted(t, tm))
        j = min(j, len(t) - 1)
        ax.plot(down[j], alt[j], "D", ms=5, color=TXT, zorder=6, alpha=0.0, gid=f"m{k}")
    mk = [l for l in ax.lines if l.get_gid() and l.get_gid().startswith("m")]
    offs = [(10, -3, "left"), (10, -14, "left"), (10, 4, "left"), (10, -12, "left"), (-12, -20, "right")]
    labels = []
    for (tm, lab), (dx, dy, ha) in zip(marks, offs):
        j = min(int(np.searchsorted(t, tm)), len(t) - 1)
        labels.append(ax.annotate(lab, (down[j], alt[j]), xytext=(dx, dy), textcoords="offset points", fontsize=9.5,
                                  color=TXT, alpha=0.0, ha=ha))
    lines, cursors = [], []
    for a, (y, lab, _) in zip(sub, series):
        a.set_xlim(0, t[-1])
        a.set_ylim(min(0, y.min()), y.max() * 1.18)
        a.set_title(lab, fontsize=9.5, color=DIM, loc="left", pad=3)
        a.tick_params(labelsize=8)
        a.plot(t, y, color=GRID, lw=0.8)
        ln = LineCollection([], linewidths=1.8)
        a.add_collection(ln)
        lines.append((ln, np.stack([np.c_[t[:-1], y[:-1]], np.c_[t[1:], y[1:]]], 1)))
        cursors.append(a.axvline(0, color=TXT, lw=0.6, alpha=0.6))
    sub[2].set_ylim(0, sp["structures"]["design_max_q_pa"] / 1e3 * 1.25)
    sub[2].axhline(sp["structures"]["design_max_q_pa"] / 1e3, color=WARN, lw=0.8, ls="--")
    sub[2].text(t[-1], sp["structures"]["design_max_q_pa"] / 1e3, "design max-q ", color=WARN, fontsize=7.5, ha="right", va="bottom")
    sub[3].axhline(sp["structures"]["max_axial_g"], color=WARN, lw=0.8, ls="--")
    sub[3].set_xlabel("mission time  s", fontsize=9)
    fig.text(0.06, 0.94, "Talon-1 ascent to a 500 km circular orbit", fontsize=17, weight="bold", color=TXT)
    fig.text(0.06, 0.905, "SIMULATION  ·  3-DOF point mass on a rotating J2 Earth, US Std. Atmosphere 1976, optimised "
             "linear-tangent upper-stage steering  ·  python -m tools.ascent_replay", fontsize=9, color=DIM)
    fig.text(0.97, 0.94, "", fontsize=1)
    leg = [("booster", C_BOOST), ("coast", C_COAST), ("upper stage", C_UPPER)]
    for k, (lab, cc) in enumerate(leg):
        fig.text(0.53 - 0.085 * (len(leg) - 1 - k), 0.865, "━ " + lab, color=cc, fontsize=9.5, ha="right")
    for f, T in enumerate(tau):
        j = int(np.searchsorted(t, T, side="right")) - 1
        j = max(j, 1)
        lc.set_segments(segs[:j])
        lc.set_color(col[:j])
        dot.set_data([down[j]], [alt[j]])
        glow.set_data([down[j]], [alt[j]])
        dot.set_color(col[j])
        hud.set_text(f"T+{t[j]:6.1f} s\nalt {alt[j]:7.1f} km\nv   {L['vel'][j]:7.0f} m/s\nMach{L['mach'][j]:6.2f}\nq   {L['q'][j] / 1e3:7.1f} kPa\n"
                     f"{L['accel_g'][j]:9.2f} g   thr {100 * L['throttle'][j]:3.0f}%")
        cur = ""
        for k, (tm, lab) in enumerate(marks):
            on = t[j] >= tm - 0.5
            mk[k].set_alpha(1.0 if on else 0.0)
            labels[k].set_alpha(1.0 if on else 0.0)
            if on and t[j] - tm < 18:
                cur = lab
        if j >= len(t) - 2:
            cur = (f"SECO into {res.events['seco']['perigee'] / 1e3:.0f} × {res.events['seco']['apogee'] / 1e3:.0f} km\n"
                   f"+ {circ['dv']:.0f} m/s apogee burn → {circ['final_alt_km']:.0f} km circular\n"
                   f"{circ['margin_kg']:.0f} kg upper-stage propellant margin")
            evtxt.set_fontsize(12)
            evtxt.get_bbox_patch().set_alpha(0.9)
        evtxt.set_text(cur)
        for (ln, sg), cu in zip(lines, cursors):
            ln.set_segments(sg[:j])
            ln.set_color(col[:j])
            cu.set_xdata([t[j], t[j]])
        fig.savefig(tmp / f"f{f:04d}.png", facecolor=BG)
    plt.close(fig)
    out.mkdir(parents=True, exist_ok=True)
    mp4, gif = out / "ascent-replay.mp4", out / "ascent-replay.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"), "-c:v", "libx264",
                    "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=none:diff_mode=rectangle", str(gif)], check=True)
    subprocess.run(["cp", str(tmp / f"f{len(tau) - 1:04d}.png"), str(out / "ascent-replay-final.png")], check=True)
    return mp4, gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--seconds", type=float, default=14)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    sp, res, circ, L = fly()
    for f in draw(sp, res, circ, L, Path(a.out), a.seconds, a.fps):
        print(f, f"{f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
