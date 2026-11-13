"""Render a cutaway side view of the closed design to build/talon1.png.

    python -m cad.render
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from build123d import Box, Pos
from design import spec as dspec, closure
from cad.build import ROOT

COLORS = {"structure": "#9aa7b4", "engine": "#3d3d3d", "leg": "#2b2b2b", "fin": "#7a5c3a",
          "fairing": "#f2f2f2", "payload": "#d9a441", "propellant_ox": "#6fa8dc", "propellant_fuel": "#e69138"}


def main():
    sp = dspec.load()
    d = closure.close(sp)
    half = Pos(0, -50, 0) * Box(100, 100, 100)  # keep y < 0: cutaway through the axis
    fig, ax = plt.subplots(figsize=(4, 12), dpi=150)
    order = ["structure", "propellant", "payload", "engine", "fin", "leg", "fairing"]
    for kind in order:
        for p in d.parts:
            if p.kind != kind:
                continue
            solid = p.solid & half if kind in ("structure", "fairing", "propellant") else p.solid
            try:
                verts, tris = solid.tessellate(0.01)
            except Exception:
                continue
            V = np.array([[v.X, v.Z] for v in verts])
            polys = [V[list(t)] for t in tris]
            key = kind if kind != "propellant" else ("propellant_ox" if "_ox" in p.name else "propellant_fuel")
            alpha = {"fairing": 0.35, "structure": 0.6}.get(kind, 1.0)
            ax.add_collection(PolyCollection(polys, facecolors=COLORS[key], edgecolors="none", alpha=alpha))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.3, d.info["length"] + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_title(f"Talon-1  GLOW {d.sizing.glow / 1000:.1f} t, {d.info['length']:.1f} m", fontsize=9)
    out = ROOT / "docs" / "design" / "talon1.png"
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
