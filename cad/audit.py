"""Geometric audit of the Talon CAD.

    python -m cad.audit [--json build/audit.json]

Checks
  1. validity: every solid passes OCCT's BRepCheck (is_valid) and has positive volume.
  2. static interference: every pair of solids, intersection volume must be <= 1 mm^3.
  3. swept interference: moving parts at full travel (engine gimbal cone, landing-leg deployment
     arc, grid-fin deployment arc) against every other solid they could reach.
Contacts by design (bolted joints) share faces and have zero intersection volume, so no allow-list
is needed; anything listed in ALLOWED must name the build step that makes room for it.
"""
import argparse
import json
import itertools
import numpy as np

TOL_M3 = 1e-9          # 1 mm^3
ALLOWED = {}           # frozenset({a, b}) -> reason; must stay empty unless justified


def _bbox(s):
    b = s.bounding_box()
    return np.array([b.min.X, b.min.Y, b.min.Z]), np.array([b.max.X, b.max.Y, b.max.Z])


def _overlap(b1, b2, pad=1e-6):
    return np.all(b1[0] <= b2[1] + pad) and np.all(b2[0] <= b1[1] + pad)


def intersection_volume(a, b):
    try:
        r = a & b
    except Exception:
        return float("nan")          # a failed boolean is reported, never treated as clear
    return 0.0 if r is None else float(r.volume)   # build123d returns None for an empty result


def validity(parts):
    bad = []
    for p in parts:
        ok = p.solid.is_valid() if callable(getattr(p.solid, "is_valid", None)) else p.solid.is_valid
        if not ok or p.solid.volume <= 0:
            bad.append({"part": p.name, "is_valid": bool(ok), "volume": p.solid.volume})
    return bad


def static_interference(parts, skip=lambda a, b: False):
    boxes = {p.name: _bbox(p.solid) for p in parts}
    hits = []
    for a, b in itertools.combinations(parts, 2):
        if skip(a, b) or not _overlap(boxes[a.name], boxes[b.name]):
            continue
        v = intersection_volume(a.solid, b.solid)
        if not (v <= TOL_M3):
            key = frozenset({a.name, b.name})
            hits.append({"a": a.name, "b": b.name, "volume_mm3": v * 1e9, "allowed": ALLOWED.get(key)})
    return hits


def swept_interference(sweeps, parts):
    """sweeps: list of (name, [pose solids], set_of_part_names_to_ignore). Reports the worst pose
    per (sweep, part) pair."""
    boxes = {p.name: _bbox(p.solid) for p in parts}
    hits = []
    for name, poses, ignore in sweeps:
        worst = {}
        for solid in poses:
            bs = _bbox(solid)
            for p in parts:
                if p.name in ignore or not _overlap(bs, boxes[p.name]):
                    continue
                v = intersection_volume(solid, p.solid)
                if not (v <= TOL_M3):
                    worst[p.name] = max(worst.get(p.name, 0.0), v)
        hits += [{"sweep": name, "part": k, "volume_mm3": v * 1e9} for k, v in worst.items()]
    return hits


def min_distance(a, b):
    try:
        return float(a.distance_to(b))
    except Exception:
        return float("nan")


def clearances(spec, parts, info, sweeps, landing_parts, landing_info):
    """Minimum distances that matter for function, each with a required value (m)."""
    P = {p.name: p.solid for p in parts}
    L = {p.name: p.solid for p in landing_parts}
    sw = {n: poses for n, poses, _ in sweeps}
    req = 0.010
    rows = []

    def row(what, d, need=req):
        rows.append({"check": what, "min_m": d, "required_m": need, "ok": bool(d >= need)})

    g = sw[[k for k in sw if k.startswith("b_engine0_gimbal")][0]]
    outer = [n for n in P if n.startswith("b_engine") and n != "b_engine0"]
    row("centre engine at max gimbal vs outer engines", min(min_distance(s, P[o]) for s in g for o in outer))
    row("centre engine at max gimbal vs heat shield", min(min_distance(s, P["b_heat_shield"]) for s in g))
    gu = sw[[k for k in sw if k.startswith("u_engine_gimbal")][0]]
    row("upper engine at max gimbal vs booster LOX dome", min(min_distance(s, P["b_ox_fwd_dome"]) for s in gu))
    row("upper engine at max gimbal vs interstage wall", min(min_distance(s, P["b_interstage"]) for s in gu))
    row("booster fuel aft dome vs thrust cone", min_distance(P["b_fuel_aft_dome"], P["b_thrust_cone"]))
    row("booster fuel aft dome vs engine powerheads",
        min(min_distance(P["b_fuel_aft_dome"], P[o]) for o in P if o.startswith("b_engine")))
    row("stowed grid fin vs body", min(min_distance(P[f"b_gridfin{k}"], P["b_interstage"]) for k in range(4)))
    row("stowed leg vs skirt", min(min_distance(P[f"b_leg{k}"], P["b_skirt"]) for k in range(4)))
    # landing: pads must reach below the nozzle exits
    pad_z = min(L[f"b_leg{k}_pad"].bounding_box().min.Z for k in range(4))
    noz_z = min(L[n].bounding_box().min.Z for n in L if n.startswith("b_engine"))
    row("ground clearance: nozzle exit above pad bottom", noz_z - pad_z, 0.30)
    return rows


def run(spec, parts, sweeps=()):
    rep = {"n_solids": len(parts), "invalid": validity(parts),
           "static": static_interference(parts, skip=lambda a, b: a.kind == "propellant" and b.kind == "propellant"),
           "swept": swept_interference(list(sweeps), parts)}
    rep["pass"] = not rep["invalid"] and all(h["allowed"] for h in rep["static"]) and not rep["swept"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from design import spec as dspec, closure
    from cad import vehicle as cv
    sp = dspec.load()
    d = closure.close(sp)
    sweeps = cv.sweeps(sp, d.parts, d.info)
    rep = run(sp, d.parts, sweeps)
    lparts, linfo = cv.build(sp, d.sizing, d.thickness, state="landing")
    rep["landing_static"] = static_interference(lparts, skip=lambda a, b: a.kind == "propellant" and b.kind == "propellant")
    rep["clearances"] = clearances(sp, d.parts, d.info, sweeps, lparts, linfo)
    rep["pass"] = rep["pass"] and not rep["landing_static"] and all(c["ok"] for c in rep["clearances"])
    print(json.dumps(rep, indent=1, default=float))
    if a.json:
        open(a.json, "w").write(json.dumps(rep, indent=1, default=float))
    raise SystemExit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
