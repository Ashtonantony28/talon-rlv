"""CAD audit: the checker itself (positive/negative controls), then the closed design."""
import pytest
from build123d import Box, Pos
from cad import audit, vehicle as cv
from design import spec as dspec

SPEC = dspec.load()


def _p(name, solid):
    return cv.Part(name, solid, 1000.0, "booster", "structure")


def test_checker_flags_overlap_and_passes_face_contact():
    a = _p("a", Box(1, 1, 1))
    b = _p("b", Pos(0.5, 0, 0) * Box(1, 1, 1))          # 0.5 m^3 overlap
    c = _p("c", Pos(1.0, 0, 0) * Box(1, 1, 1))          # shares a face with a
    d = _p("d", Pos(0.0, 0, 1.0005) * Box(1, 1, 1))     # 0.5 mm gap
    hits = audit.static_interference([a, b, c, d])
    pairs = {frozenset((h["a"], h["b"])) for h in hits}
    assert frozenset(("a", "b")) in pairs and frozenset(("b", "c")) in pairs
    assert frozenset(("a", "c")) not in pairs and frozenset(("a", "d")) not in pairs


def test_checker_flags_tiny_overlap_above_1mm3():
    a = _p("a", Box(1, 1, 1))
    b = _p("b", Pos(0.0, 0, 0.5 + 0.5 - 0.00001) * Box(0.1, 0.1, 1))   # 0.1 m^2 x 10 um = 100 mm^3
    assert audit.static_interference([a, b])


def test_swept_check_sees_every_pose():
    target = _p("t", Pos(3, 0, 0) * Box(0.2, 0.2, 0.2))
    poses = [Pos(x, 0, 0) * Box(0.2, 0.2, 0.2) for x in (0.0, 1.0, 2.0, 3.0)]
    assert audit.swept_interference([("mover", poses, set())], [target])
    assert not audit.swept_interference([("mover", poses[:3], set())], [target])


@pytest.fixture(scope="module")
def configs(design):
    asc = design.parts, design.info
    lan = cv.build(SPEC, design.sizing, design.thickness, state="landing")
    return asc, lan


def test_every_solid_is_valid(configs):
    (parts, _), (lparts, _) = configs
    assert audit.validity(parts) == [] and audit.validity(lparts) == []


def test_no_static_interference_in_either_configuration(configs):
    (parts, _), (lparts, _) = configs
    skip = lambda a, b: a.kind == "propellant" and b.kind == "propellant"
    assert audit.static_interference(parts, skip) == []
    assert audit.static_interference(lparts, skip) == []


def test_moving_parts_clear_at_full_travel(configs):
    (parts, info), _ = configs
    assert audit.swept_interference(cv.sweeps(SPEC, parts, info), parts) == []


def test_functional_clearances(configs):
    (parts, info), (lparts, linfo) = configs
    rows = audit.clearances(SPEC, parts, info, cv.sweeps(SPEC, parts, info), lparts, linfo)
    bad = [r for r in rows if not r["ok"]]
    assert not bad, bad
