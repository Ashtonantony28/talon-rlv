# Talon-1 CAD audit and polish (2026-11-24)

The M1 model was built to get mass properties for the sizing loop. It was never checked as
geometry. This pass audited it, fixed every fault and rebuilt the parts that were placeholders.
The audit now runs on every build (`python -m cad.build`), and `tests/test_audit.py` makes it a CI
gate. Machine-generated results: [CAD-audit-results.md](CAD-audit-results.md).

## How the audit works (`cad/audit.py`)

| Check | Method |
|---|---|
| Validity | OCCT `BRepCheck` (`is_valid`) and positive volume on every solid, in both configurations |
| Static interference | Every pair of solids whose bounding boxes touch is intersected. Anything over 1 mm³ fails. Bolted joints meet on shared faces, which have zero volume, so no allow-list is needed. A failed boolean counts as a failure, never as clear |
| Swept interference | Moving parts are posed through their full travel and checked against every other solid. The centre booster engine and the upper engine go to max gimbal (6° and 5°) in 16 directions. The 4 legs and 4 grid fins are stepped from stowed to deployed in 13 poses each, using solid envelopes (no lattice holes, solid tubes) |
| Landing configuration | Static check again with legs and fins deployed |
| Functional clearances | Minimum distance for the pairs that matter: gimbal to outer engines and heat shield, upper nozzle to booster dome and interstage, aft dome to thrust structure, stowed fins and legs to the body, and ground clearance under the nozzles |
| Checker controls | `tests/test_audit.py` proves the checker flags a 0.5 m³ overlap and a 100 mm³ sliver, passes face contact and a 0.5 mm gap, and catches a collision that happens only in one pose of a sweep |

## Faults found in the M1 model

| # | Fault | Severity |
|---|---|---|
| 1 | All 8 tank domes were full body radius, so each dome cut through the skirt, intertank, interstage or forward-skirt wall it sat in. That made 9 overlaps of 0.2 to 0.8 L each, and the dome mass was double-counted | Geometry and mass error |
| 2 | Propellant volumes reached into the bay walls next to them (4 overlaps) | Geometry and mass error |
| 3 | Leg foot pads cut into the engine skirt (4 overlaps of 74 cm³ each) | Geometry error |
| 4 | The payload adapter's base ring overlapped the fairing wall | Geometry error |
| 5 | The grid-fin lattice faced sideways, so flow along the vehicle axis could not pass through the cells. The fins could not work as grid fins | Functional error |
| 6 | Engines were plain cones with no chamber, throat, bell, mount or gimbal | Placeholder |
| 7 | There was no thrust structure: engines floated under the aft dome with 14 mm clearance | Missing load path |
| 8 | Legs were single tubes with a box "actuator" and could not fold or deploy | Placeholder |
| 9 | The fairing nose was a sharp cone | Placeholder |
| 10 | No raceway, feed lines, frames, heat shield, RCS or real payload | Missing detail |

## What changed

- **Tanks**: dome outer radius is now `R - max(adjacent wall t)`, so each dome sits inside the wall
  with a weld land of 1 mm or less. Domes are revolved from exact ellipses. Every tank keeps at least
  a 0.10 m barrel.
- **LOX downcomers**: booster and upper-stage downcomers run through a tunnel cut in both fuel domes
  and the fuel volume, down to the centre-engine manifold and the upper-stage powerhead.
- **Engines**: one smooth spline contour of revolution covering chamber, converging section, throat
  and Rao-style bell, with a powerhead, turbopump housing and gimbal block. The booster nozzles stick
  out 0.30 m below the skirt.
- **Engine bay**: a 12 mm thrust plate with lightening holes and a conical thrust shell to the skirt
  wall. A base heat shield has nozzle cut-outs, and the centre cut-out is sized for the 6° gimbal
  swing.
- **Upper-stage engine**: a conical mount from the aft-skirt wall. The 5° gimbal clears the booster
  dome by 348 mm.
- **Landing legs**: CFRP A-frames (main strut plus two side struts, welded as one part) with
  honeycomb crush pads. Each hinges on a fitting with a skin doubler, two clevis lugs and a pin. They
  fold up along the skirt for ascent and deploy 120° to a 3.8 m footprint radius. Nozzle ground
  clearance is 0.98 m.
- **Grid fins**: titanium lattice (6×5 cells) facing the flow, hinged 0.14 m off the skin. They fold
  flat against the interstage for ascent and deploy 90°. Each has a hollow actuator housing.
- **Fairing**: spherically blunted tangent ogive (spline ogive plus an exact arc cap). The payload
  adapter now sits inside the fairing wall.
- **Secondary structure**: internal ring frames in every dry bay, hollow CFRP raceways on both
  stages, 4 RCS pods, and a satellite payload (bus, stowed arrays, antenna, separation ring).

## Effect on the design

The detailed parts weigh more than the placeholders they replaced. Booster dry mass went from
1,483 kg to about 1,850 kg, even after the misc allowance dropped from 10 % to 6 %, because wiring
and feed lines are now modelled. Re-closing the sizing loop put lift-off mass at about 24.7 t. At
60 kN per engine, lift-off thrust-to-weight fell to 1.13, below the 1.35 requirement, so the
booster engines are uprated to 70 kN (105 kg each), giving a lift-off T/W of 1.44. All wall margins
stay positive at FS 1.4. The ascent was re-optimised on the new masses: see
[M2-ascent.md](M2-ascent.md).

## Renders

Generated by `python -m cad.render` into [renders/](renders/): full stack, axial section, aft end,
landing configuration, interstage section and fairing section.
