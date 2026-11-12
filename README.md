# Talon: reusable launch vehicle design + 6-DOF GNC

A small reusable rocket designed in code (sizing, build123d CAD, aero, structures) and flown in a
validated 6-DOF simulator: optimal ascent, boostback, and a fuel-optimal propulsive landing
(lossless-convexification / G-FOLD) compared against an RL policy, with the landing flight software
run hardware-in-the-loop on the Vortex-V RISC-V core. Plan and milestones: [docs/PLAN.md](docs/PLAN.md).

## Status
- **M0 (6-DOF core)**: quaternion rigid body with variable mass and inertia, RK4, point-mass/J2/uniform
  gravity, US Standard Atmosphere 1976. Analytic regression suite.

## Verify
```
pip install -r requirements.txt
pytest -q tests/
```
