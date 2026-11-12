# Talon: reusable launch vehicle design + 6-DOF GNC simulation

Status: DRAFT plan, 2026-11-12. Working name "Talon", suggested repo `Ashtonantony28/talon-rlv`.

## 1. Options considered

| | Option | What it shows | Fit with existing work | Risk |
|---|---|---|---|---|
| **A (recommended)** | **Reusable rocket: vehicle design + 6-DOF GNC with optimal ascent and propulsive landing** | Flight dynamics, aero, propulsion, convex-optimization guidance (G-FOLD style), MPC/TVC control, Monte Carlo dispersions, CAD | Reuses NeuroFlight's sim/RL/int8 pipeline, drone-design's build123d CAD flow, and closes the loop on **Vortex-V** (landing guidance + control running as RISC-V flight software in hardware-in-the-loop) | Medium: well-documented physics, all tools pip-installable in cloud |
| B | CubeSat mission (3U): orbit, ADCS, power, thermal, flight software | Systems engineering breadth, astrodynamics, attitude control | ADCS firmware could run on Vortex-V; PCB via Diode/Zener | Medium-low, but less visually striking and overlaps drone PCB work |
| C | Fixed-wing UAV: aero design with VLM + CFD, stability derivatives, autopilot | Aerodynamic design depth, CFD | Close to drone-design; CFD (OpenFOAM/SU2) is CPU-heavy in cloud | Medium-high on compute, less novel next to drone-design |

**Recommendation: A.** It is the most "aerospace" of the three, it is the strongest portfolio piece (the landing-burn problem is a recognised hard GNC problem with published reference solutions to validate against), and it reuses three existing projects instead of duplicating one. B is a good later add-on (same 6-DOF core, different dynamics).

## 2. Goal and headline deliverables

A small two-stage reusable launch vehicle (sounding-rocket scale for the design, with a scaled "Falcon-class" booster config for the landing study), designed in code and flown in a validated 6-DOF simulator:

1. **Vehicle design package**: mass/propulsion sizing (rocket equation + trade study), build123d parametric CAD (tanks, interstage, grid fins, landing legs, engine gimbal), mass properties exported straight from CAD, structural margin checks (axial load, max-q bending, buckling knockdown).
2. **Aerodynamics**: Barrowman + semi-empirical drag (Mach-dependent C_A, C_N_alpha, C_P travel), optionally checked against an open CFD/panel code at a few points; grid-fin control-effectiveness model.
3. **6-DOF simulator** (Python reference, numpy, quaternions, RK4, variable mass and inertia, WGS-84 + J2 gravity, US Std Atmosphere 1976, wind + gust model, TVC actuator dynamics, engine throttle limits, slosh as an optional mode).
4. **GNC**:
   - Ascent: gravity turn + closed-loop optimal ascent (PEG-style or direct collocation) to a target apogee/orbit.
   - Boostback/entry: attitude control with grid fins, entry burn.
   - **Powered descent: lossless-convexification fuel-optimal landing (Açıkmeşe & Blackmore G-FOLD), solved with cvxpy/ECOS online in a receding horizon**, tracked by a TVC attitude controller (LQR/MPC).
   - Navigation: EKF fusing simulated IMU, GNSS, radar altimeter with realistic noise/bias.
   - Baseline vs learned: an RL landing policy (PPO, reusing NeuroFlight's training scaffold) compared head-to-head against G-FOLD on fuel, touchdown error and robustness.
5. **Monte Carlo + V&V**: 1000-run dispersions (mass, Isp, thrust misalignment, winds, sensor errors), landing-ellipse plots, pass/fail criteria; analytic test cases (vacuum ballistic, constant-thrust rocket equation, torque-free precession) as the bit-for-bit regression suite.
6. **Hardware-in-the-loop on Vortex-V**: the descent controller + nav filter compiled to RV64 (fixed-point or int8 via Xmlacc for the RL policy, matching NeuroFlight's requant rules), run in Vortex-V Verilator co-sim in lockstep with the Python plant; report cycle budget per control step at 50 Hz.
7. **Visualisation**: a browser 3D replay (three.js) of a full ascent → boostback → landing flight, plus a report.

## 3. Architecture

```
talon-rlv/
  spec/vehicle.yaml        single source of truth (masses, engines, geometry)
  design/                  sizing, trade studies, structures
  cad/                     build123d models -> STEP/STL + mass properties
  aero/                    Barrowman/semi-empirical tables, grid fins
  sim/                     6-DOF core (dynamics, env, sensors, actuators)
  gnc/                     guidance (ascent, gfold), control (lqr, mpc), nav (ekf)
  rl/                      landing env (gymnasium) + PPO
  hil/                     C flight software + Vortex-V co-sim bridge
  mc/                      Monte Carlo runner + plots
  viz/                     three.js replay
  tests/                   pytest; analytic + golden fixtures
  docs/adr/                decisions
```
Frames: ECEF/ENU for translation, body FRD for attitude, quaternions scalar-first (ADR 0001). Units SI throughout.

## 4. Milestones

| M | Scope | Verify command | Exit criteria |
|---|---|---|---|
| M0 | Repo scaffold, spec, 6-DOF core (rigid body, gravity, atmosphere, variable mass) | `pytest -q tests/` | Analytic cases match to 1e-9 rel; energy/momentum conserved in torque-free tests |
| M1 | Vehicle sizing + build123d CAD + mass properties from CAD | `pytest -q tests/test_design.py` and `python -m cad.build` | CAD mass props within 2% of sizing budget; structural margins > 1.25 |
| M2 | Aero tables + ascent (open-loop gravity turn, then closed-loop optimal) | `python -m sim.fly --mission ascent` | Hits apogee target ±1%; max-q, max-g within limits |
| M3 | Descent: G-FOLD + TVC attitude control, perfect nav | `python -m sim.fly --mission landing` | Touchdown < 1 m, < 1 m/s, upright; matches published G-FOLD benchmark case |
| M4 | EKF nav + sensors, winds, Monte Carlo | `python -m mc.run --n 1000` | ≥ 99% landings within criteria |
| M5 | RL landing policy vs G-FOLD | `python -m rl.eval` | Comparison table; RL robustness under dispersions |
| M6 | HIL on Vortex-V (C flight SW, lockstep co-sim) | `make -C hil cosim` | Controller output bit-exact vs C-on-host; cycles/step reported |
| M7 | Full mission (ascent → boostback → entry → landing), 3D replay, report | `python -m sim.fly --mission full` | End-to-end flight + published replay |

## 5. Tooling (all cloud)

Python 3.11, numpy, scipy, cvxpy (+ ECOS/Clarabel), gymnasium, torch (CPU), build123d, matplotlib; Verilator + RISC-V GCC for M6 (already used by Vortex-V). No hardware purchase, nothing needed on the user's computer.

## 6. Risks

- G-FOLD numerical conditioning: scale variables, use Clarabel; validate on the paper's example first.
- Stiff TVC actuator dynamics with RK4: fixed small step for actuators, sub-stepping.
- HIL cycle budget: fall back to fixed-point controller only, RL policy as stretch.
- Scope creep: B (CubeSat) is explicitly out of scope until M7.

## 7. Decisions needed from the user

1. Direction A vs B vs C (proceeding on A).
2. Create the GitHub repo: `gh repo create Ashtonantony28/talon-rlv --public` (no README). Until then work lives in the container with a bundle at `/mnt/project-files/project-plans/aerospace/talon-rlv.bundle`.
