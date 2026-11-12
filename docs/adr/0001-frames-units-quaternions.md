# ADR 0001: Frames, units and quaternion convention

Status: Accepted (2026-11-12)

## Context
GNC bugs are most often frame or sign bugs. One convention must hold across sim, GNC, flight software and HIL.

## Decision
- SI units everywhere (m, s, kg, rad, N). Degrees only at I/O boundaries.
- Translational state in an inertial frame: Earth-centred for orbital/ascent work, local ENU (z up) for
  landing studies with uniform gravity.
- Body frame: x along the vehicle long axis (nose), y/z completing a right-handed set.
- Quaternions scalar-first `[w, x, y, z]`, Hamilton product, representing body-to-inertial rotation
  (`v_inertial = C(q) v_body`). Kinematics `q_dot = 0.5 q ⊗ [0, ω_body]`.
- Forces enter the equations of motion in the inertial frame, torques in the body frame.
- State vector `[r, v, q, ω, m]` (14 elements); inertia is a function of mass.

## Consequences
Flight software (M6) uses the same layout so HIL can compare state vectors element for element.
