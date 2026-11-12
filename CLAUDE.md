# CLAUDE.md: conventions for Talon

1. **Every change ships with a Verify command** in its PR description (pytest invocation, mission run, Monte Carlo).
2. **Analytic before numeric.** Every physics model has an analytic or published reference test in `tests/`
   before it is used in a mission. Tolerances are stated, never loosened silently.
3. **`spec/vehicle.yaml` is the single source of truth** for vehicle parameters; no magic numbers in sim/gnc code.
4. Frames, units and quaternion convention follow ADR 0001. Hard-to-reverse decisions get an ADR in `docs/adr/`.
5. Never commit to `main` directly; one logical change per PR; red CI never merges; commit messages explain why.
