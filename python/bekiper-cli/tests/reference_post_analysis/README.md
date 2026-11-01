# Migration reference

These modules retain the previous Python implementation as a test-only migration
oracle for valid existing fixtures. The supported runtime is the native Rust
engine. This reference is excluded from the Python wheel.

The old implementation predates the native fixes for competing NORT intervals,
impossible imported cross-fields, upstream QC retention and empty planned
hypotheses. Its tests establish historical parity only; the native Rust tests
and installed-style integration tests verify the current behaviour.

The reference also retains the historical NORT rule (minimum 1 s, gaps <0.5 s,
median-speed exclusion). It is not the October 7, 2026 amended policy: current
neo/imen scoring uses bouts ≥0.5 s, gaps ≤1.0 s, and descriptive-only speed.
Keep historical fixture provenance explicit; amended boundary, missing-speed,
competing-object and provenance checks live in the native Rust scoring tests.
