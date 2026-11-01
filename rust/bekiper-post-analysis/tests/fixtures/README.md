# Native statistics numerical evidence (F3)

All observations here are deterministic synthetic inputs. They are numerical
regression fixtures, not biological measurements or production decision approvals.

## Independent oracle

`generate_statistics_oracle.py` creates `statistics_oracle.json` using eight
cell-indicator columns, SciPy QR covariance, and an inclusion-exclusion change
of basis. It does not import the historical Python post-analysis engine or the
native implementation. The native implementation uses full factorial treatment
coding and a column-scaled SVD, with two residual-correction iterations.

The oracle records dependency versions and verifies unbalanced post-only and
baseline-adjusted models, numeric age and categorical sex covariates, coefficient
estimates/uncertainty, covariance, the signed `(G5-G6)-(G1-G2)` contrast, confidence
intervals, P values, fitted values, residuals, and leverage. It also supplies tied
Spearman and one-sample Student-t checks. Positive P values have an additional
relative-error assertion; they are not compared only with an absolute tolerance.

Regenerate within the existing project Python environment:

```sh
python rust/bekiper-post-analysis/tests/fixtures/generate_statistics_oracle.py
```

The extreme Student-t test separately uses an 80-digit mpmath 1.3.0 oracle:

```python
import mpmath as m
m.mp.dps = 80
m.log(m.betainc(500, m.mpf("0.5"), 0, m.mpf(1) / 1001, regularized=True))
# -3458.0568098975393881360189777227194455356509391407185387936322619937175271439597
```

The Cauchy tail and balanced/multilevel cell-means tests use analytic identities
instead of the generated QR oracle.

## Focused verification, 2026-10-04

The test target includes `src/statistics.rs` and `src/contracts.rs` directly.
During parallel implementation, coordinator-owned `lib.rs` and binary wiring
were incomplete. An isolated Cargo manifest at
`/data/scratch/tmp/opencode/bekiper-f3-statistics-ef76924e/Cargo.toml` used
`anyhow`, `serde`, `serde_json`, `nalgebra 0.32.6` and `statrs 0.18.0`, with its
`[[test]]` path pointing at the repository's `tests/statistics.rs`.

| Command | Result |
| --- | --- |
| `cargo test --manifest-path /data/scratch/tmp/opencode/bekiper-f3-statistics-ef76924e/Cargo.toml --test statistics -j 2` | **20 passed; 0 failed; 0 ignored** |
| `cargo clippy --manifest-path /data/scratch/tmp/opencode/bekiper-f3-statistics-ef76924e/Cargo.toml --test statistics -j 2 -- -D warnings` | **Passed** |
| Scoped `cargo fmt` check | **Tool permission denied; unverified** |

These results verify F3's source module and tests, not the integrated binary or
workspace. After shared crate wiring, the package-level gate is:

```sh
cargo test -p bekiper-post-analysis --test statistics -j 2
```

## Behaviour and integration notes

- Public signatures follow `NATIVE_CONTRACT.md`; shared library integration
  needs `pub mod statistics;` in coordinator-owned `lib.rs`.
- Full interactions remain in rank-deficient and residual-DF failures. No
  reduced model or covariance substitution is performed.
- Pairing keeps animal/assay/phase/threshold/pair identity and checks verification
  at both timepoints. `DeltaSA` is emitted only for the canonical principal SA
  endpoint; other endpoints expose `paired_difference` and `pre_outcome`.
- Inclusion ledgers retain QC flags, metadata exclusion reasons, missing fields,
  missing reasons, and verification status. Flags alone do not exclude animals.
- Registered empty groups yield explicit unavailable attempts. Holm counts every
  supplied planned attempt, including null and log-only P values, and clears
  stale corrections before recomputing them.
- P-value underflow produces null plus finite log probability, not literal zero.
- Censored latency cannot enter Gaussian OLS. Invalid boolean censoring fields
  are rejected instead of being interpreted as uncensored.
- Residuals and leverage are available. Shapiro–Wilk is explicitly marked
  `unavailable:not_implemented_in_native_backend`; it has not been ported.
