# Decision register

This file is **implementation commentary**, not part of the source recipe. It identifies ambiguities visible in the supplied document and practical choices that the document does not define. No underlying thesis, criteria document, dataset, or repository was supplied with this bundle.

## Resolution protocol

Use an existing documented analysis-plan decision where available; otherwise obtain an explicit researcher decision before affected production outputs. Record the value, scope, rationale, provenance, and approval/reference. Do not select a policy after examining which gives a favourable result. A synthetic development fixture can make a labelled choice without resolving the scientific question. Block only dependent outputs, preserve unaffected results, and report the overall run as partial when appropriate.

## Registered amendment YM-2026-10-04

The explicit researcher clarification in this implementation session resolves
principal 25/45/65% entry timing and dwell: instantaneous same-arm head-center
AND tail-base crossing, event at the second crossing, **no dwell requirement**.
The inspected Imen geometry is `imen_radial_vertex_span_v1`; see the normative
amendment in `BEHAVIORAL_RECIPE.md` and `YMAZE_AMENDMENT.json`. This partially
resolves D04 and the geometry discrepancy in D03. D01, primary-threshold
selection, inference, and upstream missing-pose policies remain distinct.
Historical mitigations and conflicting documentation must be retained in run
provenance. A no-dwell result is not incompatible on that account.

## D01 — Y-maze table conflicts with its verbal definition

**Source:** SRC-YM, triad table. The neutral sentence says “First two entries are the same,” but the listed neutral triplets include `ABB`, `ACC`, `BAA`, `BCC`, `CAA`, and `CBB`, whose first two entries differ. Conversely, `AAB`, `AAC`, `BBA`, `BBC`, `CCA`, and `CCB` have equal first two entries but are listed as incorrect. This is not a transcription correction.

**Decision:** confirm whether to use the explicit table or a researcher-supplied corrected rule. The literal lists partition all 27 triplets and can be implemented as an explicitly selected `source_table` policy. Do not infer a corrected list from the prose. Any alternative requires a full auditable mapping and a documented amendment.

**Affects:** Y-maze counts, both SA scores, neutral percentage, usable-triad QC, downstream statistics and figures. The supplied numerical Y-maze fixtures are conditional on `source_table`.

## D02 — Fewer than three valid entries

**Source:** SRC-YM states `total_triads = valid_total_arm_entries - 2` and gives NA handling for a zero denominator, but says nothing about zero or one valid entry. Literal substitution gives negative counts.

**Engineering safeguard:** treat such sessions as unscorable and do not produce negative valid counts or a percentage from a nonpositive denominator. A proposed representation is `total_triads = max(N - 2, 0)` with missing percentages and an `insufficient_entries` reason. This is an explicit boundary extension, not a formula quoted from the recipe. Record approval or retain an invalid/missing total for these sessions until resolved.

**Affects:** only short-entry sessions and their downstream inclusion. Synthetic fixtures label the proposed zero-window convention.

## D03 — Primary Y-maze threshold

**Source:** SRC-YM allows 25%, 45%, or 65% of arm length and prefers the selected thesis threshold; its value is not given.

**Decision:** register one primary threshold per dataset, record calibration and arm geometry, and run other supported thresholds as sensitivity analyses. Do not treat threshold replicates as different animals or pick a threshold by the resulting SA/P value. Missing coordinate/entry exports may make some sensitivity thresholds unavailable.

## D04 — Event construction, repeat entries, and time semantics

**Source:** SRC-OFT/SRC-YM fix the landmarks, first valid detection, and `>=1.0 s` dwell, but do not fully define detection confidence, tolerated tracking gaps, entry/re-entry boundaries, timestamp sampling, or whether a qualified entry is timestamped at initial crossing or completion of its dwell.

**Decision:** recover the existing acquisition/analysis conventions or register them. Define missing-data and interval semantics, threshold-crossing geometry, and what constitutes a new entry after leaving/re-entering the same arm. Do not compress repeated labels merely to simplify scoring. Do not replace head-centre AND tail-base crossing with centroid crossing in Y-maze.

**Affects:** new reconstruction from tracking and validation of upstream events; already exported metrics can retain an explicit unverified-upstream status.

## D05 — OFT incomplete sessions, occupancy, and latency

**Source:** SRC-OFT uses 300 s and `center_time_s / 300 * 100`. It does not define shortened/overlong recordings, missing tracked intervals, whether sub-second centre visits count toward occupancy despite not qualifying as entries, animals starting in the centre, boundary membership, or latency when no valid centre entry occurs. Periphery percentage has no explicit denominator formula.

**Decision:** define these policies and explain them in the run configuration. Keep the source centre-percentage denominator of 300; never silently replace it with valid tracked time. Do not assume periphery time is `300 - centre_time` when time is unobserved. Do not assign a no-entry latency of 300, zero, or missing without identifying the chosen convention; censoring has consequences for inference.

## D06 — Meaning of the NORT 15-second quality flag

**Source:** SRC-NORT specifies the name `passes_15s_exploration_threshold`, but not the exact comparison, training/test phase, total vs per-object quantity, or exclusion consequence.

**Decision:** register what is compared with 15 s and whether equality passes. Preserve a supplied upstream flag with its definition/provenance; otherwise leave it unresolved rather than default to a guessed interpretation. A failed quality flag is not automatically an exclusion.

## D07 — NORT candidate bouts and merged-gap accounting

**Source:** SRC-NORT defines distance/contact/orientation criteria, bout-level median angle, merging gaps `<0.5 s` before duration filtering, duration `>=1.0 s`, and median speed `<0.05 m/s`. It does not completely specify how orientation/contact creates candidate bouts while protecting brief frame-level failures, how contact interacts with the angle rule, whether merged gaps contribute to exploration duration or medians, how to resolve simultaneous object-zone candidates, or how “clearly object-directed” exceptions are annotated.

**Decision:** recover or register candidate construction, merger accounting, median sample selection, object-overlap arbitration, and annotation rules. Keep rule order and inequalities fixed. Do not merge across different objects. That last constraint is an engineering protection for object-specific times, not a separate stated rule in the source. Preserve raw segments and accepted/rejected/merged lineage.

**Affects:** reconstruction from frames or candidate bouts. Threshold-only golden tests do not resolve these accounting choices.

## D08 — NORT training side/location bias

**Source:** SRC-NORT gives `object_bias = (TA - TB)/(TA + TB)` and `side_bias = (TB - TA)/(TA + TB)`. Under those formulas `side_bias = -object_bias`. It also requests counterbalancing and side/location-bias QC, but supplies no object-to-location mapping.

**Decision:** preserve both source formulas exactly. Require per-session object identity/role/location/configuration metadata before making a spatial interpretation. A separately derived left/right or location-bias metric is an extension and needs its own name, definition, and provenance; do not overwrite the source `side_bias`.

## D09 — Statistical inference and reporting choices

**Source:** SRC-OFT/SRC-YM/SRC-NORT provide formulas and reporting aims, but do not fully specify model families/likelihoods, covariance estimators, estimands/weighting, factor coding, test tails, confidence-interval method/level in every use, multiplicity method and families, or the exact mixed-model formula. A Y-maze post-only formula is not explicitly stated. NORT specifies `and/or` for the within-group checks; the main group contrast remains essential.

**Decision:** reuse a documented prespecified plan or register the missing settings before confirmatory inference. Choose the OFT baseline strategy, Y-maze post-only specification when applicable, NORT within-group test(s), adjustment families, and whether summaries use SEM or 95% CI. Additional NORT covariates require predefinition. Record how baseline/covariate adjustment enters the planned contrast and what happens when cells are missing or a model is not estimable.

**Affects:** inferential outputs and some uncertainty displays, not deterministic metric arithmetic. A synthetic test configuration may use explicit test-only settings.

## D10 — Metadata, group mapping, and pairing

**Source:** SRC-GEN names metadata fields and defines G5/G6/G1/G2 for the main contrast, but supplies no actual metadata or complete mapping for every group. The phrase `age/treatment window` does not settle whether these are one or separate fields. NORT familiarization/test phases must not be confused with treatment pre/post timepoints.

**Decision:** explicitly map source columns, units, factor levels, animal/session IDs, treatment timepoints, and object configurations. Validate the contrast's four known roles. Do not invent missing group assignments or pair animals by row order.

## D11 — Behavioural secondary measures and locomotor definitions

**Source:** SRC-OFT/SRC-YM name freezing, grooming, rearing, immobility, distance, mean speed, arm preference, and optional fecal-boli count, but do not give every operational definition. In particular, freezing and immobility are not defined as interchangeable. Speed averaging and the exact arm-preference representation are unstated.

**Decision:** use documented upstream measurements/classifiers with their definitions, or register new operational definitions. Do not fabricate a speed threshold for freezing, a grooming/rearing detector, an arm-preference formula, or an inferred fecal-boli count. Label unavailable secondary measures as unavailable. Mean-speed handling depends on the tracking/time policy.

## D12 — Exclusions, missingness, and subgroup analyses

**Source:** SRC-GEN includes `excluded` and `exclusion_reason`; SRC-YM defines low-quality flags; SRC-NORT names an exploration-quality flag. The recipe does not state that either flag mandates exclusion, nor does it fully define missing-pair treatment or subgroup inferential plans.

**Decision:** keep flags separate from exclusions and state the approved inclusion/missingness policy for every endpoint/model. Export a per-animal accounting trail. Sex/age/treatment-window supplementary analyses should not become unregistered confirmatory discoveries. Always report the actual animal n used.

## D13 — Zero training exploration and optional pseudocount

**Source:** SRC-NORT explicitly makes DI/RI/percent-novel missing when `TN + TF = 0`, and LR missing if either component is zero unless a predefined pseudocount is used. It does not explicitly state the training-bias result at `TA + TB = 0`, or specify a pseudocount value/formula.

**Engineering safeguard:** return missing training biases with a zero-denominator reason rather than divide by zero; identify this boundary handling as an implementation extension. With no predefined pseudocount, the recipe's LR-missing rule applies. Do not silently smooth zero values. Enabling a pseudocount requires a declared value, units, and transformation formula, not merely a Boolean switch.

## D14 — NORT session timing and protocol availability

**Source:** SRC-NORT describes 6 min familiarization, a 5 h delay, and a 6 min test, but does not prescribe the first-valid-detection origin used explicitly for OFT/Y-maze or a policy for departures from this protocol.

**Decision:** preserve actual protocol/timing metadata and register NORT window alignment and incomplete-session handling. Do not automatically copy OFT's time origin, force missing sessions into existence, or treat the stated protocol as proof that the observed data followed it.
