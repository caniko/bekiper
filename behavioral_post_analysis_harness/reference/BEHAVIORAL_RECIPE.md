# Source reference: behavioural post-analysis recipe

> **Conversion metadata — not part of the source recipe.**
> Original file: `behavioral_analysis_recipe_OFT_Ymaze_NORT(1).docx`.
> Original SHA-256: `902b572ad21f61718047d00bec853abf17c9471307f70517d0ef8339bdb6e654`.
> This is a reading-order Markdown transcription of all non-empty body paragraphs and all three tables. Source wording, metric names, formulas, inequalities, and contradictory definitions are retained. Page headers and pagination are omitted; stable section IDs are added outside the source text.
> The recipe refers to an analysis-criteria document, a thesis draft, and literature. Those underlying materials were not supplied separately and are not incorporated or verified here.
> Implementation choices and unresolved issues are in `DECISION_REGISTER.md`, not silently incorporated into this transcript.

---

## Authoritative amendment YM-2026-10-04 (researcher decision)

This amendment supersedes the original SRC-YM dwell statement **only for the
three principal 25%, 45%, and 65% analyses**. The original transcription below
is retained as historical source provenance, not simultaneous normative guidance.

**A valid arm entry is registered once both head center and tail base have
crossed the selected 25%, 45%, or 65% threshold into the same arm. No minimum
dwell duration is required for these three primary analyses.** The false-to-true
predicate transition occurs at the second required landmark crossing. A genuine
single-frame/sub-second entry counts. No debounce, pose-fill, gap-bridging or
exit latch changes the principal predicate transitions.

Reproduce the Imen geometry as `imen_radial_vertex_span_v1`: let `c` be the
arithmetic mean of the canonical center-polygon vertices. For each arm's
canonical vertices `v`, compute `r_min = min(||v-c||)` and
`r_max = max(||v-c||)`. At fraction `p`, the radial boundary is
`r_p = r_min + p*(r_max-r_min)`. Head center and tail base must each be inside
that same arm polygon (existing ray-casting convention) and have squared
Euclidean distance from `c` **>= r_p²**. This boundary is a circular arc clipped
to the arm polygon, not an axial line, an area-centroid construction, or
`p*r_max`. No additional radial tolerance is introduced. The generic "from maze
centre" geometry wording and original >=1 s dwell are superseded/conflicting
provenance for these analyses; existing coordinates are not reinterpreted.

This decision does not resolve the table/prose triad conflict, select the
dataset's primary threshold, or prescribe statistical inference. Historical
results need geometry, landmark, dwell, mitigation and code/config evidence;
absence of dwell is correct and is not itself a verification blocker.

---

# Behavioural Analysis Recipe: OFT, Y-maze and NORT

Short reproducible workflow for PhD behavioural data analysis

Basis: uploaded analysis-criteria document, PhD thesis draft, and standard behavioural-test literature.

<a id="src-general"></a>

<!-- Source section ID: SRC-GEN -->

## General setup for all tests

- Create one metadata table with one row per animal per test/session: animal_id, sex, age/treatment window, 5XFAD status, HCAR1 status, treatment, group, timepoint, excluded, and exclusion_reason.

- Use the experimental factors: 5XFAD status x HCAR1 status x treatment. Keep pre/post as timepoint when available.

- Main biological contrast: whether the lactate effect differs between HCAR1-WT and HCAR1-KO mice in the 5XFAD background.

```text
(G5 AD-HCAR1-WT lactate - G6 AD-HCAR1-WT vehicle) -
(G1 AD-HCAR1-KO lactate - G2 AD-HCAR1-KO vehicle)
```

- Show individual animals in all graphs. Report n, mean +/- SEM or 95% CI, exact P values, and planned contrasts.

<a id="src-oft"></a>

<!-- Source section ID: SRC-OFT -->

## 1. Open field test (OFT)

- Purpose: locomotor activity and anxiety-like centre avoidance; use distance/speed as locomotor controls and centre/periphery measures as anxiety-like/thigmotaxis readouts.

- Recording: 5 min. Arena 40 x 40 x 34 cm; periphery = outer 8 cm border; centre = inner 24 x 24 cm square.

- Valid zone entry: body centroid enters a zone and remains there for >=1.0 s. Start from first valid animal detection; expected analysis duration = 300 s.

| Step | What to calculate |
| --- | --- |
| Primary outputs | total_distance_m; mean_speed_m_per_s; center_time_s; center_time_pct = center_time_s / 300 x 100; center_entries; latency_to_first_center_entry_s |
| Secondary outputs | periphery_time_pct; freezing_s; grooming_s; rearing_s; immobility_s; fecal_boli_count if available |
| Interpretation | Low center time, fewer center entries and longer latency indicate stronger centre avoidance, but interpret together with distance/speed. |

### Statistics and graphs

- Post-treatment model: outcome ~ 5XFAD_status * HCAR1_status * treatment.

- If pre/post exists, use baseline-adjusted model: post_outcome ~ pre_outcome + 5XFAD_status * HCAR1_status * treatment, or a mixed model with animal_id as random effect.

- Main figure panels: total distance, mean speed, center time %, center entries, latency to center.

<a id="src-ymaze"></a>

<!-- Source section ID: SRC-YM -->

## 2. Y-maze

- Purpose: spatial working-memory readout using spontaneous alternation from overlapping triplets of valid arm entries.

- Recording: 6 min. Start from first valid detection; expected analysis duration approximately 360 s.

- Use one predefined threshold per dataset: 25%, 45%, or 65%. Prefer the selected thesis threshold for the main analysis and use others as sensitivity analyses. The original "of arm length from the maze centre" geometry wording is superseded for reproducing Imen by `imen_radial_vertex_span_v1` in YM-2026-10-04.

> Superseded for principal 25/45/65% by YM-2026-10-04; original source wording:
> Valid arm entry: head centre and tail base both cross the selected threshold into an arm and the animal remains in the arm for >=1.0 s.

- Keep raw entries for debug outputs, but use valid_arm_entry_sequence for spontaneous-alternation scoring.

| Triad type | Definition |
| --- | --- |
| Correct | Three different arms in a sliding window of three entries: ABC, ACB, BAC, BCA, CAB, CBA. |
| Incorrect | Current arm repeats one of the two previously visited different arms: AAB, AAC, ABA, ACA, BAB, BBA, BBC, BCB, CAC, CBC, CCA, CCB. |
| Neutral | First two entries are the same, so alternation cannot be evaluated: AAA, ABB, ACC, BAA, BBB, BCC, CAA, CBB, CCC. |

### Calculations

```text
total_triads = valid_total_arm_entries - 2
usable_triads = correct_alternations + incorrect_alternations
spontaneous_alternation_pct = correct_alternations / usable_triads x 100
spontaneous_alternation_pct_standard = correct_alternations / total_triads x 100
neutral_triads_pct = neutral_triads / total_triads x 100
```

- Primary thesis-compatible SA: spontaneous_alternation_pct, where neutral triads are excluded from the denominator.

- Sensitivity/literature-standard SA: spontaneous_alternation_pct_standard, where neutral triads are included via total_triads.

- Report NA if usable_triads = 0 for the primary formula, or if total_triads = 0 for standard SA and neutral_triads_pct.

- QC outputs: valid_total_arm_entries, raw_total_arm_entries, removed_short_entries, neutral_triads_pct, usable_triads, arm_preference, distance, mean speed, freezing.

- Flag low-quality sessions: valid_total_arm_entries < 8 or usable_triads < 5.

### Statistics and graphs

- Primary endpoint: post-treatment SA. If pre/post exists, use post_SA ~ pre_SA + 5XFAD_status * HCAR1_status * treatment.

- Also calculate DeltaSA = post_SA - pre_SA for visualization and secondary analysis.

- Use Spearman QC correlations: SA vs valid_total_arm_entries, total_distance_m, mean_speed_m_per_s, and neutral_triads_pct.

- Main figure panels: post-treatment SA, DeltaSA, paired pre/post SA, valid arm entries, distance/speed, neutral triads %.

- Supplementary: 25/45/65 threshold sensitivity and primary SA vs standard SA.

<a id="src-nort"></a>

**Researcher instruction, 2026-10-07, reaffirmed 2026-10-13:** the current
Neo/Imen NORT policy is
minimum bout duration **≥0.5 s after merging**, same-object gap merging
**≤1.0 s**, and **disabled speed exclusion**. Exact endpoints are inclusive;
speed remains descriptive. This amendment supersedes the earlier bout/gap/speed
criteria transcribed below and the 2026-10-12 recipe selection. Retain historical
outputs with their recorded rules and producer identities; they are not
current-policy rescoring evidence.

<!-- Source section ID: SRC-NORT -->

## 3. Novel object recognition test (NORT / NOR)

- Purpose: recognition-memory readout. Always report raw exploration times together with DI/RI because object preference can be influenced by exploration level, object salience, neophobia and location bias.

- Protocol: habituation to empty arena, familiarization with two identical objects for 6 min, then test after 5 h with one familiar and one novel object for 6 min. Counterbalance object positions/configurations.

| Criterion | Rule |
| --- | --- |
| Object zone | Use object mask/convex hull; interaction zone extends 3 cm from object perimeter. |
| Candidate exploration | Body centroid outside object mask; nose <=3 cm from object perimeter; head oriented toward object or nose contact. |
| Head angle | Bout-level median head-orientation angle <=45 degrees; do not remove good bouts for brief frame-level failures. |
| Bout merging | Merge candidate bouts separated by gaps/distractions <0.5 s before applying duration filter. |
| Valid bout | After merging, duration >=1.0 s and bout-level median nose speed <0.05 m/s. |
| Exclusions | Do not count climbing/sitting on the object, grooming, or rearing unless clearly object-directed. |

### Calculations

```text
Training:
TA = exploration time Object A
TB = exploration time Object B
total_training_exploration = TA + TB
object_bias = (TA - TB) / (TA + TB)
side_bias = (TB - TA) / (TA + TB)
Test:
TN = novel object exploration time
TF = familiar object exploration time
total_test_exploration = TN + TF
DI = (TN - TF) / (TN + TF)
RI = TN / (TN + TF)
percent_novel = 100 x RI
LR = ln(TN / TF)
```

- If TN + TF = 0, report DI, RI and percent_novel as NA. If TN = 0 or TF = 0, report LR as NA unless a predefined pseudocount is used.

- Use passes_15s_exploration_threshold as the main exploration-quality flag.

### Statistics and graphs

- Primary endpoint: DI.

- Model: DI ~ 5XFAD_status * HCAR1_status * treatment.

- Add sex, age/treatment_window, total_test_exploration and object_configuration only if predefined.

- Within each group, test DI vs 0 and/or percent_novel vs 50 as memory/preference checks, with multiple-comparison correction.

- Main figure panels: training Object A vs Object B exploration, test familiar vs novel exploration, DI by group with dashed line at 0, percent novel by group with dashed line at 50%, and total test exploration.

- QC/supplementary: object bias, side/location bias, DI vs total exploration, number of valid bouts, and bouts removed by duration/speed/angle filters.

<a id="src-figures"></a>

<!-- Source section ID: SRC-FIG -->

## Minimal final figure set

- Figure 1: OFT locomotion/anxiety-like behaviour.

- Figure 2: Y-maze spontaneous alternation plus locomotor/QC outputs.

- Figure 3: NORT recognition memory plus exploration QC.

- Supplementary: threshold sensitivity, sex/age/treatment-window subgroup analyses, and debug/QC outputs.

<a id="src-notes"></a>

<!-- Source section ID: SRC-NOTE -->

## Short reference notes

- Y-maze: spontaneous alternation is commonly based on overlapping triplets of arm entries; total possible alternations are often calculated as total entries minus two.

- OFT: centre/periphery behaviour should be interpreted together with total distance and speed because locomotion can confound anxiety-like readouts.

- NORT: report both raw exploration times and DI/RI, because novelty preference can be affected by object salience, neophobia, exploration level and location bias.
