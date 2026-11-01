//! Multi-parameter bout selection (design: docs/src/pipeline/multi-parameter-runs.md, 4.2/4.4).
//!
//! Everything here reproduces [`select_bouts_with_gap_bound`] exactly. Durations
//! are always taken from the caller's `duration(start, end)` closure (the
//! `BoutTiming` clock); this module does no duration arithmetic of its own.
//! Gap density between maximal true runs is exactly 0 in the legacy code, so a
//! positive `gap_density` disables bridging and is encoded as a constant.
//!
//! [`select_bouts_with_gap_bound`]: crate::temporal::select_bouts_with_gap_bound
use std::collections::BTreeMap;
use std::ops::Range;

use crate::temporal::DurationComparison;

/// `select_bouts_with_gap_bound` acceptance rule.
#[inline]
fn accepts(duration: f64, minimum: f64, comparison: DurationComparison) -> bool {
    minimum == 0.0 || comparison.accepts(duration, minimum)
}

/// Legacy bridging rule for one gap.
#[inline]
fn bridges(elapsed: f64, rule: &BoutRule) -> bool {
    rule.maximum_gap > 0.0
        && (if rule.strict_gap {
            elapsed < rule.maximum_gap
        } else {
            elapsed <= rule.maximum_gap
        })
        && 0.0 >= rule.gap_density
}

/// Merge/duration parameters of one variant.
#[derive(Debug, Clone, Copy)]
pub struct BoutRule {
    pub minimum: f64,
    pub comparison: DurationComparison,
    pub maximum_gap: f64,
    pub gap_density: f64,
    pub strict_gap: bool,
}

/// Accepted (post-duration) merged spans of a mask, from its runs. `O(B)`.
/// Identical to the true spans of `select_bouts_with_gap_bound(..).mask`.
pub fn select_runs(
    runs: &[Range<usize>],
    duration: impl Fn(usize, usize) -> f64,
    rule: BoutRule,
) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut current: Option<Range<usize>> = None;
    for run in runs {
        match current.take() {
            Some(prev) if bridges(duration(prev.end, run.start), &rule) => {
                current = Some(prev.start..run.end);
            }
            Some(prev) => {
                if accepts(
                    duration(prev.start, prev.end),
                    rule.minimum,
                    rule.comparison,
                ) {
                    out.push(prev);
                }
                current = Some(run.clone());
            }
            None => current = Some(run.clone()),
        }
    }
    if let Some(last) = current
        && accepts(
            duration(last.start, last.end),
            rule.minimum,
            rule.comparison,
        )
    {
        out.push(last);
    }
    out
}

/// First sample whose ending boundary qualifies the span (legacy
/// `qualification_index`), by binary search: duration is monotone in `end`.
pub fn qualification_index(
    span: &Range<usize>,
    duration: impl Fn(usize, usize) -> f64,
    minimum: f64,
    comparison: DurationComparison,
) -> Option<usize> {
    let (mut lo, mut hi) = (span.start, span.end);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if minimum == 0.0 || comparison.accepts(duration(span.start, mid + 1), minimum) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    (lo < span.end).then_some(lo)
}

pub fn spans_to_mask(spans: &[Range<usize>], len: usize) -> Vec<bool> {
    let mut mask = vec![false; len];
    for span in spans {
        mask[span.clone()].fill(true);
    }
    mask
}

/// Merged spans of a growing activation set, with running aggregates over the
/// spans that satisfy the minimum-duration rule.
pub struct Sweep<D: Fn(usize, usize) -> f64> {
    duration: D,
    rule: BoutRule,
    active: Vec<bool>,
    /// `lo -> hi` of disjoint merged spans (`hi` is the end of the last raw run).
    spans: BTreeMap<usize, usize>,
    accepted_count: usize,
    accepted_samples: usize,
}

impl<D: Fn(usize, usize) -> f64> Sweep<D> {
    pub fn new(n: usize, duration: D, rule: BoutRule) -> Self {
        Self {
            duration,
            rule,
            active: vec![false; n],
            spans: BTreeMap::new(),
            accepted_count: 0,
            accepted_samples: 0,
        }
    }

    pub fn is_active(&self, i: usize) -> bool {
        self.active[i]
    }

    fn accepted(&self, lo: usize, hi: usize) -> bool {
        accepts(
            (self.duration)(lo, hi),
            self.rule.minimum,
            self.rule.comparison,
        )
    }

    fn add(&mut self, lo: usize, hi: usize) {
        if self.accepted(lo, hi) {
            self.accepted_count += 1;
            self.accepted_samples += hi - lo;
        }
        self.spans.insert(lo, hi);
    }

    fn remove(&mut self, lo: usize) {
        let hi = self.spans.remove(&lo).expect("span exists");
        if self.accepted(lo, hi) {
            self.accepted_count -= 1;
            self.accepted_samples -= hi - lo;
        }
    }

    /// Activate frame `i`. Idempotent. `O(log B)` plus the duration calls.
    pub fn activate(&mut self, i: usize) {
        if std::mem::replace(&mut self.active[i], true) {
            return;
        }
        let pred = self.spans.range(..=i).next_back().map(|(&l, &h)| (l, h));
        if let Some((_, hi)) = pred
            && hi > i
        {
            return; // inside a bridged span: gaps only shrink, extents unchanged
        }
        let (mut lo, mut hi) = (i, i + 1);
        if let Some((pl, ph)) = pred
            && (ph == i || bridges((self.duration)(ph, i), &self.rule))
        {
            self.remove(pl);
            lo = pl;
        }
        let succ = self.spans.range(i + 1..).next().map(|(&l, &h)| (l, h));
        if let Some((sl, sh)) = succ
            && (sl == i + 1 || bridges((self.duration)(i + 1, sl), &self.rule))
        {
            self.remove(sl);
            hi = sh;
        }
        self.add(lo, hi);
    }

    pub fn accepted_count(&self) -> usize {
        self.accepted_count
    }

    /// Samples covered by accepted spans, bridged gaps included.
    pub fn accepted_samples(&self) -> usize {
        self.accepted_samples
    }

    /// Accepted spans in order. `O(B)`.
    pub fn accepted_spans(&self) -> Vec<Range<usize>> {
        self.spans
            .iter()
            .filter(|&(&l, &h)| self.accepted(l, h))
            .map(|(&l, &h)| l..h)
            .collect()
    }
}

/// One threshold's result of [`sweep_family`].
#[derive(Debug, Clone, PartialEq)]
pub struct SweepSnapshot {
    pub accepted_count: usize,
    pub accepted_samples: usize,
    pub spans: Vec<Range<usize>>,
}

/// Float-safety band half-width (design 4.1). Generous on purpose: a wider
/// band only costs extra legacy evaluations, never correctness.
pub fn band_eps(critical: f64, threshold: f64) -> f64 {
    1e-9 * (critical.abs() + threshold.abs()) + 1e-12
}

/// Frame order by critical value; non-finite values never activate.
pub fn critical_order(critical: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..critical.len())
        .filter(|&i| critical[i].is_finite())
        .collect();
    order.sort_by(|&a, &b| critical[a].total_cmp(&critical[b]));
    order
}

/// `(certain, band)`: `order[..certain]` are active at `threshold` with
/// certainty, `order[band]` lie within the float-safety band and must be
/// decided by the legacy scalar predicate; the rest are inactive.
pub fn band_split(critical: &[f64], order: &[usize], threshold: f64) -> (usize, Range<usize>) {
    let lo = order.partition_point(|&i| critical[i] < threshold - band_eps(critical[i], threshold));
    let hi =
        order.partition_point(|&i| critical[i] <= threshold + band_eps(critical[i], threshold));
    (lo, lo..hi.max(lo))
}

fn certain(critical: f64, threshold: f64) -> bool {
    critical < threshold - band_eps(critical, threshold)
}

/// Temporal sweep over a threshold family: frame `i` is active at
/// `thresholds[k]` iff the legacy predicate `legacy(i, k)` holds, which the
/// kernel's `critical[i] <= thresholds[k]` reproduces outside the float band.
/// Results are in the caller's threshold order. Cost
/// `O(n log n + sum_k (delta_k + band_k + B_k))`; a threshold is rebuilt from
/// scratch only when the legacy predicate is non-monotone inside the band.
pub fn sweep_family(
    critical: &[f64],
    thresholds: &[f64],
    legacy: impl Fn(usize, usize) -> bool,
    duration: impl Fn(usize, usize) -> f64 + Copy,
    rule: BoutRule,
    want_spans: bool,
) -> Vec<SweepSnapshot> {
    let n = critical.len();
    let order = critical_order(critical);
    let mut ks: Vec<usize> = (0..thresholds.len()).collect();
    ks.sort_by(|&a, &b| thresholds[a].total_cmp(&thresholds[b]));
    let mut out: Vec<Option<SweepSnapshot>> = vec![None; thresholds.len()];
    let mut sweep = Sweep::new(n, duration, rule);
    let mut done = 0usize; // order[..done] are active as certain frames
    let mut temp: Vec<usize> = Vec::new(); // band frames active beyond `done`
    for k in ks {
        let t = thresholds[k];
        let (prefix, band) = band_split(critical, &order, t);
        let band_true: Vec<usize> = order[band]
            .iter()
            .copied()
            .filter(|&i| legacy(i, k))
            .collect();
        let kept = |i: usize| certain(critical[i], t) || band_true.contains(&i);
        if temp.iter().any(|&i| !kept(i)) {
            sweep = Sweep::new(n, duration, rule);
            done = 0;
            temp.clear();
        }
        for &i in &order[done..prefix.max(done)] {
            sweep.activate(i);
        }
        done = done.max(prefix);
        temp.retain(|&i| !certain(critical[i], t));
        for &i in &band_true {
            if !sweep.is_active(i) {
                sweep.activate(i);
                temp.push(i);
            }
        }
        out[k] = Some(SweepSnapshot {
            accepted_count: sweep.accepted_count(),
            accepted_samples: sweep.accepted_samples(),
            spans: if want_spans {
                sweep.accepted_spans()
            } else {
                Vec::new()
            },
        });
    }
    out.into_iter()
        .map(|s| s.expect("every threshold visited"))
        .collect()
}

// ---------------------------------------------------------------------------
// Special cases (design 4.2, "cheaper special cases")
// ---------------------------------------------------------------------------

/// Accepted count and covered samples of one variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totals {
    pub count: usize,
    pub samples: usize,
}

struct Fenwick(Vec<i64>);

impl Fenwick {
    fn new(n: usize) -> Self {
        Self(vec![0; n + 1])
    }
    fn add(&mut self, mut i: usize, v: i64) {
        i += 1;
        while i < self.0.len() {
            self.0[i] += v;
            i += i & i.wrapping_neg();
        }
    }
    /// Sum over `[0, i)`.
    fn prefix(&self, mut i: usize) -> i64 {
        let mut s = 0;
        while i > 0 {
            s += self.0[i];
            i -= i & i.wrapping_neg();
        }
        s
    }
}

/// Min-duration only (OFT dwell): sorted run durations + prefix sums,
/// `O(B log B)` once, `O(log B)` per variant.
pub fn min_duration_totals(
    runs: &[Range<usize>],
    duration: impl Fn(usize, usize) -> f64,
    variants: &[(f64, DurationComparison)],
) -> Vec<Totals> {
    let mut by_duration: Vec<(f64, usize)> = runs
        .iter()
        .map(|r| (duration(r.start, r.end), r.len()))
        .collect();
    by_duration.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut suffix_samples = vec![0usize; by_duration.len() + 1];
    for i in (0..by_duration.len()).rev() {
        suffix_samples[i] = suffix_samples[i + 1] + by_duration[i].1;
    }
    variants
        .iter()
        .map(|&(minimum, comparison)| {
            let first = by_duration.partition_point(|d| !accepts(d.0, minimum, comparison));
            Totals {
                count: by_duration.len() - first,
                samples: suffix_samples[first],
            }
        })
        .collect()
}

/// Fixed mask, K `(gap, min-duration)` variants: sort the `B-1` gaps, merge
/// runs with union-find in that order, and read each variant's accepted
/// components from a Fenwick tree over compressed component durations.
/// `O(B log B + K log B)`.
pub fn gap_duration_totals(
    runs: &[Range<usize>],
    duration: impl Fn(usize, usize) -> f64,
    variants: &[BoutRule],
) -> Vec<Totals> {
    let b = runs.len();
    if b == 0 {
        return vec![
            Totals {
                count: 0,
                samples: 0
            };
            variants.len()
        ];
    }
    let gaps: Vec<f64> = (0..b - 1)
        .map(|j| duration(runs[j].end, runs[j + 1].start))
        .collect();
    let mut gap_order: Vec<usize> = (0..b - 1).collect();
    gap_order.sort_by(|&a, &c| gaps[a].total_cmp(&gaps[c]));
    // Pass 1: component durations in merge order; component = run interval [a, z].
    let mut left: Vec<usize> = (0..b).collect(); // root -> first run
    let mut right: Vec<usize> = (0..b).collect(); // root -> last run
    let mut parent: Vec<usize> = (0..b).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    struct Merge {
        removed: [(f64, usize); 2],
        added: (f64, usize),
    }
    let single = |j: usize| (duration(runs[j].start, runs[j].end), runs[j].len());
    let mut merges: Vec<Merge> = Vec::with_capacity(b.saturating_sub(1));
    for &j in &gap_order {
        let (ra, rb) = (find(&mut parent, j), find(&mut parent, j + 1));
        let (a, z) = (left[ra], right[rb]);
        let comp = |l: usize, r: usize| {
            (
                duration(runs[l].start, runs[r].end),
                runs[r].end - runs[l].start,
            )
        };
        let removed = [comp(left[ra], right[ra]), comp(left[rb], right[rb])];
        let added = comp(a, z);
        parent[rb] = ra;
        left[ra] = a;
        right[ra] = z;
        merges.push(Merge { removed, added });
    }
    // Compress every duration that can appear.
    let mut all: Vec<f64> = (0..b).map(|j| single(j).0).collect();
    all.extend(merges.iter().map(|m| m.added.0));
    all.sort_by(f64::total_cmp);
    all.dedup_by(|a, c| a.total_cmp(c).is_eq());
    let slot = |d: f64| {
        all.binary_search_by(|x| x.total_cmp(&d))
            .expect("compressed")
    };
    let mut count = Fenwick::new(all.len());
    let mut samples = Fenwick::new(all.len());
    for j in 0..b {
        let (d, s) = single(j);
        count.add(slot(d), 1);
        samples.add(slot(d), s as i64);
    }
    // Pass 2: answer variants in ascending bridged-prefix order.
    let mut order: Vec<usize> = (0..variants.len()).collect();
    let prefix_of = |rule: &BoutRule| {
        if rule.gap_density > 0.0 {
            return 0; // positive density: no bridging at all
        }
        gap_order.partition_point(|&j| {
            rule.maximum_gap > 0.0
                && if rule.strict_gap {
                    gaps[j] < rule.maximum_gap
                } else {
                    gaps[j] <= rule.maximum_gap
                }
        })
    };
    order.sort_by_key(|&v| prefix_of(&variants[v]));
    let mut out = vec![
        Totals {
            count: 0,
            samples: 0
        };
        variants.len()
    ];
    let mut applied = 0usize;
    for v in order {
        let rule = &variants[v];
        let target = prefix_of(rule);
        for m in &merges[applied..target] {
            for (d, s) in m.removed {
                count.add(slot(d), -1);
                samples.add(slot(d), -(s as i64));
            }
            count.add(slot(m.added.0), 1);
            samples.add(slot(m.added.0), m.added.1 as i64);
        }
        applied = applied.max(target);
        let first = all.partition_point(|&d| !accepts(d, rule.minimum, rule.comparison));
        out[v] = Totals {
            count: (count.prefix(all.len()) - count.prefix(first)) as usize,
            samples: (samples.prefix(all.len()) - samples.prefix(first)) as usize,
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Bitset algebra (design 4.4)
// ---------------------------------------------------------------------------

/// Fixed-length bitset; word-wise AND/OR for multi-parameter variant masks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    pub fn from_bools(values: &[bool]) -> Self {
        let mut set = Self::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            if v {
                set.set(i);
            }
        }
        set
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    pub fn get(&self, i: usize) -> bool {
        self.words[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn and(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a & b)
    }

    pub fn or(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a | b)
    }

    pub fn and_not(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a & !b)
    }

    fn zip(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        assert_eq!(self.len, other.len);
        Self {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(&a, &b)| f(a, b))
                .collect(),
            len: self.len,
        }
    }

    pub fn not(&self) -> Self {
        let mut out = Self {
            words: self.words.iter().map(|w| !w).collect(),
            len: self.len,
        };
        if self.len % 64 != 0
            && let Some(last) = out.words.last_mut()
        {
            *last &= (1u64 << (self.len % 64)) - 1;
        }
        out
    }

    pub fn to_bools(&self) -> Vec<bool> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    /// Next index `>= from` whose bit equals `want`, or `len` if none.
    fn next_bit(&self, from: usize, want: bool) -> usize {
        let mut i = from;
        while i < self.len {
            let word = self.words[i / 64];
            let word = (if want { word } else { !word }) >> (i % 64);
            if word != 0 {
                return (i + word.trailing_zeros() as usize).min(self.len);
            }
            i = (i / 64 + 1) * 64;
        }
        self.len
    }

    /// Maximal runs of set bits, `O(n/64 + B)` via `trailing_zeros`.
    pub fn runs(&self) -> Vec<Range<usize>> {
        let mut runs = Vec::new();
        let mut i = 0;
        loop {
            let start = self.next_bit(i, true);
            if start >= self.len {
                return runs;
            }
            i = self.next_bit(start, false);
            runs.push(start..i);
        }
    }
}

/// `bit(t) = [critical <= t]` for every threshold, built incrementally over the
/// sorted order (`O(n/64 + delta)` per threshold) and corrected inside the
/// float band by the legacy scalar predicate `legacy(frame, k)`.
pub fn threshold_bitsets(
    critical: &[f64],
    thresholds: &[f64],
    legacy: impl Fn(usize, usize) -> bool,
) -> Vec<BitSet> {
    let order = critical_order(critical);
    let mut ks: Vec<usize> = (0..thresholds.len()).collect();
    ks.sort_by(|&a, &b| thresholds[a].total_cmp(&thresholds[b]));
    let mut cur = BitSet::new(critical.len());
    let mut done = 0usize;
    let mut out: Vec<Option<BitSet>> = vec![None; thresholds.len()];
    for k in ks {
        let (prefix, band) = band_split(critical, &order, thresholds[k]);
        for &i in &order[done..prefix.max(done)] {
            cur.set(i);
        }
        done = done.max(prefix);
        let mut bits = cur.clone();
        for &i in &order[band] {
            if legacy(i, k) {
                bits.set(i);
            }
        }
        out[k] = Some(bits);
    }
    out.into_iter().map(|b| b.expect("visited")).collect()
}
