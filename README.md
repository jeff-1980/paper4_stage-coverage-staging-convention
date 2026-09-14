# Data and code for "Stage-conditional coverage of RUL prediction intervals depends on the staging convention: evidence from C-MAPSS"

This repository corresponds to the **third version** of the note (Stage 0-R2:
three implementation fixes on top of the Stage 0-R clean, leak-free pipeline;
no retraining), with a **Stage 0-R3 patch** applied on top (one further, more
narrowly scoped fix — see below). See `legacy_leaked/` for why the *first*
version's numbers were withdrawn, and "Known retracted analyses" below for
two more recent, narrower retractions.

## Stage 0-R3 patch (post-Stage-0-R2 correction)

One further fix, scoped to Mondrian CP (split) only, applied on top of the
three Stage 0-R2 fixes below — no retraining, no other method touched.

- **Mondrian CP (split) calibration quantile, off-by-one fix**
  (`scripts/build_canonical.py::build_mondrian_split_quantiles`). The
  per-stage empirical quantile was computed as
  `np.quantile(scores, ceil((m+1)(1-alpha))/m, method='higher')` — an
  approximation of the standard split-CP quantile that is NOT exactly the
  k-th order statistic: `np.quantile`'s `'higher'` method scales its virtual
  index by `(m-1)`, not `m`, so dividing `k` by `m` first and rescaling
  inside `np.quantile` silently shifts the selected index by one for generic
  `m` (verified case: `m=397`, `k=⌈(m+1)(1-α)⌉=359` — the old call selected
  the 360th smallest calibration score, not the 359th; the correct value is
  the 359th order statistic, `1.500141`). Now computed directly:
  `k = ceil((m+1)(1-alpha))`; `q_hat = sorted(scores)[k-1]` if `k <= m` else
  `np.inf`.
  *Effect on conclusions:* rebuilding the canonical file
  (`per_sample_final_v3.csv`, replacing v2's `per_sample_final_v2.csv`)
  changes ONLY the `mf`/`covered_mf`/`is_inf_mf` columns (+ their `_h5`/`_h10`
  siblings) — every other column (`mc`, `sp`, `mo`, all `stage_*` labels) is
  bit-identical to v2, verified by a full column-by-column diff. Checking
  Mondrian CP (split)'s worst-stage cell in every table that reports one and
  includes this method (main table, train-set-percentile full-availability,
  Matched, and the reintroduced h-sensitivity table): **0 cells flip**.
  `table_h_sensitivity_v3` (k fixed, h ∈ {5,7,10}, 48 rows) is reintroduced in
  this patch — it existed only in the withdrawn v1 pipeline before now.

## v3 changelog (Stage 0-R2)

Three implementation fixes, applied together in one canonical rebuild
(`per_sample_final_v2.csv`, replacing v1's `per_sample_final.csv`; further
patched to `per_sample_final_v3.csv` above). None of them retrain the base
models — only inference-time and analysis-time code changed.

1. **MC-Dropout total variance formula** (`scripts/models/bayesian_lstm.py::predict_mc`).
   Was `sigma_total = sqrt(mean(sigma_b)^2 + Var(mu_b))` — squaring the mean
   instead of averaging the squares, which by Jensen's inequality
   systematically understated total variance. Now
   `sigma_total = sqrt(mean(sigma_b^2) + Var(mu_b))`.
   *Effect on conclusions:* combined with fix 2 below, regenerating the full
   192-cell main table (4 conventions × 4 methods × 3 stages × ... , finite-only
   ECR, h=7 primary) against v1 gives max absolute cell change 0.2223, mean
   absolute change 0.0257. Only 10 of 192 cells flip which stage is worst; 9
   of those 10 are Mondrian CP (online or split) cells, 1 is MC-Dropout under
   the Matched convention on FD001 (a near-exact tie, 0.8835 vs 0.9125 in v1
   flipped by the small CUSUM-occupancy shift feeding the Matched convention's
   cut points). **MC-Dropout and Split CP's headline findings are unchanged**:
   fixed-fraction middle-worst and CUSUM late-worst are each 8/8 across
   FD × method, before and after all three fixes.
2. **CUSUM baseline window and untriggered-instance convention**
   (`scripts/conformal/mondrian_cp.py::detect_cp_cusum`). The baseline window
   used to estimate a unit's own in-control mean/std was hardcoded to the
   first 10 scores (`scores[:10]`); it is now `max(floor(0.2*n), 20)` scores
   of that unit's own trajectory, so short units get a baseline that scales
   with their length instead of a fixed constant that could exceed 50% of a
   short unit's data. An instance whose CUSUM statistic never crosses the
   threshold now returns `cp = n` exactly (was `cp = n - 1`), so
   `assign_stage(t, n, n)` leaves that instance's LATE bucket genuinely empty,
   rather than containing one spurious leftover timestep.
   *Effect on conclusions:* changes `stage_cusum` labels for every method that
   reads them (all four), and changes Mondrian's own CUSUM-driven warm-up
   composition. Non-bucketed methods' conclusions are unaffected for the same
   reason as fix 1 (see above); this is the fix that most directly feeds the
   worst-stage flips concentrated in Mondrian CP.
3. **Mondrian CP (split) redefinition, replacing "frozen"**
   (`scripts/build_canonical.py::run_mondrian_split`, replacing the deleted
   `run_mondrian_frozen_kfixed`). The previous "frozen" variant was a
   predict-only `copy.deepcopy()` of the adaptive, exponentially-weighted
   online instance — still an adaptive machine underneath, just not fed
   eval-time feedback. The new "split" variant is genuinely non-adaptive:
   `build_mondrian_split_quantiles()` fills each of the 3 stage buckets ONCE
   from ALL calibration units' scores with uniform weight (no λ, no time
   index, no exponential decay), then takes the standard split-CP empirical
   quantile per stage (see the Stage 0-R3 patch above for the exact-quantile
   fix); `run_mondrian_split()` applies that one fixed per-stage quantile to
   every evaluation sample — no `.update()`, no shared mutable state, and
   evaluation-unit order is provably irrelevant (the function carries no
   state at all). Column names (`mf`, `covered_mf`, `width_mf`, `is_inf_mf`,
   ...) are unchanged from v1; what they mean is not.
   *Effect on conclusions:* this is the other main contributor (with fix 2)
   to the worst-stage flips concentrated in Mondrian CP cells; confirmed no
   code path in this repository still calls the removed
   `run_mondrian_frozen_kfixed`.

Two further v3 analyses (not comparisons to v1, new in this round; table
names below are the Stage 0-R3-patched `_v3` versions):

- **Cohort restriction** (instances/engines with ≥1 window in all 3 stages
  of a convention — see Design notes below for the exact definition):
  restricting the train-set-percentile convention to its 150/455-instance
  cohort flips 14 of 16 (fd, method) cells from early-worst to middle-worst —
  all 8/8 MC-Dropout/Split CP cells flip (matching an independent external
  recomputation exactly); only Mondrian CP (online) on FD001/FD002 stays
  early-worst. The same restriction applied to CUSUM (425/455 in cohort)
  leaves all 16 cells unchanged (0/16 flips), including Mondrian CP (online).
  Reported as two coexisting readings (full availability vs. matched
  population), not resolved in favor of either
  (`tables/table_trainpct_full_vs_cohort_v3.{csv,tex}`,
  `tables/table_cusum_cohort_check_v3.csv`).
- **Paired-difference CIs** via a unique-engine-block bootstrap (10,000
  resamples; see Design notes): 15 of 16 cells exclude zero. The one
  exception (FD001, MC-Dropout, CUSUM late−middle: mean −0.044, 95% CI
  [−0.0947, 0.0067]) is a genuine boundary result, not a bootstrap
  implementation error — it is the smallest-in-absolute-value mean
  difference of all 16 cells, on the smallest-N subset (`tables/paired_diff_v3.csv`).

## Directory overview

- `data/per_sample_final_v3.csv` — the canonical per-sample data source,
  current as of the Stage 0-R3 patch. One row per (fd, seed, engine, t): raw
  model outputs (mu, sigma, y_true, s_t), all four methods' interval bounds
  (MC-Dropout, Split CP, Mondrian CP online, Mondrian CP split — Mondrian's
  at h ∈ {5, 7, 10}, k=0.7 fixed), and four staging-convention labels
  (fixed-fraction, CUSUM, train-set-percentile, occupancy-matched).
  `data/per_sample_final_v3.md5` is its md5
  (`f164af8d4a76559f817269c32131ee4e`). Every table and figure under `tables/`
  and `figures/` with a `_v3` suffix is generated from this one file.
  `data/per_sample_final_v2.csv` (+ `.md5`) is **v2**, kept on disk,
  **superseded** — it predates the Stage 0-R3 patch above; only its
  `mf`/`covered_mf`/`is_inf_mf` columns (+ `_h5`/`_h10` siblings) differ from
  v3, everything else is bit-identical. `data/per_sample_final.csv` (+ `.md5`)
  is **v1**, also kept, also superseded — see the v3 changelog above for the
  three fixes it predates. Do not mix columns across v1/v2/v3 in one table.
- `data/per_unit_clocks_final_v3.csv`, `data/trainpct_thresholds_final_v3.csv`
  — secondary per-unit and per-(fd,seed) statistics produced alongside the
  v3 canonical file (RUL-clip knee position, CUSUM change-point position, the
  train-set 33rd/67th raw-cycle percentiles used by the train-set-percentile
  convention). These are unaffected by the Stage 0-R3 patch (it only touches
  Mondrian CP (split)'s calibration quantile) but are regenerated alongside
  for a consistent (fd,seed) build; `_v2`- and unsuffixed counterparts are
  the superseded v2/v1 versions, kept on disk.
- `models/` — 20 trained Bayesian LSTM checkpoints (`best_{FD}_seed{N}.pt`,
  4 subsets × 5 seeds) and their manifests (`meta_{FD}_seed{N}.json`: 4-way
  engine-level split hash, the engine list for each of the 4 partitions,
  scaler fit statistics and their source). **Unchanged since v1/v2/v3** — no
  fix in this repository's history retrains anything. Stage 2 training uses
  a fixed epoch cap of 100 (`train_clean.py::CFG['epochs_stage2']`, early-stop
  patience 15); stage 1 uses 50 epochs (patience 10).
- `scripts/` — `train_clean.py` (trains the 20 checkpoints), `inference_clean.py`
  (shared inference library: seeded MC-Dropout, Mondrian CP warm-up/online;
  its dead `run_mondrian_frozen` function — left over from before the split
  redefinition in the v3 changelog above — has been deleted in this patch, it
  was never called from anywhere), `build_canonical.py` (produces
  `data/per_sample_final_v3.csv`), `aggregate.py` (produces everything in
  `tables/` from that one file, including the reintroduced h-sensitivity
  table), `figures_final.py` (produces everything in `figures/`). Also
  includes the two small local packages these scripts import:
  `scripts/models/bayesian_lstm.py` and
  `scripts/conformal/{mondrian_cp,adaptive_lambda_cp}.py`.
- `tables/` — every current `.csv`/`.tex` table, all generated by
  `aggregate.py` from `data/per_sample_final_v3.csv`: `table_main_v3`
  (main ECR table, 4 conventions × 4 methods × 3 stages),
  `table_mondrian_unbounded_v3` (unbounded-interval rates),
  `table_B_trainpct_v3` / `table_trainpct_cohort_v3` /
  `table_trainpct_full_vs_cohort_v3` (train-set-percentile convention: full
  availability vs. the ≥1-window-per-stage cohort, side by side),
  `table_cusum_cohort_check_v3` (same cohort restriction applied to CUSUM, as
  a confirmation check), `paired_diff_v3` (within-engine paired differences,
  both bootstrap variants), `table_matched_v3` (occupancy-matched
  convention), `bucket_geometry_v3`, `A1_clocks_v3`, `A2_decile_v3`
  (mechanism diagnostics), `table_h_sensitivity_v3` (CUSUM convention, k
  fixed, h ∈ {5,7,10}, 48 rows — reintroduced in this patch, absent from v2).
  Every `.tex` fragment's header comment records the `aggregate.py` function
  that produced it and `per_sample_final_v3.csv`'s md5 at generation time.
  `tables/v2_superseded/` holds the complete set of v2 tables (all of the
  above, `_v2`-suffixed, pre-patch), and `tables/v1_superseded/` the v1 set
  (`table_main_final`, `h_sensitivity_kfixed_final`, `paired_diff_final`,
  `A1_clocks_final`, `A2_decile_final`, `A3_crosstab_final`,
  `A4_occupancy_final`, `B_trainpct_final`, `C_matched_final`, and their
  `.tex` siblings) — both unmodified, for audit.
- `figures/` — `fig_clocks_v3.pdf`, `fig_residual_deciles_v3.pdf`,
  `fig_failure_deciles_v3.pdf`, all produced by `scripts/figures_final.py`
  from the v3 data. `figures/v2_superseded/` and `figures/v1_superseded/`
  hold the pre-patch v2 and original v1 figures respectively.
- `legacy_leaked/` — the complete first-version pipeline, archived (not
  deleted) for audit. See its own note below. **Do not cite or reproduce
  anything from this directory** — every number it produced is withdrawn.

## Reproduction

Dependencies: Python 3, `numpy`, `pandas`, `torch`, `scipy`, `matplotlib`,
`scikit-learn`. GPU is optional (helps `train_clean.py`; inference-only
scripts run fine on CPU, just slower — but see the CUDA/CPU note below).

**Regenerate every v3 table and figure from the canonical file (fastest, no
GPU, no raw data needed):**
```
python3 scripts/aggregate.py       # writes tables/*_v3.{csv,tex} and related
python3 scripts/figures_final.py   # writes figures/*_v3.pdf, reads data/per_unit_clocks_final_v3.csv + tables/A2_decile_v3.csv
```

**Regenerate only the h-sensitivity table** (k fixed, h ∈ {5,7,10}, 48 rows;
reintroduced in the Stage 0-R3 patch, run `scripts/aggregate.py` at least
once first if `data/per_sample_final_v3.csv` isn't already loaded/cached):
```
python3 -c "import sys; sys.path.insert(0, 'scripts'); from aggregate import table_h_sensitivity; table_h_sensitivity()"
```

**Rebuild `data/per_sample_final_v3.csv` from the 20 checkpoints in `models/`**
(re-runs inference only, does not retrain):
1. Download the NASA C-MAPSS Turbofan Degradation dataset (`train_FD001.txt`
   .. `train_FD004.txt`, `test_FD00*.txt`, `RUL_FD00*.txt`) from the [NASA
   Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
   and place them under `data/cmapss/`. Raw data is NOT included in this
   repository.
2. `python3 scripts/build_canonical.py` — reads `models/*.pt` + `meta_*.json`,
   writes `data/per_sample_final_v3.csv` (+ `.md5`,
   `per_unit_clocks_final_v3.csv`, `trainpct_thresholds_final_v3.csv`). This
   is the slow step: one full MC-Dropout inference pass (50 dropout draws per
   window) over every calibration and evaluation engine, for all 20 (fd, seed)
   models, at h ∈ {5, 7, 10} in the same pass. **Note:** MC-Dropout is seeded
   deterministically (see Design notes below), but the dropout RNG stream
   differs between CPU and CUDA backends for the same seed — the canonical
   files in this repository were built on CUDA; regenerating on CPU-only
   hardware will NOT reproduce them bit-for-bit (verified: re-running on CPU
   shifts calibration-bucket sizes, e.g. FD001/seed0's CUSUM middle bucket
   goes from 397 to 394 calibration scores). This does not affect
   `aggregate.py`/`figures_final.py`, which only read the already-built CSV.

**Retrain the 20 base models from scratch** (only needed to verify training
itself, not to reproduce the note's numbers — the checkpoints in `models/`
already are the ones every table in this repo comes from):
1. Same C-MAPSS download as above, under `data/cmapss/`.
2. `python3 scripts/train_clean.py` — trains all 4 subsets × 5 seeds,
   overwriting `models/`. Stage 1: 50 epochs, patience 10. Stage 2: 100
   epochs (`CFG['epochs_stage2']`), patience 15, cosine-annealed LR.
   Non-deterministic in the sense that a from-scratch retrain is not
   guaranteed to reproduce these exact 20 checkpoints bit-for-bit (ordinary
   PyTorch/CUDA training non-determinism); the inference layer downstream of
   a trained checkpoint (`build_canonical.py`) *is* fully deterministic given
   a checkpoint and a fixed CPU/CUDA backend, via seeded MC-Dropout (see
   above and below).

## Design notes

- **4-way engine-level split**, per (subset, seed), disjoint by construction:
  train 60% / early-stop-val 15% / calibration 12.5% / evaluation 12.5%.
  Recorded per checkpoint in `models/meta_*.json` (`split_hash`, and the
  explicit engine list for each of the 4 partitions).
- **Scaler**: `MinMaxScaler` fit ONLY on the train partition's rows;
  early-stop-val/calibration/evaluation are `.transform()`-only.
- **Early stopping**: both training stages' early-stopping criterion is
  computed ONLY on the early-stop-val partition — never on data later used
  for calibration or evaluation. (This is the fix for v1's leak; see
  `legacy_leaked/` note.)
- **MC-Dropout**: `n_samples=50` per window (`models/bayesian_lstm.py::predict_mc`);
  total predictive variance = aleatoric mean + epistemic (between-sample)
  variance: `sigma_total = sqrt(mean(sigma_b^2) + Var(mu_b))` — **[v3
  changelog fix]** averaging the per-draw variances *before* the square root
  (was `sqrt(mean(sigma_b)^2 + Var(mu_b))`, which understates total variance
  by Jensen's inequality). Dropout draws are seeded deterministically per
  `(subset, seed, engine)` via `inference_clean.py::deterministic_seed`
  (`fd_num*1e7 + seed*1e5 + unit`, applied via `torch.manual_seed()`
  immediately before the 50 forward passes) — every unit's mu/sigma is
  reproducible regardless of call order or process, ON A FIXED CPU/CUDA
  backend (the dropout RNG stream differs between the two backends for the
  same seed; the canonical files here were built on CUDA — see Reproduction
  above). (v1 never seeded this at all; see `legacy_leaked/` note — a
  separate, orthogonal issue from the CPU/CUDA backend note here.)
- **Window/label alignment**: window k covers raw rows k..k+29 (SEQ=30),
  label is row k+29's own RUL (its last frame) — mirrors `train_clean.py`'s
  training-time windowing exactly, for k=0..L-30 inclusive (L-29 windows
  total for an L-row unit).
- **CUSUM change-point detection** — **[v3 changelog redefinition]** baseline
  window is `max(floor(0.2*n), 20)` scores of the unit's OWN trajectory (was
  a hardcoded `scores[:10]`), scaling with unit length instead of a fixed
  constant. `mu0`/`sigma` are that window's mean/std (`+1e-8` floor on
  sigma). An instance whose CUSUM statistic never crosses the threshold
  returns `cp = n` exactly (was `cp = n - 1`), so `assign_stage(t, n, n)`
  leaves that instance's LATE bucket genuinely empty rather than containing
  one leftover timestep. k=0.7σ₀ FIXED across h ∈ {5.0, 7.0, 10.0} (not
  linked to h); all three h values' intervals are generated in the SAME
  construction pass in `build_canonical.py` (h=7 is primary/unsuffixed
  columns, h=5/10 give `_h5`/`_h10` columns).
- **Mondrian CP — online mode**: one `MondirianCP(alpha=0.10, W_max=200,
  total_lifetime_est=AVG_LIFETIME[fd])` instance per h is warmed by feeding
  every calibration engine's (mu, sigma, y_true) triple, in the order
  `cal_units` appears in `meta_*.json`; λ_init=0.005 (never overridden). The
  SAME warmed instance is then shared and continuously updated
  (`.update()` after every `.predict_interval()`) across ALL evaluation
  engines, in the order `eval_units` appears in `meta_*.json` — order
  matters here by construction. Internally,
  `AdaptiveLambdaCP.predict_interval()` appends a pseudo-score of `+inf`
  with unnormalized weight 1.0 to the buffer's exponentially-decaying
  weights before taking the weighted quantile — this is the mechanism that
  produces genuinely unbounded intervals when the real buffer's weighted
  mass at the target quantile level is thin (small/young buffer, or a stage
  whose calibration scores are unusually tight). Untouched by the Stage 0-R3
  patch.
- **Mondrian CP — split mode** — **[v3 changelog redefinition, was "frozen";
  Stage 0-R3 patch below]**: NOT an online/adaptive instance.
  `build_mondrian_split_quantiles()` fills each of the 3 stage buckets ONCE
  from ALL `cal_units`' scores (uniform weight — no λ, no time index, no
  exponential decay), then takes the standard split-CP empirical quantile as
  the direct k-th order statistic, `k = ceil((m+1)(1-alpha))` (1-indexed;
  `q_hat = inf` if `k > m`) — **[Stage 0-R3 fix]** previously approximated via
  `np.quantile(scores, ceil((m+1)(1-alpha))/m, method='higher')`, which is
  NOT exactly the k-th order statistic (its virtual index is scaled by
  `(m-1)`, not `m`, so dividing `k` by `m` first shifts the selected index by
  one for generic `m`; verified case `m=397, k=359`: old call selected the
  360th smallest score, correct value is the 359th, `1.500141`).
  `run_mondrian_split()` applies that one fixed per-stage quantile to every
  evaluation sample — no `.update()`, no rollback, no shared mutable state;
  evaluation-unit order is provably irrelevant (the function has no state at
  all). The original "frozen" definition (`copy.deepcopy()` of the online
  instance, predict-only, never fed eval-time feedback) was still an
  adaptive, exponentially-weighted machine underneath; this is the
  classical, non-adaptive Mondrian conformal predictor the name "split" is
  meant to convey. Column names (`mf`, `covered_mf`, `width_mf`, `is_inf_mf`,
  ...) are unchanged from v1; what they mean is not.
  Unbounded-interval threshold: `np.isinf(width)`, `width = hi - lo`. For
  Mondrian (split), a stage's interval is unbounded when that stage's
  calibration bucket is empty or `k > m` (standard split-CP small-sample
  rule) — never from buffer/weight thinness, since there is no buffer.
  Per-engine ECR aggregation reports two numbers for Mondrian CP: one
  excluding unbounded samples ("finite-only"), and one treating every
  unbounded interval as covered ("unbounded-as-covered" — correct by
  definition, since an infinite-width interval covers any finite true
  value), plus the unbounded rate itself (numerator/denominator both given)
  — never one silently substituted for the other. This finite-only
  convention is applied uniformly across every table in `tables/`.
- **Estimand and unique-engine-block bootstrap**: the headline ECR estimand
  is a per-engine average (compute each engine's own coverage rate first,
  then average across engines within a cell), not a pooled per-timestep
  average — this is what makes an "engine" the natural resampling unit.
  Paired-difference confidence intervals (`paired_diff_v3.csv`) use a
  unique-engine-block bootstrap (10,000 resamples): the resampling block is
  one raw C-MAPSS unit id's full set of instances **across all 5 seeds**,
  drawn as a whole block — not a single (seed, unit) instance — so that
  resampling respects the fact that the same physical engine reappears
  across every seed's calibration/evaluation split. The older,
  per-(seed,unit)-instance bootstrap (which treats each seed's copy of an
  engine as an independent draw) is kept alongside as a sensitivity column,
  not as the primary estimate.
- **Cohort restriction (≥1 window in all 3 stages)**: a "cohort" instance is
  one `(fd, seed, unit)` triple that has at least one window assigned to
  EACH of a convention's 3 stages (early/middle/late) — **not** `(seed,
  unit)` alone, since raw C-MAPSS unit ids are not globally unique (e.g.
  FD001 and FD003 both have units 1..100; grouping without `fd` silently
  merges two different physical engines' stage-membership sets — see "Known
  retracted analyses" below). `compute_cohort_instances(df, stage_col)`
  groups by `['fd', 'seed', 'unit']`. This restriction exists to check
  whether a convention's stage-wise conclusions are an artifact of
  structural population imbalance (e.g. short-lived evaluation units that
  never reach a stage at all under a convention with an absolute, non-self-
  referential threshold) rather than a genuine coverage effect.

## `legacy_leaked/`

This directory is the complete first-version pipeline (code, trained
checkpoints, and every table/figure it produced), archived for audit —
**not deleted, but withdrawn**. Its base models' early-stopping criterion was
computed on the same held-out data later used for conformal calibration and
evaluation (a leak: model selection saw feedback from the calibration set
before that set was used to calibrate); its `MinMaxScaler` was fit on the
full pre-split training file rather than the train partition alone; and its
MC-Dropout inference was never seeded, making every reported number
irreproducible even in principle (two independent runs of the same model on
the same engine gave different predictions). Every number the first-version
manuscript reported is withdrawn as a consequence. Nothing in this directory
should be cited, reproduced, or reused — it exists only so a reader can see
exactly what was wrong and why the pipeline was rebuilt.

## Known retracted analyses

Two analyses produced during this repository's history are retracted and
must not be cited, independently of the `legacy_leaked/` first-version
withdrawal above:

1. **Sample-size subsampling attribution analysis** (`subsample_instability.py`
   / `subsample_instability_v2.py` in the working history, not included in
   this repository's `scripts/`). It tested whether Mondrian CP's
   non-late-worst instability on FD001/FD003 was a calibration
   sample-size effect, by bootstrap-subsampling FD002/FD004's evaluation
   engines down to FD001/FD003's count. Its stated conclusion (sample size
   is not the primary cause; FD001/FD003's own data characteristics are)
   is **retracted** — not because the conclusion is necessarily wrong, but
   because it was computed from cached predictions in a per-sample file
   built by **unseeded** MC-Dropout inference (the same seeding gap
   described under `legacy_leaked/`), which is irreproducible even in
   principle. It has not been recomputed on seeded, v3 predictions and
   should not be cited until it is.
2. **The early Stage 0-R2 "4/16" cohort-restriction result**, and the
   associated "Mondrian CP (online) flips in 2/4 cells" confirmation-check
   result. Both were produced by a bug in `compute_cohort_instances`: it
   grouped by `(seed, unit)` without `fd`, and since C-MAPSS unit ids are
   NOT globally unique across subsets (FD001 and FD003 both have units
   1..100), `(seed, unit)` tuples collided across FDs, silently merging two
   different physical engines' stage-membership sets. This produced a wrong
   total instance count (221, vs. the correct 455) and a wrong cohort size
   (110, vs. the correct 150), which in turn produced the erroneous 4/16 and
   2/4 numbers. **Corrected** (see the v3 changelog above and
   `tables/table_trainpct_full_vs_cohort_v3.tex`,
   `tables/table_cusum_cohort_check_v3.csv`): 14/16 flips for the
   train-set-percentile cohort (matching an independent external
   recomputation exactly), 0/16 flips for the CUSUM cohort. The 4/16 and 2/4
   figures must not be cited; only the corrected 14/16 and 0/16 figures are
   current. (These cohort figures are themselves unaffected by the Stage
   0-R3 patch — the patch changes only Mondrian CP (split), which was never
   the source of the 4/16 or 2/4 errors.)

## File tree

```
.
├── README.md
├── data/
│   ├── per_sample_final.csv               (v1, superseded)
│   ├── per_sample_final.md5               (v1, superseded)
│   ├── per_sample_final_v2.csv            (v2, superseded)
│   ├── per_sample_final_v2.md5            (v2, superseded)
│   ├── per_sample_final_v3.csv
│   ├── per_sample_final_v3.md5
│   ├── per_unit_clocks_final.csv          (v1, superseded)
│   ├── per_unit_clocks_final_v2.csv       (v2, superseded)
│   ├── per_unit_clocks_final_v3.csv
│   ├── trainpct_thresholds_final.csv      (v1, superseded)
│   ├── trainpct_thresholds_final_v2.csv   (v2, superseded)
│   └── trainpct_thresholds_final_v3.csv
├── models/
│   ├── best_{FD001..FD004}_seed{0..4}.pt      (20 files)
│   └── meta_{FD001..FD004}_seed{0..4}.json    (20 files)
├── scripts/
│   ├── train_clean.py
│   ├── inference_clean.py
│   ├── build_canonical.py
│   ├── aggregate.py
│   ├── figures_final.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── bayesian_lstm.py
│   └── conformal/
│       ├── __init__.py
│       ├── mondrian_cp.py
│       └── adaptive_lambda_cp.py
├── tables/
│   ├── table_main_v3.{csv,tex}
│   ├── table_mondrian_unbounded_v3.csv
│   ├── table_B_trainpct_v3.csv
│   ├── table_trainpct_cohort_v3.{csv,tex}
│   ├── table_trainpct_full_vs_cohort_v3.{csv,tex}
│   ├── table_cusum_cohort_check_v3.csv
│   ├── paired_diff_v3.csv
│   ├── table_paired_diff_v3.tex
│   ├── table_matched_v3.{csv,tex}
│   ├── bucket_geometry_v3.csv
│   ├── A1_clocks_v3.csv
│   ├── A2_decile_v3.csv
│   ├── table_h_sensitivity_v3.{csv,tex}
│   ├── v2_superseded/
│   │   └── (all of the above, _v2-suffixed, pre-Stage-0-R3-patch)
│   └── v1_superseded/
│       ├── table_main_final.{csv,tex}
│       ├── h_sensitivity_kfixed_final.csv
│       ├── table_h_sensitivity_final.tex
│       ├── paired_diff_final.csv
│       ├── table_paired_diff_final.tex
│       ├── A1_clocks_final.csv
│       ├── A2_decile_final.{csv,tex}
│       ├── A3_crosstab_final.csv
│       ├── A4_occupancy_final.csv
│       ├── B_trainpct_final.csv
│       └── C_matched_final.csv
├── figures/
│   ├── fig_clocks_v3.pdf
│   ├── fig_residual_deciles_v3.pdf
│   ├── fig_failure_deciles_v3.pdf
│   ├── v2_superseded/
│   │   ├── fig_clocks_v2.pdf
│   │   ├── fig_residual_deciles_v2.pdf
│   │   └── fig_failure_deciles_v2.pdf
│   └── v1_superseded/
│       ├── fig_clocks_final.pdf
│       ├── fig_residual_deciles_final.pdf
│       └── fig_failure_deciles_final.pdf
└── legacy_leaked/
    ├── README_v1.md
    ├── results/
    └── scripts/
```
