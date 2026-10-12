# Results and interpretation

Release date: 12 October 2026. This is the Julia implementation used for the
new replication and inference results. It separates a received-code audit,
complete-choice WARP calculations, and a distinct empty-demand CMU
selection-sensitivity extension. They are not interchangeable models or tests.

## 1. A serious checker defect changes the received candidate family

The received `RUM_43_genAfw_new.m` uses the condition
`if max(F.*F') == true`. For a binary adjacency matrix, MATLAB returns a vector
of column maxima; its nonscalar `if` requires every entry to be true. The
condition therefore asks whether every selected budget has a mutual partner,
not whether any prohibited two-budget reversal exists. The scalar predicate
for the intended WARP check is `any(any(F.*F'))`.

This is a consequential model-definition bug, not a solver tolerance issue.
It does not establish intent or identify the historical source revision that
generated a published table. The original supplied files were not modified
or executed through MATLAB/CVX.

The package includes an exact synthetic example with positive prices and
positive choices:

| Budget | Prices | Choice | Expenditure |
|---|---|---|---:|
| 1 | (2, 3, 1) | (2, 1, 1) | 8 |
| 2 | (3, 2, 3) | (1, 2, 1) | 10 |
| 3 | (1, 3, 2) | (1, 1, 2) | 8 |

Cross-expenditures, with price budgets as rows and choices as columns, are
`[8 9 7; 11 10 11; 7 9 8]`. Budgets 1 and 3 strictly afford each other's
distinct choices. The received prefix filter accepts order 1,2,3 but rejects
order 1,3,2; correct WARP rejects all six budget permutations.

All 19 saved three-good candidate sets match the received recursive rule.
Of 947,924 saved columns, 873,769 violate direct WARP. Four windows
(1983–1989, 1985–1991, 1986–1992, 1993–1999) contain the entire Cartesian
product of patch choices. In those windows every nonnegative, adding-up-one
patch-probability vector is representable: there is no cross-budget
revealed-preference restriction. The other 15 retain exclusions from the
order-dependent filter, not the intended WARP axiom.

For the focal 1986–1992 window:

| Family | Types | Point statistic J | Bootstrap diagnostic |
|---|---:|---:|---:|
| Received family, not WARP | 1,680 | 0.001339 | Saved p=0.297; fresh Julia p=0.272 |
| Correct complete WARP | 347 | 10.946927 | Fresh Julia p=0.012 |
| Correct complete transitive RUM | 333 | 10.946927 | Fresh Julia p=0.012 |

The received family has 1,333 WARP-invalid types. The saved statistic and
297/1,000 upper-tail count are recovered from its supplied output. Fresh Julia
streams are not the original MATLAB substreams, so the fresh p-values are
not bitwise replications. The controlled point-fit discrepancy persists when
the saved target and original criterion are held fixed.

Numerical reconciliation retains the raw annual inputs, price construction,
patch order and both control-function stages. Across the 19 windows, regenerated
probabilities differ from the saved vectors by at most 1.382e-10 and prices by
at most 2.082e-17. The focal corrected RUM statistic is enclosed by
[10.946927290953003, 10.946927352032029]; the table displays the original
six-decimal rounding. These are numerical/reproduction checks, not sampling
confidence intervals. The source-profile diagnostic uses tau=0 and 1,000
draws. Saved-output reconciliation and a fresh estimator/bootstrap run are
explicitly distinguished in the released reports.

## 2. Correct ordinary WARP versus transitive RUM in the UK application

The maintained source sample has 46,391 annual observations, three commodity
groups, cubic control-function estimation, and 19 overlapping seven-year
windows. Correct WARP prohibits direct two-cycles; RUM prohibits all directed
cycles. A two-budget probability violation is shared by both models, so
allowing longer cycles cannot rescue the three focal windows.

At the nominal 5% source-profile diagnostic threshold, both models reject in
the same three windows (1986–1992, 1987–1993, 1988–1994) and do not reject in
the remaining 16. No audited window produces RUM rejection with correct WARP
nonrejection. Nonrejection is not proof that either model is true.

Ordinary WARP nevertheless has resolved positive distance gains in five
windows, 1977–1983 through 1981–1987. In the primary raw-affine,
normalized-hull calculation, the pooled squared losses are 0.00490310409
for RUM and 0.00478618815 for WARP, a 2.3845% reduction. Windowwise reductions
are 8.7401%, 4.1823%, 2.2230%, 4.0605%, and 10.6323%. The source-cone
calculation uses a different metric/set convention and gives a different
range; it must not be silently substituted for these normalized results.

## 3. Three distinct paired joint comparisons

| Comparison | Target and criterion | Primary c=1 p |
|---|---|---:|
| Complete WARP versus RUM | Source-clipped target, full-coordinate cones | 0.148851 |
| Complete WARP versus RUM | Raw-affine target, normalized convex hulls | 0.151848 |
| Witnessed actual-empty CMU inner family versus RUM | Raw normalized-hull loss minus the product-simplex floor | 0.093906 |

These are relative-distance comparisons, not model-adequacy tests. All 19
windows are retained. Frozen weights are `N_window / 46,391`; their sum is
not one because the windows overlap. Calendar-year household resampling and
both control-function refits are paired across windows and models. The
windows are not treated as 19 independent experiments. No Bonferroni/Holm
adjustment or favorable-window selection is used for these joint tests.

The numerical second-order calibration uses c=1 as the frozen primary rule.
Each displayed joint tail uses 1,000 draws and the Monte Carlo correction
`(exceedances + 1) / (1,000 + 1)`, unlike the source-profile diagnostic's
unadjusted tail fraction.
For the empty-demand comparison, c=0.5/1/2 gives p=0.119880/0.093906/0.039960.
The primary Monte Carlo 95% interval is [0.075718, 0.112719], crossing 10%.
The evidence is suggestive at a nominal 10% level, not robust 5% superiority.
Selecting c=2 because its real-data p-value is lower is not justified.

## 4. What the empty-demand extension adds

The extra deterministic profiles have actual common coherent-CMU witnesses:
complete WARP profiles, one-budget-only responders for every patch, and an
all-null profile. They are not arbitrary independently deleted choices.
An externally imposed absence cap and a common-response restriction define
a witnessed **inner** family, not all coherent CMU preferences.

At an assumed 10% per-budget absence cap, the pooled inner-family squared
loss is 0.00161678762, a 67.03% reduction from the unchanged RUM benchmark.
This is a descriptive sensitivity result. The supplied UK records do not
measure empty-choice frequencies; the cap is neither an estimated nonchoice
rate nor a tuning parameter to choose for significance. Preferences in this
construction need not have robust demand existence at counterfactual budgets.
No common latent coupling across all 25 annual budgets is established.

The positive-cap null reduces to equality between RUM loss and the original
product-simplex distance floor. The floor is subtracted from the loss; the
raw target is not clipped or replaced by its simplex projection. Zero absence
has a different null and is excluded from this equivalence.

## 5. What remains valuable

The complete-choice WARP/CMU theory and its observable dual are logically
separate from the defective implementation. Building in the direction of
Koida–Shirai's RUM dual, the finite weak-axiom dual removes latent type masses
and supplies observable weighted inequalities. A violated row certifies that
no common complete-WARP population can jointly reproduce the choice events.
The full finite characterization need not have a compact binary row catalogue
or polynomial complexity. Sharp bounds on observed cyclic-type mass apply to
feasible population vectors, not automatically to projected infeasible estimates.

Finite Lean certificates and exact optimization audits verify explicitly
stated algebraic/geometric claims. They do not establish UK survey-design or
control-function identification, negligible series bias, bootstrap validity,
critical-quantile regularity, or historical table provenance. The continuous
CMU implementation is a separate analytic bridge. Strict observed-budget
rationalization is required; uniqueness alone is not its substitute.

Separate known-Gaussian simulations study calibration and power, including
genuine-empty CMU alternatives. Their independent finite-anchor calibration
is not a uniform UK sampling theorem or proof of an optimal c. Known-face
experiments expose finite-sample failures of blindly substituting a limiting
cone law. A Lean-checked conservative c-free norm bound is available subject
to a valid noise-tail premise, but need not be sharp.

## Remaining replication questions

The received three-good candidate matrices themselves are not WARP, whatever
their historical origin. The exact source/output revisions used for the
tables still require confirmation. Native MATLAB/CVX was not run, and the
complete corrected five-good table has not been reproduced. No result here
asserts a general equivalence of GARP and SARP: the current transitive benchmark
uses strict open patches and excludes intersection/tie atoms.

## Numerical receipts

The [checker findings receipt](data/checker_findings_receipt.json) records
the allowlisted original audit findings and their content hashes, including
the synthetic example and the distinction between saved and fresh focal tails.
The [release validation](data/release_validation.json) records fresh all-19
type/projection and utility-witness checks, together with reaggregation of all
stored joint tails at c=0.5,1,2 and the per-window source-profile bootstrap tails.
Replaying stored aggregate draws does not itself refit household estimators.
