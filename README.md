# Stochastic WARP and coherent-CMU replication

Julia replication code for the corrected UK three-good WARP/RUM calculations,
paired relative-distance comparisons, and the distinct actual-empty-demand
CMU sensitivity results. This is the implementation used for our new results;
MATLAB is not required for the public aggregate-data replay.

The release includes synthetic checker examples, all 19 aggregate UK budget
designs, derived bootstrap inputs, numerical verification, and optional finite
dual/Lean material. It excludes household records, supplied MATLAB/CVX files,
licenses, and private correspondence. See [DATA_ACCESS.md](DATA_ACCESS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quickstart

Use Julia **1.12.7** (compatible project series: 1.12), then run from this
repository's root:

```sh
julia scripts/instantiate.jl
julia --project=. scripts/verify_checksums.jl
julia --project=. test/runtests.jl
```

The acceptance tests include the positive-price counterexample, correct WARP
enumeration, normalized projections, actual-empty CMU certificates, exact
restrictiveness bounds, and an independent analytic inference adapter. These
do not require household microdata. The main and isolated Clarabel solver
environments are both pinned; the setup script instantiates both.

## Reproduce the released numerical results

The supported portable commands are in `scripts/`. They use released aggregate
fixtures and write to an explicitly named fresh output location. They separate
fresh type enumeration/projection from verification of saved joint bootstrap
draws. A saved-draw replay is not a new household-level bootstrap refit.

```sh
# All 19 windows, all three comparison targets and c=.5/1/2:
# reaggregate saved derivative envelopes and verify recorded tails/quantiles.
julia --project=. scripts/replay_aggregates.jl \
  --mode recorded --output generated/recorded

# Fresh enumeration, utility-witness checks, source-cone J statistics,
# normalized point projections, seven absence-cap fits and pooled losses.
julia --threads=auto --project=. scripts/replay_aggregates.jl \
  --mode points --cells all --output generated/points

# Optional, substantially more expensive: re-solve bootstrap QPs using all
# 1,000 released paired aggregate draws; does not re-estimate households.
julia --threads=auto --project=. scripts/replay_aggregates.jl \
  --mode bootstrap --cells all --output generated/bootstrap
```

Output directories must not already exist. Neither saved draws nor unfavorable
windows are discarded. The release's selected Julia tests passed **3,697**
assertions; the self-contained finite-dual Python suite passed **39** tests.
GitHub Actions runs the public tests and recorded numerical replay. Fresh
all-window point validation is documented separately in the release evidence.

See [RESULTS.md](RESULTS.md) for the numerical findings, model definitions,
primary calibration, and remaining inference limitations. In particular:

| Paired comparison | Frozen primary c=1 p |
|---|---:|
| Correct complete WARP versus RUM, source cones | 0.148851 |
| Correct complete WARP versus RUM, raw normalized hulls | 0.151848 |
| Witnessed actual-empty CMU inner family versus RUM | 0.093906 |

The final row is a different model and null, not ordinary WARP. Its assumed
absence cap is not measured in the UK data. Do not promote the c=2 sensitivity
result to the primary test. No corrected ordinary-WARP pass/RUM-fail window
has been found in this audited three-good application.

## External-input re-estimation

The focal raw-input Julia port can be run with separately acquired,
hash-matching inputs:

```sh
julia --project=. --threads=auto run_wilfried_replication.jl \
  --archive /path/to/RUMCODEFINAL.zip \
  --package /path/to/extracted/RUMCODEFINAL \
  --data-archive /path/to/231552-V1.zip \
  --bootstrap 1000 --seed 13732 \
  --output generated/focal-reestimation.json
```

Do not publish `external/` or generated private-input reports. The focal driver
records the original input identity and fresh Julia random streams; its tails
are not asserted to be bitwise identical to an earlier program's streams.

Root-level research-stage drivers are preserved for computational provenance
and reusable helper functions. Some retain historical report/hash guards and
are **not** supported standalone release commands; use the portable `scripts/`
entry points. Missing external evidence must fail, not silently bypass a guard.

## Optional theory

See [theory/README.md](theory/README.md) for the data-free Python finite-dual
checks and pinned Lean proofs. Lean verifies stated deterministic finite and
geometric results. It does not certify the UK sampling model, control-function
identification, bootstrap validity, an optimal choice of c, or every continuous
CMU representation claim.

## Integrity and attribution

`SHA256SUMS` identifies the exact released files. `ORIGINAL_SOURCES.json`
records the original computational source hashes and release-only path
adaptations. Derived/sanitized fixtures have their own new hashes; an original
report hash is provenance, not the hash of the transformed fixture.

This is research replication material, not a new authorship or collaboration
agreement. The mathematical/computational lineage and redistribution boundaries
are documented in `THIRD_PARTY_NOTICES.md`. No blanket sublicense of a supplied
third-party program or dataset is granted by this release.
