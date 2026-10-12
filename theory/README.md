# Optional data-free theory checks

This subtree contains the finite observable stochastic-WARP dual and a small
set of deterministic CMU and distance-calibration proofs. It does not require
household observations, the collaborator's supplied archive, or MATLAB/CVX.
The module layout is retained to preserve the existing Lean imports.

## Python validation

Use Python 3.13 (tested with 3.13.0). From this directory:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r dual/requirements.txt
THEORY_PYTHON=.venv/bin/python bash verify.sh --python-only
```

The 39 tests check exact-arithmetic H-rule provenance, support against small
enumerated benchmark graphs, consistency witnesses and separating inequalities,
and independent projection auditing. In particular, they recover the five-cycle
inequality, the wheel's coefficient-two inequality, complete-budget adding-up,
and an observable counterexample missed by weaker pairwise conditions.
The optimized `rpm_hrep.py` producer uses Python's standard library; the
consistency/projection implementations and tests also use NumPy and SciPy.
The `stochastic/warp_core.py` helper supplies a synthetic three-budget
WARP-but-not-SARP validation geometry; no household data are loaded by the tests.
Its geometry/checker functions are unchanged; only an unused private-paper CLI
illustration and its dispatcher/imports were removed from this release copy.
The original source was not edited.
Generation bounds raise an exception rather than return a partial rule as a
complete characterization.

## Lean compilation

Install Lean through elan, then, from this directory:

```sh
lake exe cache get
bash verify.sh --lean-only
```

`lean-toolchain` pins Lean 4.32.0-rc1. The Lake files pin Mathlib to
`a3364faec42918fcd84a03a255b50570129f9ead`, including its transitive dependencies.
The verification script compiles the five finite RPM modules in dependency
order, then the four CMU proof files, and prints the axioms of their principal
theorems. This explicit script is the verification target: plain `lake build`
has no default manuscript target. Mathlib cache retrieval needs network access
and a sizeable dependency download on the first run. Compilation is sequential.

The finite RPM consumer proves complete-budget mixture equivalence with its
generated observable H rows, and the integer/unit-right-hand-side versions.
`RPM` is the historical module name; here it refers to the finite mixture of
complete conflict-free/WARP choice profiles, not to an empty-demand extension.

The four CMU files cover:

- `CMUEmptyDemandGeometry.lean`: deterministic geometry used for empty-demand
  witnesses.
- `CMUEmptySuperiority.lean`: response/mass accounting and the squared-distance
  gain sandwich for the specified inner construction.
- `CMULocalCalibration.lean`: ray bounds, deterministic calibration transfer,
  weighted interval aggregation, and a nonconvex local-gap example.
- `CMURawNullGeometry.lean`: a raw-target squared-distance-gap bound for nested
  sets under an explicit common-projection hypothesis.

## Exact limits of the checks

These are not a complete formalization of the economic paper or empirical
inference. The finite Lean H producer is distinct from the optimized Python
deletion/disjunction producer: the Python tests do not constitute a Lean
implementation-refinement proof. The continuous individual CMU representation,
strict observed-budget rationalization, sampling/control-function assumptions,
unknown-face calibration, uniform statistical size, and optimal tuning have
not all been proved by these files. The deterministic tail-control transfer
requires a valid external noise-tail premise. The empty-demand computations
concern the explicitly witnessed common-response **inner family**, not every
coherent CMU demand system; absence caps are sensitivity assumptions, not
estimated empty-choice frequencies.

The independent tests and formal proofs therefore support the finite dual,
explicit geometry and certified computations. They do not alone establish a
new empirical WARP-versus-RUM pass/fail contrast.

## Release validation

On 12 October 2026, the staged package passed all 39 Python tests with inherited
`PYTHONPATH` and `PYTHONHOME` cleared. All nine included Lean files were
compiler-checked against the pinned dependencies with warnings treated as
errors (the two unused-variable linters are explicitly disabled). The 57
printed principal-theorem audits reported only Lean's standard axioms:
`propext`, `Classical.choice`, and `Quot.sound`. No `sorry` or custom axiom is
present in these included proof sources. These are source/compiler and small
benchmark checks, not an additional statistical validity theorem.
