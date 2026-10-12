# Attribution and third-party notices

This release contains new Julia/Lean implementations, synthetic tests and
derived aggregate summaries. It does not redistribute external data archives,
original MATLAB/C++/CVX source packages, proprietary solver binaries, private
correspondence, or unpublished third-party manuscripts. No blanket license
for those materials is granted or implied.

## Scientific and computational sources

- **Kitamura and Stoye (2018)**, *Nonparametric Analysis of Random Utility
  Models*, Econometrica. [Article and supplements](https://doi.org/10.3982/ECTA14478).
  Source of the stochastic-demand testing framework and original replication
  lineage. A Julia implementation is not a certification of their complete
  original package or a claim of authorship of their method.
- **Smeulders, Cherchye and De Rock (2021)**, *Nonparametric Analysis of Random
  Utility Models: Computational Tools for Statistical Testing*, Econometrica.
  [Author code repository](https://github.com/BSmeulders/RandomUtilitywithCG/tree/919ba915623273a208e25666a47120745b7510f9).
  Pinned implementation ancestor and mathematical control-function profile.
  The inspected repository tree has no LICENSE/COPYING/NOTICE file; its
  availability is not treated as permission to redistribute its files.
  Its README requires separately acquired original KS files and a CPLEX
  installation for the upstream MATLAB workflow.
- **Wilfried Youmbi Fotso**, received `RUMCODEFINAL.zip` replication snapshot.
  The supplied snapshot provides the code/data/output evidence investigated
  here. Original files and saved outputs remain unchanged and are not
  redistributed. This release identifies a particular checker defect and its
  consequences; it does not establish intent or the historical generating
  revision of all manuscript tables.
- **Abi Adams**, *Mutually Consistent Revealed Preference Demand Predictions*,
  separate `231552-V1.zip` derivative data deposit used for annual-input
  reconciliation. This is not identified as the complete KS replication
  package; obtain its data independently under its own terms.
- **Koida and Shirai**, *A dual approach to nonparametric characterization for
  random utility models*. [Author arXiv version](https://arxiv.org/abs/2403.04328v3).
  Motivation and comparison for the observable weak-axiom dual. Their theorem
  is not an assumed proof premise for the new finite WARP characterization.
- **Chen and Fang**, inference for functionals with first-order degeneracy.
  [Author arXiv paper](https://arxiv.org/abs/1901.04861).
  Basis for the numerical second-order calibration investigated here; the
  application-specific sampling assumptions are maintained separately.
- **Dean, Ravindran and Stoye**, *A Better Test of Choice Overload*.
  [Author working paper](https://www.columbia.edu/~md3405/Working_Paper_24.pdf).
  Motivation for separating model restrictiveness from power and for
  investigating localized calibration. Its arguments are not automatically
  transferred to a potentially nonconvex distance-gap functional.

## Software dependencies

Julia, JuMP, HiGHS, OSQP, Clarabel, Mathlib and other dependencies remain under
their respective licenses. Project/manifest and Lean toolchain files pin the
versions used where included. Install dependencies through their official
package managers rather than bundling local caches or binaries. Consult each
dependency's own license for use and redistribution.

MATLAB/CVX/CPLEX are not included. The external received archive contains
third-party solver distributions and license files, which are outside the
release allowlist. Do not copy them into this repository or its ZIP.

## Scope of attribution

Attribution records provenance, not endorsement, coauthorship of this code,
permission to redistribute third-party assets, or agreement with every
interpretation. Any license later added for newly written code must state
its scope and must not purport to relicense excluded external inputs.
