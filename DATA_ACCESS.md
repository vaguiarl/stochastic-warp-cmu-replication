# Data access and immutable prerequisites

No household records, supplied MAT workspaces, original MATLAB/CVX code,
solver licenses, private correspondence, or third-party archives are included.
The included checker example is synthetic; released numerical summaries are
aggregates. Public availability of a source is not a blanket redistribution
license. Obtain external inputs independently under their applicable terms.

## Inputs used in the received-package investigation

| External input | SHA-256 | Role |
|---|---|---|
| `RUMCODEFINAL.zip` | `b72d55c823926b680f041320e0553138a2270b26d66d2ee2b6074c18efeb64e1` | Received code/data/output snapshot; required for the original saved-column and focal-output audit |
| `231552-V1.zip` | `2a7c2473baf5a9eb68a392e087ca9d3c0499777d72efc7798f9419b18c13fece` | Separate derivative FES source archive used for immutable annual-input reconciliation |

The latter deposit is Abi Adams's *Mutually Consistent Revealed Preference
Demand Predictions*, not the complete original Kitamura–Stoye testing package.
The annual raw CSV inputs and prices from the received package match the
checked derivative archive byte-for-byte. A matching filename is insufficient;
verify the archive hash before a replay. The received archive is obtained
from its provider, not from a private sharing URL published in this package.

The focal supplied workspace is
`RUM_7LengthBudget_3Classes_3PolyDegree_1000BSreps_Series_Endogenous0taunot0_20231216_202.mat`,
SHA-256 `1c24ca7e5a1d814ce1170ef72e935a2ff78b92005f41a8383209b967fdc7b634`.
It is an external prerequisite, not a released file. The audit uses it as
immutable evidence; it does not execute its embedded source or establish
which historical code revision created it.

## Published computational lineage

- Kitamura and Stoye, *Nonparametric Analysis of Random Utility Models*,
  Econometrica (2018): [article and supplements](https://doi.org/10.3982/ECTA14478).
  The original data/program supplement is a separate prerequisite where needed.
- Smeulders, Cherchye and De Rock, *Nonparametric Analysis of Random Utility
  Models: Computational Tools for Statistical Testing*, Econometrica (2021):
  [author repository](https://github.com/BSmeulders/RandomUtilitywithCG), pinned
  commit `919ba915623273a208e25666a47120745b7510f9`.
  Its MATLAB files supplement the original KS package; it is not a complete
  copy of the received application. The C++ input format contains
  count-prefixed share/expenditure/instrument arrays and normalized prices.

The public CG-input workflow and the received raw-CSV workflow are distinct.
Do not interchange their inputs or normalize the already normalized CG prices
again. Reproducing the received focal output requires the received snapshot,
not merely a current checkout of the public CG repository.

## Replay boundaries

Run the synthetic checks without external data. For data-dependent commands,
follow the repository README and provide explicit paths to independently
obtained archives/checkouts. Keep originals unchanged and write generated
outputs into a fresh output directory. Input checksums, software versions,
command options and numerical gates should be recorded before and after runs.

The Julia port does not require shipping proprietary MATLAB/CVX/CPLEX binaries.
Its random-number streams differ from the original MATLAB streams. Complete
saved-output reconciliation and fresh paired Julia diagnostics are different
reproduction targets. Fresh Monte Carlo tails must not be described as bitwise
copies of saved MATLAB tails.

Do not publish local data directories or their contents when sharing a replay.
Aggregate reports should retain window/sample/type counts, objective bounds,
tail counts and scientific scope, without household identifiers, household
frequency vectors, private paths, account tokens or correspondence identifiers.
