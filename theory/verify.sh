#!/usr/bin/env bash
# All examples below are data-free. Full empirical replication is separate.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
theory_root="$(pwd)"
theory_mode="${1:---all}"
if [[ "$#" -gt 1 ]] || [[ "$theory_mode" != "--all" && "$theory_mode" != "--python-only" && "$theory_mode" != "--lean-only" ]]; then
  echo "Usage: bash theory/verify.sh [--all|--python-only|--lean-only]" >&2
  exit 2
fi

if [[ "$theory_mode" != "--lean-only" ]]; then
  theory_python="${THEORY_PYTHON:-python3}"
  env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 "$theory_python" -m pytest -q -p no:cacheprovider \
    dual/test_rpm_hrep.py dual/test_rpm_consistency.py dual/test_rpm_projection_audit.py
fi

if [[ "$theory_mode" != "--python-only" ]]; then
  # Dependency object files are generated locally, never borrowed from another
  # manuscript. Compilation is sequential to keep memory use modest.
  for theory_module in RPMFourierMotzkin RPMIntegerH RPMGraphDual RPMGraphIntegerH RPMUnitH; do
    LEAN_PATH="$theory_root${LEAN_PATH:+:$LEAN_PATH}" lake env lean \
      -DwarningAsError=true -Dlinter.unusedSectionVars=false -Dlinter.unusedVariables=false \
      -o "spinoff/laws_of_demand/lean/$theory_module.olean" \
      "spinoff/laws_of_demand/lean/$theory_module.lean"
  done
  for theory_module in CMULocalCalibration CMURawNullGeometry CMUEmptyDemandGeometry CMUEmptySuperiority; do
    LEAN_PATH="$theory_root${LEAN_PATH:+:$LEAN_PATH}" lake env lean \
      -DwarningAsError=true -Dlinter.unusedSectionVars=false -Dlinter.unusedVariables=false \
      "docs/$theory_module.lean"
  done
fi
