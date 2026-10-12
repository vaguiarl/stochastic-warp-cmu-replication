"""Audited finite stochastic-WARP law without a WARP-filtered type catalogue.

Columns are ALL complete budget-compatible Cartesian candidate paths.  A
nonnegative law must reproduce event marginals, have mass one, and give every
conflict-pair cylinder mass zero.  By default their nonnegative indicators are
summed into one badness moment: nonnegative masses make its zero expectation
equivalent to every conflict cylinder having zero expectation.  This is an
analytic compression of the
all-event-subset law to its complete-budget face, not yet a separately verified
Lean implementation refinement.  No candidate is filtered using WARP.

The LP is numerical.  A successful law is audited against the original target;
a failed law needs an independently checked Farkas vector.  If rationalization
works, the affine observable inequality is verified exactly against every
unfiltered candidate column.  No raw data are read or saved, and no bootstrap
p-value or population hypothesis-test decision is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm, prod
from time import monotonic
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class CandidateLimit(RuntimeError):
    """Geometry expansion exceeds a declared bound before allocation."""


@dataclass(frozen=True)
class ConsistencySystem:
    offsets: tuple[int, ...]
    conflicts: tuple[tuple[int, int], ...]
    candidates: np.ndarray
    matrix: sparse.csc_matrix
    metadata: dict

    @property
    def n_events(self) -> int:
        return self.offsets[-1]

    @property
    def n_budgets(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_candidates(self) -> int:
        return self.candidates.shape[0]

    @property
    def n_conflict_rows(self) -> int:
        return self.matrix.shape[0] - 1 - self.n_events


def build_consistency_system(offsets: Sequence[int], conflict_pairs: Iterable[tuple[int, int]], *,
                             max_candidates: int = 500_000,
                             max_nonzeros: int = 6_000_000,
                             aggregate_conflicts: bool = True) -> ConsistencySystem:
    """Build sparse marginal/cylinder rows from geometry only.

    Within-budget pairs are allowed (their cylinder rows are identically zero).
    Complete candidate choices already encode one patch per budget.  Supplied
    cross-budget pairs must be the intended full mutual-affordability conflicts;
    this API cannot establish geometric realization/completeness by itself.
    """
    raw = np.asarray(offsets)
    if (raw.ndim != 1 or len(raw) < 2 or not np.issubdtype(raw.dtype, np.integer) or
            raw[0] != 0 or np.any(np.diff(raw) <= 0)):
        raise ValueError("offsets must be strictly increasing integers beginning at zero")
    off = tuple(int(x) for x in raw)
    if type(max_candidates) is not int or type(max_nonzeros) is not int or min(max_candidates, max_nonzeros) < 1:
        raise ValueError("candidate and nonzero bounds must be positive integers")
    if type(aggregate_conflicts) is not bool:
        raise TypeError("aggregate_conflicts must be boolean")
    counts = tuple(b - a for a, b in zip(off, off[1:]))
    number = prod(counts)
    if number > max_candidates:
        raise CandidateLimit(f"unfiltered candidate product {number} exceeds {max_candidates}")
    if off[-1] > np.iinfo(np.int32).max:
        raise CandidateLimit("event count exceeds the declared int32 geometry representation")
    event_budget = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    pairs = set()
    for pair in conflict_pairs:
        if len(pair) != 2:
            raise ValueError("each conflict pair needs two event indices")
        u, v = pair
        if (isinstance(u, (bool, np.bool_)) or isinstance(v, (bool, np.bool_)) or
                not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)) or
                not (0 <= u < off[-1] and 0 <= v < off[-1]) or u == v):
            raise ValueError("conflict endpoints must be distinct valid integer event indices")
        pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    conflicts = tuple(sorted(pairs))
    expected_nonzeros = number * (2 + len(counts)) if aggregate_conflicts else \
        number * (1 + len(counts)) + sum(
            number // (counts[int(event_budget[u])] * counts[int(event_budget[v])])
            for u, v in conflicts if event_budget[u] != event_budget[v])
    if expected_nonzeros > max_nonzeros:
        raise CandidateLimit(f"geometry matrix needs {expected_nonzeros} nonzeros, above {max_nonzeros}")

    # Mixed-radix generation: every Cartesian path, without any conflict test.
    candidates = np.empty((number, len(counts)), dtype=np.int32)
    indices = np.arange(number, dtype=np.int64)
    stride = 1
    for j in range(len(counts) - 1, -1, -1):
        candidates[:, j] = (indices // stride) % counts[j] + off[j]
        stride *= counts[j]
    row_chunks = [np.zeros(number, dtype=np.int64), 1 + candidates.ravel()]
    column_chunks = [indices, np.repeat(indices, len(counts))]
    data_chunks = [np.ones(number), np.ones(number * len(counts))]
    badness = np.zeros(number, dtype=np.int64) if aggregate_conflicts else None
    for row, (u, v) in enumerate(conflicts, start=1 + off[-1]):
        j, k = int(event_budget[u]), int(event_budget[v])
        if j == k:
            continue
        selected = np.flatnonzero((candidates[:, j] == u) & (candidates[:, k] == v))
        if aggregate_conflicts:
            badness[selected] += 1
        else:
            row_chunks.append(np.full(len(selected), row, dtype=np.int64))
            column_chunks.append(selected)
            data_chunks.append(np.ones(len(selected)))
    if aggregate_conflicts:
        selected = np.flatnonzero(badness)
        row_chunks.append(np.full(len(selected), 1 + off[-1], dtype=np.int64))
        column_chunks.append(selected)
        data_chunks.append(badness[selected].astype(float))
    rows, columns = np.concatenate(row_chunks), np.concatenate(column_chunks)
    conflict_rows = 1 if aggregate_conflicts else len(conflicts)
    matrix = sparse.coo_matrix((np.concatenate(data_chunks), (rows, columns)),
                               shape=(1 + off[-1] + conflict_rows, number)).tocsc()
    matrix.sort_indices()
    if ((not aggregate_conflicts and matrix.nnz != expected_nonzeros) or
            matrix.nnz > expected_nonzeros or not np.all(matrix.data >= 1) or
            not np.all(matrix.data == np.floor(matrix.data))):
        raise ArithmeticError("Cartesian cylinder matrix construction audit failed")
    candidates.flags.writeable = False
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    metadata = {"candidate_domain": "ALL complete budget-compatible Cartesian paths",
                "candidate_filter": "none; conflicts imposed by zero cylinder moments",
                "warp_filtered_columns": False, "n_candidates": number,
                "n_events": off[-1], "n_budgets": len(counts), "patch_counts": list(counts),
                "n_conflict_pairs": len(conflicts), "matrix_nonzeros": matrix.nnz,
                "n_conflict_rows": conflict_rows, "aggregate_conflicts": aggregate_conflicts,
                "zero_moment_equivalence": "sum of nonnegative conflict indicators has expectation zero iff each does",
                "within_budget_zero_cylinder_rows": sum(event_budget[u] == event_budget[v]
                                                         for u, v in conflicts),
                "max_candidates": max_candidates, "max_nonzeros": max_nonzeros,
                "complete_face_compression": "analytic; no separate Lean refinement verified",
                "statistical_scope": "point-estimate membership only; no updated bootstrap p-values"}
    metadata["within_budget_zero_cylinder_rows"] = int(metadata["within_budget_zero_cylinder_rows"])
    return ConsistencySystem(off, conflicts, candidates, matrix, metadata)


def build_from_conflict_graph(offsets: Sequence[int], adjacency: Sequence[Iterable[int]], **kwargs):
    """Adapter for warp_core.conflict_graph; validate symmetry before expansion."""
    neighbors = []
    for row in adjacency:
        entries = tuple(row)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in entries):
            raise ValueError("graph endpoints must be integer event indices")
        neighbors.append(frozenset(int(v) for v in entries))
    neighbors = tuple(neighbors)
    if not len(offsets) or int(offsets[-1]) != len(neighbors):
        raise ValueError("graph length must equal the final event offset")
    for u, row in enumerate(neighbors):
        for v in row:
            if not 0 <= v < len(neighbors) or u == v or u not in neighbors[v]:
                raise ValueError("conflict graph must be loopless, in-range and symmetric")
    return build_consistency_system(offsets, ((u, v) for u, row in enumerate(neighbors)
                                            for v in row if u < v), **kwargs)


def _input(system, pi):
    values = list(pi)
    point = np.asarray(values, dtype=float)
    if point.shape != (system.n_events,) or not np.isfinite(point).all():
        raise ValueError("one finite raw marginal estimate per event is required")
    # Keep the provided rational value or the exact stored binary-float value.
    exact = [Fraction(float(x)) if isinstance(x, (float, np.floating)) else Fraction(x)
             for x in values]
    b = np.concatenate(([1.], point, np.zeros(system.n_conflict_rows)))
    audit = {"negative_entries": int(np.count_nonzero(point < 0)),
             "above_one_entries": int(np.count_nonzero(point > 1)),
             "block_sum_max_error": max(abs(point[a:b].sum() - 1)
                                         for a, b in zip(system.offsets, system.offsets[1:])),
             "simplex_repair_applied": False,
             "rational_target_convention": "provided rationals or exact stored binary floats"}
    return point, exact, b, audit


def _exact_column_audit(system: ConsistencySystem, integers: Sequence[int]):
    minimum = None
    zeros = 0
    for column in range(system.n_candidates):
        begin, end = system.matrix.indptr[column:column + 2]
        value = sum(integers[int(row)] * int(coefficient)
                    for row, coefficient in zip(system.matrix.indices[begin:end], system.matrix.data[begin:end]))
        if value < 0:
            return None
        minimum = value if minimum is None else min(minimum, value)
        zeros += value == 0
    return {"all_unfiltered_columns_checked": system.n_candidates,
            "minimum_column_value_integer": minimum, "zero_columns": zeros,
            "exact_nonnegative_cancellation": True}


def rationalize_certificate(system: ConsistencySystem, dual: Sequence[float], pi: Sequence,
                            *, denominator_bounds=(1, 2, 4, 8, 16, 32, 64, 128, 256, 1000, 10000, 1000000),
                            max_lcm_bits: int = 4096) -> dict | None:
    """Try primitive integer Farkas coefficients and audit EVERY candidate exactly.

    This verifies a certificate, not exact feasibility of an approximate primal
    witness.  The negative affine value refers to the supplied digital estimate,
    not to an unknown population vector or sampling hypothesis.
    """
    y = np.asarray(dual, dtype=float)
    if y.shape != (system.matrix.shape[0],) or not np.isfinite(y).all():
        raise ValueError("dual certificate has incorrect shape or nonfinite coefficients")
    scale = np.max(np.abs(y))
    if not scale:
        return None
    normalized = y / scale
    _, exact_pi, _, _ = _input(system, pi)
    seen = set()
    for bound in denominator_bounds:
        fractions = [Fraction(float(value)).limit_denominator(bound) for value in normalized]
        denominator = 1
        for value in fractions:
            denominator = lcm(denominator, value.denominator)
            if denominator.bit_length() > max_lcm_bits:
                break
        if denominator.bit_length() > max_lcm_bits:
            continue
        integers = [value.numerator * (denominator // value.denominator) for value in fractions]
        divisor = reduce(gcd, (abs(value) for value in integers), 0)
        if not divisor:
            continue
        integers = [value // divisor for value in integers]
        key = tuple(integers)
        if key in seen:
            continue
        seen.add(key)
        affine = Fraction(integers[0]) + sum(
            (integers[1 + event] * exact_pi[event] for event in range(system.n_events)), Fraction(0))
        if affine >= 0:
            continue
        audit = _exact_column_audit(system, integers)
        if audit is None:
            continue
        return {"kind": "exact_primitive_integer_affine_H_certificate",
                "constant": integers[0], "event_coefficients": integers[1:1 + system.n_events],
                "conflict_moment_multipliers": integers[1 + system.n_events:],
                "aggregate_conflicts": system.metadata["aggregate_conflicts"],
                "inequality": "constant + sum(event_coefficients[e] * pi[e]) >= 0",
                "observed_affine_value_exact": str(affine),
                "observed_affine_value_float": float(affine),
                "strictly_violated_at_digital_point": True,
                "rationalization_denominator_bound": bound, "column_audit": audit,
                "meaning": "geometry-valid exact H certificate at the supplied digital estimate"}
    return None


def _primal_audit(system: ConsistencySystem, weights, target, tolerance):
    raw = np.asarray(weights, dtype=float)
    if raw.shape != (system.n_candidates,) or not np.isfinite(raw).all() or raw.min() < -tolerance:
        return None, {"audit_passed": False, "reason": "invalid numerical law"}
    law = np.maximum(raw, 0)
    discarded = 0.
    conflict_set = set(system.conflicts)
    # Only the returned support is checked/cleaned; the LP candidate catalogue
    # remains unfiltered.  Remove numerical dust on forbidden cylinders.
    for column in np.flatnonzero(law > 0):
        chosen = system.candidates[column]
        if any((min(int(u), int(v)), max(int(u), int(v))) in conflict_set
               for j, u in enumerate(chosen) for v in chosen[j + 1:]):
            discarded += law[column]
            law[column] = 0
    if discarded > tolerance or law.sum() <= 0:
        return None, {"audit_passed": False, "reason": "positive forbidden-path mass",
                      "discarded_conflict_dust": float(discarded)}
    before_normalization = float(law.sum())
    law /= before_normalization
    observed = np.asarray(system.matrix @ law).ravel()
    residual = observed - target
    marginal_error = float(np.max(np.abs(residual[1:1 + system.n_events])))
    conflict_error = float(np.max(np.abs(observed[1 + system.n_events:]), initial=0))
    passed = marginal_error <= tolerance and conflict_error == 0 and abs(law.sum() - 1) <= tolerance
    audit = {"audit_passed": bool(passed), "nonnegative_mass": bool(np.all(law >= 0)),
             "mass_sum": float(law.sum()), "marginal_max_abs_error": marginal_error,
             "conflict_cylinder_max_abs_mass": conflict_error,
             "all_conflict_cylinders_zero_after_audit": conflict_error == 0,
             "support_paths": int(np.count_nonzero(law)), "clipped_negative_entries": int(np.count_nonzero(raw < 0)),
             "discarded_conflict_dust": float(discarded),
             "witness_pre_normalization_mass": before_normalization,
             "target_repaired": False, "audit_tolerance": tolerance,
             "exact_membership_claim": False}
    return law if passed else None, audit


def check_rpm_consistency(system: ConsistencySystem, pi: Sequence, *, audit_tol: float = 2e-8,
                          time_limit: float = 30., include_witness: bool = False) -> dict:
    """Return feasible / infeasible / unknown with explicit numerical/exact audits."""
    if not np.isfinite([audit_tol, time_limit]).all() or min(audit_tol, time_limit) <= 0:
        raise ValueError("positive finite audit tolerance and LP time limit are required")
    started = monotonic()
    point, _, b, input_audit = _input(system, pi)
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9,
               "time_limit": float(time_limit)}
    result = {"status": "unknown", "system": system.metadata.copy(), "input_audit": input_audit,
              "interpretation": "point-estimate membership; not a statistical rejection or bootstrap retest",
              "implementation_refinement": "complete-face compression analytic; separate Lean refinement unverified"}
    primal = linprog(np.zeros(system.n_candidates), A_eq=system.matrix, b_eq=b,
                     bounds=(0, None), method="highs", options=options)
    result["primal_solver"] = {"status": int(primal.status), "message": primal.message}
    if primal.success:
        law, audit = _primal_audit(system, primal.x, b, audit_tol)
        result["primal_audit"] = audit
        if law is not None:
            result["status"] = "feasible"
            result["membership_evidence"] = "audited numerical nonnegative law; not exact rational feasibility"
            if include_witness:
                result["sparse_law"] = [{"candidate_index": int(h),
                                         "chosen_events": system.candidates[h].astype(int).tolist(),
                                         "mass": float(law[h])} for h in np.flatnonzero(law)]
            result["elapsed_seconds"] = monotonic() - started
            return result
    elif primal.status != 2:
        result["unknown_reason"] = "LP did not return an audited feasible or infeasible status"
        result["elapsed_seconds"] = monotonic() - started
        return result

    # Explicit Farkas search: M^T y >=0 and b*y <=-1, minimizing split-variable L1.
    # This is independent of any inaccessible solver infeasibility ray.
    rows = system.matrix.shape[0]
    transpose = system.matrix.T.tocsr()
    constraints = sparse.vstack([
        sparse.hstack([-transpose, transpose], format="csr"),
        sparse.csr_matrix(np.concatenate([b, -b])[None, :])], format="csr")
    rhs = np.concatenate([np.zeros(system.n_candidates), [-1.]])
    dual = linprog(np.ones(2 * rows), A_ub=constraints, b_ub=rhs,
                   bounds=(0, None), method="highs", options=options)
    result["dual_solver"] = {"status": int(dual.status), "message": dual.message}
    if not dual.success or not np.isfinite(dual.x).all():
        result["unknown_reason"] = "no successful explicit Farkas certificate solve"
        result["elapsed_seconds"] = monotonic() - started
        return result
    y = dual.x[:rows] - dual.x[rows:]
    scale = float(np.max(np.abs(y)))
    if not scale:
        result["unknown_reason"] = "zero separating vector"
        result["elapsed_seconds"] = monotonic() - started
        return result
    y /= scale
    column_values = np.asarray(system.matrix.T @ y).ravel()
    minimum, affine = float(column_values.min()), float(b @ y)
    numeric_passed = minimum >= -audit_tol and affine < -audit_tol
    result["dual_audit"] = {"audit_passed": bool(numeric_passed),
                            "all_unfiltered_columns_checked": system.n_candidates,
                            "minimum_column_value": minimum, "observed_affine_value": affine,
                            "normalization": "max absolute dual coefficient one", "audit_tolerance": audit_tol}
    certificate = rationalize_certificate(system, y, pi)
    if certificate is not None:
        result["status"] = "infeasible"
        result["membership_evidence"] = "exact geometry-valid affine H certificate at digital point"
        result["certificate"] = certificate
    elif numeric_passed:
        result["status"] = "infeasible"
        result["membership_evidence"] = "numerically audited Farkas certificate; exact rationalization unavailable"
        result["certificate"] = {"kind": "numerically_audited_affine_H_certificate",
                                 "constant": float(y[0]), "event_coefficients": y[1:1 + system.n_events].tolist(),
                                 "conflict_moment_multipliers": y[1 + system.n_events:].tolist(),
                                 "aggregate_conflicts": system.metadata["aggregate_conflicts"],
                                 "inequality": "constant + sum(event_coefficients[e] * pi[e]) >= 0",
                                 "observed_affine_value": affine, "exact_geometry_certificate": False}
    else:
        result["unknown_reason"] = "neither exact nor tolerance-audited Farkas certificate verified"
    result["elapsed_seconds"] = monotonic() - started
    return result
