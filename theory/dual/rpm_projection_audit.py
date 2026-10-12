"""Independent numerical projection certificates from observed RPM conflicts.

The supplied mixture columns are *only a primal feasibility witness*: their
completeness is neither assumed nor needed. Global support is maximized instead
over binary patch variables, one per budget, with every conflict pair forbidden.
This is the finite price-sign graph representation of complete WARP selections.

For the normalized tightened null K_tau = tau*mean + (1-tau)*K, support is
tau*n@mean + (1-tau)*max_{a in K} n@a. We require the MILP's optimal status and
dual objective bound, not merely an incumbent. These are floating-point solver
audits, not exact arithmetic, statistical validity, or formal Lean certificates.
No estimation, resampling, clipping of estimated probabilities, or retuning is
performed here. In particular, changing representation must not change a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


class ProjectionAuditError(RuntimeError):
    """No certificate is returned on a solver limit or failed numerical audit."""

    def __init__(self, message: str, audit: dict | None = None):
        super().__init__(message)
        self.audit = {} if audit is None else audit


@dataclass(frozen=True)
class PreparedMixture:
    """Verified complete conflict-free columns, expressed as local patch IDs."""

    incidence: sparse.csc_matrix
    mean: np.ndarray
    columns: int


class ConflictGraphProjectionAuditor:
    """Audit an existing hull projection without enumerating the graph's types.

    ``offsets`` starts at zero and ends at the number of observed patches.
    ``conflicts`` contains unordered pairs of global patch-coordinate indices.
    Within-budget pairs may be supplied but are redundant with block equality.
    ``types`` has shape (H, budgets), with *local* patch indices, and supplies a
    possibly incomplete primal witness. Its uniform mean must match ``mean``.

    ``weights`` passed to :meth:`audit` are the original full mixture weights:
    sum(weights)=1 and, when tau>0, weights>=tau/H. This preserves exactly the
    previous tightening center. The method does not use filtered columns to
    maximize the supporting normal, so omitted admissible selections are detected.
    """

    def __init__(self, offsets: Sequence[int], conflicts: Sequence[Sequence[int]],
                 *, types: np.ndarray | None = None, mean: Sequence[float] | None = None,
                 feasibility_tol: float = 2e-10, gap_tol: float = 2e-10,
                 time_limit_seconds: float = 30.):
        raw_offsets = np.asarray(offsets)
        if (raw_offsets.ndim != 1 or len(raw_offsets) < 2
                or not np.issubdtype(raw_offsets.dtype, np.integer)
                or raw_offsets[0] != 0 or np.any(np.diff(raw_offsets) <= 0)):
            raise ValueError("Offsets must be increasing integers beginning at zero.")
        if (not np.isfinite([feasibility_tol, gap_tol, time_limit_seconds]).all()
                or min(feasibility_tol, gap_tol, time_limit_seconds) <= 0):
            raise ValueError("Audit tolerances and solver time limit must be positive.")
        self.offsets = tuple(int(x) for x in raw_offsets)
        self.events = self.offsets[-1]
        self.budgets = len(self.offsets) - 1
        self.feasibility_tol = float(feasibility_tol)
        self.gap_tol = float(gap_tol)
        self.time_limit_seconds = float(time_limit_seconds)
        pairs = np.asarray(conflicts)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        if (pairs.ndim != 2 or pairs.shape[1] != 2
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.any(pairs < 0) or np.any(pairs >= self.events)
                or np.any(pairs[:, 0] == pairs[:, 1])):
            raise ValueError("Conflicts must be distinct in-range integer patch pairs.")
        self.conflicts = np.unique(np.sort(pairs.astype(np.int64), axis=1), axis=0)
        self.conflicts.flags.writeable = False
        rows, columns = [], []
        for t, (lo, hi) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            rows.extend([t] * (hi - lo))
            columns.extend(range(lo, hi))
        for edge, (u, v) in enumerate(self.conflicts):
            rows.extend([self.budgets + edge] * 2)
            columns.extend([int(u), int(v)])
        matrix = sparse.csc_matrix((np.ones(len(rows)), (rows, columns)),
                                  shape=(self.budgets + len(self.conflicts), self.events))
        lower = np.r_[np.ones(self.budgets), np.full(len(self.conflicts), -np.inf)]
        upper = np.ones(self.budgets + len(self.conflicts))
        self.constraints = LinearConstraint(matrix, lower, upper)
        self.bounds = Bounds(np.zeros(self.events), np.ones(self.events))
        self.integrality = np.ones(self.events, dtype=np.int32)
        self._prepared: list[PreparedMixture] = []
        if types is None and mean is not None:
            raise ValueError("A mean without a primal mixture witness is not certified.")
        self.mixture = None if types is None else self.prepare_mixture(types, mean=mean)

    def prepare_mixture(self, types: np.ndarray, *,
                        mean: Sequence[float] | None = None) -> PreparedMixture:
        """Check supplied columns geometrically, without claiming completeness."""
        choices = np.asarray(types)
        counts = np.diff(self.offsets)
        if (choices.ndim != 2 or choices.shape[1] != self.budgets
                or not choices.shape[0] or not np.issubdtype(choices.dtype, np.integer)
                or np.any(choices < 0) or np.any(choices >= counts[None, :])):
            raise ValueError("Mixture types must contain one valid local patch per budget.")
        choices = choices.astype(np.int64, copy=True)
        for u, v in self.conflicts:
            bu = int(np.searchsorted(self.offsets, u, side="right") - 1)
            bv = int(np.searchsorted(self.offsets, v, side="right") - 1)
            if np.any((choices[:, bu] == u - self.offsets[bu])
                      & (choices[:, bv] == v - self.offsets[bv])):
                raise ValueError("A supplied mixture column selects a forbidden conflict pair.")
        H = len(choices)
        indices = (choices + np.asarray(self.offsets[:-1])).ravel()
        columns = np.repeat(np.arange(H), self.budgets)
        incidence = sparse.csc_matrix((np.ones(len(indices)), (indices, columns)),
                                     shape=(self.events, H))
        expected_mean = np.asarray(incidence.mean(axis=1)).ravel()
        if mean is not None:
            supplied_mean = np.asarray(mean, dtype=float)
            if (supplied_mean.shape != (self.events,)
                    or not np.isfinite(supplied_mean).all()
                    or np.max(np.abs(supplied_mean - expected_mean)) > self.feasibility_tol):
                raise ValueError("The tightening mean differs from the original witness mean.")
            expected_mean = supplied_mean.copy()
        for array in (incidence.data, incidence.indices, incidence.indptr, expected_mean):
            array.flags.writeable = False
        prepared = PreparedMixture(incidence, expected_mean, H)
        self._prepared.append(prepared)
        return prepared

    def audit(self, target: Sequence[float], projected: Sequence[float], *,
              weights: Sequence[float], tau: float = 0.,
              mixture: PreparedMixture | None = None) -> dict:
        """Certify a numerical primal witness and independent support upper bound.

        For n=target-projected, a feasible projected point is a projection exactly
        when support_{K_tau}(n)-n@projected<=0. The returned upper gap uses the
        MILP dual bound. A failed primal or support check raises instead of being
        relabeled a successful projection, or discarded from a bootstrap test.
        """
        witness = self.mixture if mixture is None else mixture
        if witness is None:
            raise ValueError("Prepare a primal mixture witness before auditing.")
        if not any(witness is prepared for prepared in self._prepared):
            raise ValueError("The mixture witness was not verified against this conflict graph.")
        if witness.incidence.shape != (self.events, witness.columns):
            raise ValueError("Prepared mixture dimensions differ from this graph.")
        point, q = np.asarray(target, dtype=float), np.asarray(projected, dtype=float)
        w = np.asarray(weights, dtype=float)
        if (point.shape != (self.events,) or q.shape != point.shape
                or w.shape != (witness.columns,) or not np.isfinite(point).all()
                or not np.isfinite(q).all() or not np.isfinite(w).all()):
            raise ValueError("Finite target, projection and full mixture weights are required.")
        if not np.isfinite(tau) or not 0 <= tau < 1:
            raise ValueError("tau must lie in [0, 1).")
        tau = float(tau)
        contraction = 1. - tau
        free = (w - tau / witness.columns) / contraction
        implied = tau * witness.mean + contraction * (witness.incidence @ free)
        equality_error = abs(float(free.sum()) - 1.)
        negative_error = max(0., -float(free.min()))
        marginal_error = float(np.max(np.abs(implied - q)))
        block_error = max(abs(float(q[lo:hi].sum()) - 1.)
                          for lo, hi in zip(self.offsets[:-1], self.offsets[1:]))
        audit = {"certificate_kind": "floating-point numerical primal and independent MILP support audit",
                 "not_an_exact_or_statistical_certificate": True,
                 "support_uses_type_enumeration": False,
                 "witness_enumeration_completeness_assumed": False,
                 "events": self.events, "budgets": self.budgets,
                 "conflict_pairs": len(self.conflicts), "witness_columns": witness.columns,
                 "positive_free_witness_columns": int(np.count_nonzero(free > 0)),
                 "tau": tau, "free_mixture_mass_error": equality_error,
                 "free_mixture_nonnegativity_violation": negative_error,
                 "affine_mixture_marginal_max_error": marginal_error,
                 "budget_adding_up_max_error": block_error,
                 "feasibility_tolerance": self.feasibility_tol,
                 "gap_tolerance": self.gap_tol, "all_witness_columns_conflict_free": True,
                 "tightened_set": "tau*original_mean + (1-tau)*conv(complete conflict-free selections)"}
        if max(equality_error, negative_error, marginal_error, block_error) > self.feasibility_tol:
            raise ProjectionAuditError("The supplied projected point lacks the claimed primal witness.", audit)
        normal = point - q
        if not np.isfinite(normal).all():
            raise ValueError("The target-minus-projection normal must remain finite.")
        normal_scale = float(np.max(np.abs(normal)))
        if normal_scale == 0.:
            free_upper = free_incumbent = 0.
            audit.update(milp_status=0, milp_message="Zero normal: analytic zero support.",
                         milp_optimal=True, milp_dual_bound_used=True, milp_node_count=0,
                         milp_relative_gap=0., milp_called=False)
        else:
            solution = milp(-normal / normal_scale, integrality=self.integrality,
                            bounds=self.bounds, constraints=self.constraints,
                            options={"presolve": True, "mip_rel_gap": 0.,
                                     "time_limit": self.time_limit_seconds})
            status = int(solution.status)
            audit.update(milp_status=status, milp_message=str(solution.message),
                         milp_optimal=bool(solution.success and status == 0), milp_called=True)
            dual = getattr(solution, "mip_dual_bound", None)
            if (not solution.success or status != 0 or dual is None
                    or not np.isfinite(dual) or solution.x is None):
                raise ProjectionAuditError("MILP did not return an optimal finite dual support bound.", audit)
            candidate = np.asarray(solution.x, dtype=float)
            binary = np.rint(candidate)
            block = [binary[lo:hi].sum() for lo, hi in zip(self.offsets[:-1], self.offsets[1:])]
            pair_violation = float(np.max(binary[self.conflicts[:, 0]]
                                         + binary[self.conflicts[:, 1]] - 1., initial=0.))
            if (not np.isfinite(candidate).all()
                    or np.max(np.abs(candidate - binary)) > self.feasibility_tol
                    or np.any((binary < 0) | (binary > 1))
                    or not np.all(np.asarray(block) == 1) or pair_violation > 0):
                raise ProjectionAuditError("MILP incumbent failed the independent binary graph audit.", audit)
            # SciPy minimizes -n@a. Its dual lower bound therefore gives an
            # upper bound for the required maximization, after restoring scale.
            free_upper = -float(dual) * normal_scale
            free_incumbent = float(normal @ binary)
            if free_upper < free_incumbent - self.gap_tol:
                raise ProjectionAuditError("MILP dual support bound is below a feasible incumbent.", audit)
            audit.update(milp_dual_bound_used=True,
                         milp_minimization_scaled_dual_bound=float(dual),
                         milp_node_count=int(getattr(solution, "mip_node_count", 0)),
                         milp_relative_gap=float(getattr(solution, "mip_gap", 0.)))
        center_score = float(normal @ witness.mean)
        projected_score = float(normal @ q)
        support_upper = tau * center_score + contraction * free_upper
        signed_gap = support_upper - projected_score
        gap = max(0., signed_gap)
        # A nonnegative normalized proxy makes the tiny primal residual explicit.
        # It is used only for the numerical error bound, never as a changed fit.
        proxy_free = np.maximum(free, 0.)
        proxy_free /= proxy_free.sum()
        proxy = tau * witness.mean + contraction * (witness.incidence @ proxy_free)
        delta = float(np.linalg.norm(proxy - q))
        objective_error = 2. * gap + 2. * float(np.linalg.norm(normal)) * delta + delta ** 2
        audit.update(normal_scale=normal_scale, supporting_normal=normal.tolist(),
                     unshifted_support_upper=free_upper,
                     unshifted_support_incumbent=free_incumbent,
                     tightening_center_score=center_score,
                     tightened_support_upper=support_upper,
                     projected_normal_score=projected_score,
                     signed_global_kkt_gap_upper=signed_gap,
                     global_kkt_gap_upper=gap,
                     primal_proxy_l2_residual=delta,
                     squared_distance=float(normal @ normal),
                     squared_objective_error_upper_bound=objective_error,
                     audit_passed=bool(gap <= self.gap_tol))
        if not audit["audit_passed"]:
            raise ProjectionAuditError("Independent graph support finds a nonzero projection KKT gap.", audit)
        return audit
