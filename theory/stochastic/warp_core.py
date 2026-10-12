"""Finite off-intersection stochastic WARP geometry and auditable linear programs.

These routines are numerical, not Lean proofs.  Budgets are positive-price
hyperplanes in the nonnegative orthant.  Cross-budget intersections are excluded:
each patch has strictly positive or negative expenditure signs at every other
budget.  Exact duplicate normalized budgets are rejected unless merging is
explicitly requested.  Enumeration is exponential and stops at declared limits.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy
from scipy.optimize import linprog


LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9,
              "dual_feasibility_tolerance": 1e-9}
AUDIT_TOL = 2e-8


@dataclass(frozen=True)
class Patch:
    budget: int
    index: int
    signature: tuple[int, ...]
    representative: tuple[float, ...]
    strict_margin: float
    equality_residual: float
    margin_constraint_residual: float


@dataclass
class BudgetPatches:
    normalized_prices: np.ndarray
    patches: tuple[tuple[Patch, ...], ...]
    original_to_merged: tuple[int, ...]
    excluded_signatures: int
    solver_counts: dict[str, int]

    @property
    def n_budgets(self) -> int:
        return len(self.patches)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(np.cumsum([0] + [len(p) for p in self.patches]).tolist())

    @property
    def n_events(self) -> int:
        return self.offsets[-1]

    def to_dict(self) -> dict:
        return {
            "normalized_prices": self.normalized_prices.tolist(),
            "original_to_merged": list(self.original_to_merged),
            "patch_counts": [len(p) for p in self.patches],
            "excluded_signatures": self.excluded_signatures,
            "patch_lp_solver_counts": self.solver_counts,
            "patches": [[vars(p) for p in group] for group in self.patches],
            "no_budget_intersection_mass_assumption": True,
        }


def build_patches(
    prices: Sequence[Sequence[float]],
    wealth: Sequence[float],
    *,
    duplicate_policy: str = "reject",
    duplicate_tol: float = 1e-11,
    strict_margin_tol: float = 1e-8,
    max_signatures_per_budget: int = 4096,
) -> BudgetPatches:
    """Enumerate realizable strict signatures by maximizing their common margin.

    Patch order follows bit order among the other budgets in ascending order:
    for three budgets it is (--), (+-), (-+), (++), omitting unrealizable rows.
    ``merge`` reports the original-to-merged mapping; it does NOT pool data.
    """
    p = np.asarray(prices, dtype=float)
    w = np.asarray(wealth, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
        raise ValueError("Need at least two budgets and two goods.")
    if w.shape != (p.shape[0],) or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError("Finite prices and one finite wealth per budget required.")
    if (p <= 0).any() or (w <= 0).any():
        raise ValueError("Prices and wealth must be strictly positive.")
    if duplicate_policy not in {"reject", "merge"}:
        raise ValueError("duplicate_policy must be 'reject' or 'merge'.")
    if duplicate_tol < 0 or strict_margin_tol <= 0:
        raise ValueError("Tolerances must be nonnegative / positive respectively.")
    normalized = p / w[:, None]
    unique: list[np.ndarray] = []
    mapping: list[int] = []
    for j, row in enumerate(normalized):
        matches = [k for k, old in enumerate(unique)
                   if np.allclose(row, old, rtol=duplicate_tol, atol=duplicate_tol)]
        if matches:
            if duplicate_policy == "reject":
                raise ValueError(f"Budget {j} duplicates normalized budget {matches[0]}.")
            mapping.append(matches[0])
        else:
            mapping.append(len(unique))
            unique.append(row)
    a = np.asarray(unique)
    t, goods = a.shape
    if t < 2:
        raise ValueError("Merging left fewer than two distinct budgets.")
    if 2 ** (t - 1) > max_signatures_per_budget:
        raise ValueError("Signature enumeration limit exceeded; no approximation was run.")
    groups: list[tuple[Patch, ...]] = []
    excluded = 0
    statuses: dict[str, int] = {}
    for j in range(t):
        others = [k for k in range(t) if k != j]
        group: list[Patch] = []
        for bits in range(2 ** (t - 1)):
            signs = np.zeros(t, dtype=int)
            for bit, k in enumerate(others):
                signs[k] = 1 if bits & (1 << bit) else -1
            # signs[k] * (a[k] @ x - 1) >= delta.
            aub = np.asarray([np.r_[-signs[k] * a[k], 1.] for k in others])
            bub = np.asarray([-signs[k] for k in others], dtype=float)
            objective = np.r_[np.zeros(goods), -1.]
            aeq = np.asarray([np.r_[a[j], 0.]])
            result = linprog(objective, A_ub=aub, b_ub=bub,
                             A_eq=aeq, b_eq=[1.], bounds=(0, None),
                             method="highs", options=LP_OPTIONS)
            statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
            if result.status == 2:
                excluded += 1
                continue
            if not result.success:
                raise RuntimeError(f"Patch LP failed, budget {j}, signs {signs}: {result.message}")
            x = result.x[:-1]
            delta = float(result.x[-1])
            if delta <= strict_margin_tol:
                excluded += 1
                continue
            eq_res = abs(float(a[j] @ x - 1.))
            ub_res = max(0., float(np.max(aub @ result.x - bub)))
            actual_margin = float(np.min(signs[others] * (a[others] @ x - 1.)))
            if eq_res > AUDIT_TOL or ub_res > AUDIT_TOL or actual_margin <= strict_margin_tol:
                raise RuntimeError("Patch witness failed independent residual audit.")
            group.append(Patch(j, len(group), tuple(int(s) for s in signs),
                               tuple(float(z) for z in x), actual_margin,
                               eq_res, ub_res))
        if not group:
            raise RuntimeError(f"No strict patches found on budget {j}.")
        groups.append(tuple(group))
    return BudgetPatches(a, tuple(groups), tuple(mapping), excluded, statuses)


def mutually_affordable(a: Patch, b: Patch) -> bool:
    return a.signature[b.budget] == -1 and b.signature[a.budget] == -1


def revealed_graph(system: BudgetPatches, choice: Sequence[int]) -> np.ndarray:
    """i -> j means the chosen bundle at j is strictly affordable at budget i."""
    n = len(choice)
    graph = np.zeros((n, n), dtype=bool)
    for j, patch_index in enumerate(choice):
        patch = system.patches[j][patch_index]
        for i in range(n):
            graph[i, j] = i != j and patch.signature[i] == -1
    return graph


def has_cycle(graph: np.ndarray) -> bool:
    indegree = graph.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    removed = 0
    while ready:
        i = ready.pop()
        removed += 1
        for j in np.flatnonzero(graph[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(int(j))
    return removed != graph.shape[0]


def enumerate_types(
    system: BudgetPatches,
    *,
    axiom: str = "WARP",
    allowed_patches: Sequence[Sequence[int]] | None = None,
    max_types: int = 1_000_000,
    max_nodes: int = 10_000_000,
) -> tuple[tuple[int, ...], ...]:
    """Backtrack complete choices, pruning pair conflicts and (for SARP) cycles.

    Restricting to patches with positive target probabilities is exact for a
    point-probability feasibility problem.  It is generally NOT exact for bands.
    """
    if axiom not in {"WARP", "SARP"}:
        raise ValueError("axiom must be WARP or SARP.")
    if max_types < 1 or max_nodes < 1:
        raise ValueError("Enumeration limits must be positive.")
    allowed = ([list(range(len(group))) for group in system.patches]
               if allowed_patches is None else [list(s) for s in allowed_patches])
    if len(allowed) != system.n_budgets:
        raise ValueError("One allowed-patch list per budget required.")
    for j, ids in enumerate(allowed):
        if len(set(ids)) != len(ids) or any(i < 0 or i >= len(system.patches[j]) for i in ids):
            raise ValueError("Invalid or duplicated allowed patch index.")
    result: list[tuple[int, ...]] = []
    nodes = 0
    choice: list[int] = []

    def visit() -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("Type search node limit exceeded; result is not complete.")
        j = len(choice)
        if j == system.n_budgets:
            result.append(tuple(choice))
            if len(result) > max_types:
                raise RuntimeError("Type count limit exceeded; result is not complete.")
            return
        for idx in allowed[j]:
            patch = system.patches[j][idx]
            if any(mutually_affordable(patch, system.patches[k][choice[k]]) for k in range(j)):
                continue
            choice.append(idx)
            if axiom == "WARP" or not has_cycle(revealed_graph(system, choice)):
                visit()
            choice.pop()

    visit()
    return tuple(result)


def type_matrix(system: BudgetPatches, types: Sequence[Sequence[int]]) -> np.ndarray:
    matrix = np.zeros((system.n_events, len(types)))
    for h, choice in enumerate(types):
        if len(choice) != system.n_budgets:
            raise ValueError("Types must select exactly one patch per budget.")
        for j, idx in enumerate(choice):
            if idx < 0 or idx >= len(system.patches[j]):
                raise ValueError("Invalid type patch index.")
            matrix[system.offsets[j] + idx, h] = 1.
    return matrix


def warp_only_indicator(system: BudgetPatches, types: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray([float(has_cycle(revealed_graph(system, choice))) for choice in types])


def flatten_probabilities(system: BudgetPatches, probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    if len(probabilities) != system.n_budgets:
        raise ValueError("One probability row per distinct budget required.")
    rows = []
    for group, row in zip(system.patches, probabilities):
        values = np.asarray(row, dtype=float)
        if values.shape != (len(group),) or not np.isfinite(values).all():
            raise ValueError("Probability row has wrong size or nonfinite entries.")
        if (values < 0).any() or (values > 1).any() or abs(values.sum() - 1.) > 1e-10:
            raise ValueError("Each probability row must be nonnegative and sum to one.")
        rows.append(values)
    return np.concatenate(rows)


def conflict_graph(system: BudgetPatches) -> tuple[frozenset[int], ...]:
    """Within-budget cliques plus cross-budget mutual-affordability edges."""
    patches = [p for group in system.patches for p in group]
    neighbors: list[set[int]] = [set() for _ in patches]
    for a in range(len(patches)):
        for b in range(a):
            if patches[a].budget == patches[b].budget or mutually_affordable(patches[a], patches[b]):
                neighbors[a].add(b)
                neighbors[b].add(a)
    return tuple(frozenset(s) for s in neighbors)


def integer_capacity(system: BudgetPatches, types: Sequence[Sequence[int]],
                     event_weights: Sequence[int]) -> dict:
    """Exact integer maximum over the supplied COMPLETE independent transversals.

    The type set must have been exhaustively enumerated on the intended support.
    An unconstrained independent-set capacity is not silently substituted.
    """
    values = list(event_weights)
    if len(values) != system.n_events or any(isinstance(z, bool) or not isinstance(z, (int, np.integer))
                                            or z < 0 for z in values):
        raise ValueError("One nonnegative integer event weight per patch required.")
    if not types:
        raise ValueError("No complete deterministic types exist on this support.")
    scores = [sum(int(values[system.offsets[j] + idx]) for j, idx in enumerate(choice))
              for choice in types]
    best = max(scores)
    argmax = scores.index(best)
    return {"capacity": best, "maximizing_type_zero_based": list(types[argmax]),
            "capacity_domain": "supplied complete independent transversals"}


def graph_independent_capacity(graph: Sequence[Iterable[int]], event_weights: Sequence[int],
                               *, max_nodes: int = 2_000_000) -> dict:
    """Audit-only exact weighted independent-set capacity, allowing incomplete paths.

    This can exceed complete-type capacity and is NOT the stochastic WARP test.
    Zero-weight vertices may be omitted without affecting the objective.
    """
    weights = [int(x) for x in event_weights]
    if len(weights) != len(graph) or any(x < 0 or int(x) != x for x in event_weights):
        raise ValueError("Graph requires nonnegative integer vertex weights.")
    adjacency = [sum(1 << int(j) for j in neighbors) for neighbors in graph]
    active = sum(1 << i for i, w in enumerate(weights) if w > 0)
    best = 0
    best_vertices: tuple[int, ...] = ()
    nodes = 0

    def visit(mask: int, score: int, selected: tuple[int, ...]) -> None:
        nonlocal best, best_vertices, nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("Independent-set audit limit exceeded.")
        vertices = [i for i in range(len(weights)) if mask & (1 << i)]
        if score + sum(weights[i] for i in vertices) <= best:
            return
        if not vertices:
            if score > best:
                best, best_vertices = score, selected
            return
        v = max(vertices, key=lambda i: (adjacency[i] & mask).bit_count())
        remaining = mask & ~(1 << v)
        visit(remaining & ~adjacency[v], score + weights[v], selected + (v,))
        visit(remaining, score, selected)

    visit(active, 0, ())
    return {"capacity": best, "maximizing_vertices_zero_based": list(best_vertices),
            "search_nodes": nodes, "capacity_domain": "all independent sets, possibly incomplete"}


def _audit_lp(result, c: np.ndarray, *, aeq=None, beq=None, aub=None, bub=None, bounds=None) -> dict:
    """Independent primal, dual, stationarity, and complementarity checks."""
    report = {"status": int(result.status), "success": bool(result.success),
              "message": str(result.message), "iterations": int(result.nit or 0),
              "crossover_iterations": int(getattr(result, "crossover_nit", 0) or 0)}
    if not result.success:
        return report
    x = result.x
    bound_list = [(0., None)] * len(c) if bounds is None else list(bounds)
    equality = 0. if aeq is None else float(np.max(np.abs(aeq @ x - beq), initial=0.))
    inequality = 0. if aub is None else max(0., float(np.max(aub @ x - bub, initial=0.)))
    stationarity = c.copy()
    dual_objective = 0.
    complementarity = 0.
    if aeq is not None:
        stationarity -= aeq.T @ result.eqlin.marginals
        dual_objective += float(beq @ result.eqlin.marginals)
    if aub is not None:
        stationarity -= aub.T @ result.ineqlin.marginals
        dual_objective += float(bub @ result.ineqlin.marginals)
        complementarity = float(np.max(np.abs((aub @ x - bub) * result.ineqlin.marginals), initial=0.))
    stationarity -= result.lower.marginals + result.upper.marginals
    bound_violation = 0.
    for i, (lo, hi) in enumerate(bound_list):
        if lo is not None:
            bound_violation = max(bound_violation, float(lo - x[i]))
            dual_objective += float(lo * result.lower.marginals[i])
            complementarity = max(complementarity, abs(float((x[i] - lo) * result.lower.marginals[i])))
        if hi is not None:
            bound_violation = max(bound_violation, float(x[i] - hi))
            dual_objective += float(hi * result.upper.marginals[i])
            complementarity = max(complementarity, abs(float((x[i] - hi) * result.upper.marginals[i])))
    metrics = {"equality_max_abs": equality, "inequality_max_violation": inequality,
               "bound_max_violation": max(0., bound_violation),
               "stationarity_max_abs": float(np.max(np.abs(stationarity), initial=0.)),
               "complementarity_max_abs": complementarity,
               "primal_dual_objective_gap_abs": abs(float(c @ x) - dual_objective)}
    report.update(metrics)
    report["audit_passed"] = max(metrics.values()) <= AUDIT_TOL
    if not report["audit_passed"]:
        raise RuntimeError(f"LP numerical certificate failed: {report}")
    return report


def _solve_lp(c, *, aeq=None, beq=None, aub=None, bub=None, bounds=None):
    c = np.asarray(c, dtype=float)
    aeq = None if aeq is None else np.asarray(aeq, dtype=float)
    beq = None if beq is None else np.asarray(beq, dtype=float)
    aub = None if aub is None else np.asarray(aub, dtype=float)
    bub = None if bub is None else np.asarray(bub, dtype=float)
    result = linprog(c, A_eq=aeq, b_eq=beq, A_ub=aub, b_ub=bub,
                     bounds=(0, None) if bounds is None else bounds,
                     method="highs", options=LP_OPTIONS)
    audit = _audit_lp(result, c, aeq=aeq, beq=beq, aub=aub, bub=bub, bounds=bounds)
    if result.status not in {0, 2}:
        raise RuntimeError(f"LP failed without a feasibility decision: {result.message}")
    return result, audit


def _sparse_mixture(types: Sequence[Sequence[int]], values: np.ndarray) -> list[dict]:
    return [{"type_zero_based": list(types[h]), "mass": float(value)}
            for h, value in enumerate(values) if value > 1e-10]


def mixture_feasibility(system: BudgetPatches, types: Sequence[Sequence[int]],
                        probabilities: Sequence[Sequence[float]]) -> dict:
    pi = flatten_probabilities(system, probabilities)
    matrix = type_matrix(system, types)
    if not types:
        return {"feasible": False, "reason": "no complete types on supplied support"}
    aeq = np.vstack([matrix, np.ones(len(types))])
    beq = np.r_[pi, 1.]
    result, audit = _solve_lp(np.zeros(len(types)), aeq=aeq, beq=beq)
    report = {"feasible": bool(result.success), "lp_audit": audit}
    if result.success:
        report["mixture"] = _sparse_mixture(types, result.x)
    else:
        report["separator"] = separating_weights(system, types, probabilities)
    return report


def separating_weights(system: BudgetPatches, types: Sequence[Sequence[int]],
                       probabilities: Sequence[Sequence[float]]) -> dict:
    """Find bounded real weights violating the support inequality, if possible.

    Nonnegative weights suffice because every block of probabilities and every
    deterministic type sum to one.  This is a numerical LP certificate, not an
    exact integer/rational certificate; explicit integer weights use capacity().
    """
    pi = flatten_probabilities(system, probabilities)
    matrix = type_matrix(system, types)
    if not types:
        raise ValueError("Separator routine needs at least one complete type.")
    # max w @ pi - t, subject to t >= w @ every deterministic column.
    aub = np.c_[matrix.T, -np.ones(len(types))]
    objective = np.r_[-pi, 1.]
    bounds = [(0., 1.)] * len(pi) + [(0., None)]
    result, audit = _solve_lp(objective, aub=aub, bub=np.zeros(len(types)), bounds=bounds)
    weights = result.x[:-1]
    capacity = float(np.max(weights @ matrix))
    expected = float(weights @ pi)
    return {"event_weights": weights.tolist(), "capacity": capacity,
            "observed_score": expected, "violation": expected - capacity,
            "separates_at_audit_tolerance": expected - capacity > AUDIT_TOL,
            "certificate_kind": "numerical real nonnegative weights", "lp_audit": audit}


def sharp_warp_only_mass(system: BudgetPatches, warp_types: Sequence[Sequence[int]],
                         probabilities: Sequence[Sequence[float]]) -> dict:
    """Sharp identified bounds on mass of types that satisfy WARP but fail SARP."""
    pi = flatten_probabilities(system, probabilities)
    matrix = type_matrix(system, warp_types)
    if not warp_types:
        return {"feasible": False, "lower": None, "upper": None}
    aeq = np.vstack([matrix, np.ones(len(warp_types))])
    beq = np.r_[pi, 1.]
    cyclic = warp_only_indicator(system, warp_types)
    low, low_audit = _solve_lp(cyclic, aeq=aeq, beq=beq)
    if not low.success:
        return {"feasible": False, "lower": None, "upper": None, "lower_lp_audit": low_audit}
    high, high_audit = _solve_lp(-cyclic, aeq=aeq, beq=beq)
    return {"feasible": True, "lower": float(cyclic @ low.x), "upper": float(cyclic @ high.x),
            "lower_mixture": _sparse_mixture(warp_types, low.x),
            "upper_mixture": _sparse_mixture(warp_types, high.x),
            "lower_lp_audit": low_audit, "upper_lp_audit": high_audit}


def probability_band_mass_bound(system: BudgetPatches, warp_types: Sequence[Sequence[int]],
                                lower: Sequence[Sequence[float]], upper: Sequence[Sequence[float]]) -> dict:
    """Minimize WARP-only mass over all probabilities in simultaneous bands."""
    def flatten_intervals(rows):
        if len(rows) != system.n_budgets:
            raise ValueError("One interval row per budget required.")
        values = []
        for group, row in zip(system.patches, rows):
            array = np.asarray(row, dtype=float)
            if array.shape != (len(group),) or not np.isfinite(array).all():
                raise ValueError("Interval row has wrong size or nonfinite entries.")
            values.append(array)
        return np.concatenate(values)
    lo, hi = flatten_intervals(lower), flatten_intervals(upper)
    if (lo < 0).any() or (hi > 1).any() or (lo > hi).any():
        raise ValueError("Need 0 <= lower <= upper <= 1.")
    for j in range(system.n_budgets):
        section = slice(system.offsets[j], system.offsets[j + 1])
        if lo[section].sum() > 1. + 1e-10 or hi[section].sum() < 1. - 1e-10:
            raise ValueError("Probability intervals exclude the budget's simplex.")
    if not warp_types:
        return {"band_warp_feasible": False, "lower": None}
    matrix = type_matrix(system, warp_types)
    cyclic = warp_only_indicator(system, warp_types)
    aub = np.vstack([matrix, -matrix])
    bub = np.r_[hi, -lo]
    result, audit = _solve_lp(cyclic, aeq=np.ones((1, len(warp_types))), beq=[1.], aub=aub, bub=bub)
    report = {"band_warp_feasible": bool(result.success), "lower": None, "lp_audit": audit}
    if result.success:
        report["lower"] = float(cyclic @ result.x)
        report["minimizing_mixture"] = _sparse_mixture(warp_types, result.x)
        reconstructed = matrix @ result.x
        report["minimizing_probabilities"] = [reconstructed[system.offsets[j]:system.offsets[j + 1]].tolist()
                                              for j in range(system.n_budgets)]
    return report


def hoeffding_mass_bound(system: BudgetPatches, warp_types: Sequence[Sequence[int]],
                         counts: Sequence[Sequence[int]], *, alpha: float = .05) -> dict:
    """Conservative simultaneous bands; requires iid sampled units within budget.

    No independence across budgets is needed for the union bound.  Repeated
    transactions from one customer are NOT iid sampled customers and cannot be
    passed off as satisfying this condition.  Budget distributions must target
    the same population with observed, fixed common budgets/conditioning cells.
    """
    if not 0 < alpha < 1 or len(counts) != system.n_budgets:
        raise ValueError("Need alpha in (0,1) and one count row per budget.")
    empirical, lo, hi, totals, radii = [], [], [], [], []
    for group, row in zip(system.patches, counts):
        values = list(row)
        if len(values) != len(group) or any(isinstance(z, bool) or not isinstance(z, (int, np.integer))
                                           or z < 0 for z in values):
            raise ValueError("One nonnegative integer count per patch required.")
        total = sum(int(z) for z in values)
        if total <= 0:
            raise ValueError("Every budget needs a positive sample size.")
        probability = np.asarray(values, dtype=float) / total
        radius = math.sqrt(math.log(2 * system.n_events / alpha) / (2 * total))
        empirical.append(probability.tolist())
        lo.append(np.maximum(0., probability - radius).tolist())
        hi.append(np.minimum(1., probability + radius).tolist())
        totals.append(total)
        radii.append(radius)
    report = probability_band_mass_bound(system, warp_types, lo, hi)
    report.update({"alpha": alpha, "simultaneous_coverage_at_least": 1. - alpha,
                   "method": "coordinate Hoeffding bounds with union bound over all patches",
                   "sample_sizes": totals, "radii": radii, "empirical_probabilities": empirical,
                   "probability_lower": lo, "probability_upper": hi,
                   "sampling_scope": "iid sampled units within each fixed budget; cross-budget dependence allowed",
                   "not_valid_for": "unadjusted dependent transactions, changing budget cells, unmodeled measurement error",
                   "interpretation": "lower confidence bound on WARP-only type mass conditional on population WARP; "
                                     "if bands admit no WARP mixture, reject stochastic WARP at level alpha"})
    return report


def analyze_input(payload: dict) -> dict:
    system = build_patches(payload["prices"], payload["wealth"],
                           duplicate_policy=payload.get("duplicate_policy", "reject"))
    # After explicit merging a caller must supply rows for the DISTINCT budgets.
    warp = enumerate_types(system, axiom="WARP", max_types=payload.get("max_types", 1_000_000),
                           max_nodes=payload.get("max_nodes", 10_000_000))
    report = {"geometry": system.to_dict(), "n_warp_types": len(warp),
              "n_sarp_types": int(len(warp) - warp_only_indicator(system, warp).sum())}
    if "probabilities" in payload:
        report["point_feasibility"] = mixture_feasibility(system, warp, payload["probabilities"])
        report["sharp_warp_only_mass"] = sharp_warp_only_mass(system, warp, payload["probabilities"])
    if "event_weights" in payload:
        report["integer_capacity"] = integer_capacity(system, warp, payload["event_weights"])
    if "counts" in payload:
        report["confidence_mass_lower_bound"] = hoeffding_mass_bound(system, warp, payload["counts"],
                                                                     alpha=payload.get("alpha", .05))
    return report
