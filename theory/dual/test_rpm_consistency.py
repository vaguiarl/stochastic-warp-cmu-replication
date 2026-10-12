"""Primitive law tests; old filtered V-LPs appear ONLY as small validation oracles."""

from fractions import Fraction as Q
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

import rpm_consistency
from rpm_consistency import (CandidateLimit, build_consistency_system, build_from_conflict_graph,
                             check_rpm_consistency, rationalize_certificate)


def supported_triangle(*, aggregate=True):
    # E0,F0; E1,F1; E2,F2. E events pairwise conflict, F events do not.
    return build_consistency_system([0, 2, 4, 6], [(0, 2), (0, 4), (2, 4),
                                                   (0, 1), (2, 3), (4, 5)],
                                    aggregate_conflicts=aggregate)


def exact_certificate_check(system, result, pi):
    certificate = result["certificate"]
    assert certificate["kind"] == "exact_primitive_integer_affine_H_certificate"
    y = [certificate["constant"], *certificate["event_coefficients"],
         *certificate["conflict_moment_multipliers"]]
    assert len(y) == system.matrix.shape[0]
    for h in range(system.n_candidates):
        rows = system.matrix.indices[system.matrix.indptr[h]:system.matrix.indptr[h + 1]]
        data = system.matrix.data[system.matrix.indptr[h]:system.matrix.indptr[h + 1]]
        assert sum(y[int(row)] * int(a) for row, a in zip(rows, data)) >= 0
    assert Q(y[0]) + sum(Q(a) * Q(value) for a, value in zip(y[1:1 + system.n_events], pi)) < 0
    assert certificate["column_audit"]["all_unfiltered_columns_checked"] == system.n_candidates


def test_all_candidates_are_unfiltered_and_badness_is_an_integer_count():
    system = supported_triangle()
    assert system.n_candidates == 8
    assert system.candidates.tolist() == list(map(list, itertools.product((0, 1), (2, 3), (4, 5))))
    assert system.metadata["warp_filtered_columns"] is False
    assert system.matrix[-1].toarray().ravel().tolist() == [3, 1, 1, 0, 1, 0, 0, 0]
    assert system.n_conflict_rows == 1
    assert not system.candidates.flags.writeable
    assert not system.matrix.data.flags.writeable


def test_K3_supported_E_point_has_an_exact_geometry_valid_H_certificate():
    system = supported_triangle()
    pi = [Q(1, 2)] * 6
    result = check_rpm_consistency(system, pi)
    assert result["status"] == "infeasible"
    assert result["dual_audit"]["all_unfiltered_columns_checked"] == 8
    exact_certificate_check(system, result, pi)
    assert "bootstrap" in result["interpretation"]
    json.dumps(result)


def test_audited_nonnegative_law_reproduces_marginals_and_all_zero_conflicts():
    system = supported_triangle()
    pi = [Q(1, 3), Q(2, 3)] * 3
    result = check_rpm_consistency(system, pi, include_witness=True)
    assert result["status"] == "feasible"
    assert result["primal_audit"]["nonnegative_mass"]
    assert result["primal_audit"]["mass_sum"] == pytest.approx(1)
    assert result["primal_audit"]["all_conflict_cylinders_zero_after_audit"]
    assert result["primal_audit"]["marginal_max_abs_error"] < 1e-12
    assert not result["primal_audit"]["exact_membership_claim"]
    assert len(result["sparse_law"]) == 3
    assert all(sum(e % 2 == 0 for e in atom["chosen_events"]) == 1 for atom in result["sparse_law"])
    json.dumps(result)


def test_aggregate_and_individual_conflict_moments_agree():
    aggregate, individual = supported_triangle(), supported_triangle(aggregate=False)
    assert individual.n_conflict_rows == 6
    assert individual.metadata["within_budget_zero_cylinder_rows"] == 3
    rng = np.random.default_rng(999)
    for _ in range(8):
        probabilities = rng.random(3)
        pi = np.column_stack([probabilities, 1 - probabilities]).ravel()
        a, b = check_rpm_consistency(aggregate, pi), check_rpm_consistency(individual, pi)
        assert a["status"] == b["status"]
        assert a["status"] != "unknown"
        if a["status"] == "infeasible":
            exact_certificate_check(aggregate, a, pi)
            exact_certificate_check(individual, b, pi)


def test_known_WARP_only_three_budget_path_is_feasible_but_SARP_V_LP_fails():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stochastic"))
    import warp_core
    geometry = warp_core.build_patches([[2, 3, 3], [3, 2, 3], [3, 3, 2]], [30] * 3)
    system = build_from_conflict_graph(geometry.offsets, warp_core.conflict_graph(geometry))
    warp = warp_core.enumerate_types(geometry, axiom="WARP")
    sarp = warp_core.enumerate_types(geometry, axiom="SARP")
    assert len(warp) == 27 and len(sarp) == 25
    cyclic = next(path for path in warp if warp_core.has_cycle(warp_core.revealed_graph(geometry, path)))
    pi = warp_core.type_matrix(geometry, [cyclic]).ravel()
    result = check_rpm_consistency(system, pi, include_witness=True)
    assert system.n_candidates == 64 > len(warp)
    assert result["status"] == "feasible"
    assert result["primal_audit"]["support_paths"] == 1
    assert result["primal_audit"]["conflict_cylinder_max_abs_mass"] == 0
    S = warp_core.type_matrix(geometry, sarp)
    old_sarp = linprog(np.zeros(len(sarp)), A_eq=S, b_eq=pi, bounds=(0, None), method="highs")
    assert old_sarp.status == 2


def test_small_design_matches_old_filtered_WARP_V_LP_only_in_validation():
    system = build_consistency_system([0, 2, 4], [(0, 2), (0, 1), (2, 3)])
    # The validation oracle is explicitly enumerated here, not in the producer.
    paths = [path for path in itertools.product((0, 1), (2, 3)) if path != (0, 2)]
    A = np.array([[int(e in path) for path in paths] for e in range(4)])
    for a, b in itertools.product((.1, .4, .8), repeat=2):
        pi = [a, 1 - a, b, 1 - b]
        primitive = check_rpm_consistency(system, pi)
        old = linprog(np.zeros(len(paths)), A_eq=A, b_eq=pi, bounds=(0, None), method="highs")
        assert primitive["status"] == ("feasible" if old.success else "infeasible")


def test_raw_estimates_are_not_repaired_and_invalid_normalization_is_detected():
    system = build_consistency_system([0, 2, 4], [])
    for pi in ([1.1, -.1, .4, .6], [.5, .5, .3, .3]):
        result = check_rpm_consistency(system, pi)
        assert result["status"] == "infeasible"
        assert not result["input_audit"]["simplex_repair_applied"]
        exact_certificate_check(system, result, pi)
    with pytest.raises(ValueError, match="finite"):
        check_rpm_consistency(system, [np.nan, .5, .5, .5])


def test_product_and_nonzero_guards_precede_candidate_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate allocation must not happen")
    monkeypatch.setattr(rpm_consistency.np, "empty", forbidden)
    with pytest.raises(CandidateLimit, match="product"):
        build_consistency_system([0, 1000, 2000], [], max_candidates=500000)
    with pytest.raises(CandidateLimit, match="nonzeros"):
        build_consistency_system([0, 2, 4], [], max_nonzeros=2)


def test_bad_geometry_and_uncertified_solver_results_are_never_success(monkeypatch):
    with pytest.raises(ValueError, match="offsets"):
        build_consistency_system([0, 2, 2], [])
    with pytest.raises(ValueError, match="endpoints"):
        build_consistency_system([0, 2, 4], [(0, 4)])
    with pytest.raises(ValueError, match="symmetric"):
        build_from_conflict_graph([0, 2, 4], [[2], [], [], []])
    with pytest.raises(ValueError, match="integer"):
        build_from_conflict_graph([0, 2, 4], [[2.1], [], [0], []])
    with pytest.raises(ValueError, match="integer"):
        build_from_conflict_graph([0, 2], [[True], [0]])
    system = supported_triangle()
    monkeypatch.setattr(rpm_consistency, "linprog",
                        lambda *a, **k: SimpleNamespace(status=1, success=False, message="test limit"))
    result = check_rpm_consistency(system, [.5] * 6)
    assert result["status"] == "unknown"
    assert "certificate" not in result


def test_failed_or_false_Farkas_result_remains_unknown(monkeypatch):
    system = supported_triangle()
    primal = SimpleNamespace(status=2, success=False, message="test infeasible")
    failed_dual = SimpleNamespace(status=1, success=False, message="test limit")
    false_dual = SimpleNamespace(status=0, success=True, message="test false dual",
                                 x=np.zeros(2 * system.matrix.shape[0]))
    for dual in (failed_dual, false_dual):
        results = iter((primal, dual))
        monkeypatch.setattr(rpm_consistency, "linprog", lambda *a, **k: next(results))
        result = check_rpm_consistency(system, [.5] * 6)
        assert result["status"] == "unknown"
        assert "certificate" not in result


def test_false_and_zero_rational_duals_are_rejected():
    system = supported_triangle()
    assert rationalize_certificate(system, [0] * system.matrix.shape[0], [.5] * 6) is None
    false = [-1] + [0] * (system.matrix.shape[0] - 1)
    assert rationalize_certificate(system, false, [.5] * 6) is None
