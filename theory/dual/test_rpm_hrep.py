"""Small exhaustive validation is deliberately confined to tests, not the producer."""

from fractions import Fraction as Q
import inspect
import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

import rpm_hrep
from rpm_hrep import (GenerationLimit, Limits, Row, complete_budget_face, primitive_row,
                      stable_set_hrep, verify_arithmetic_provenance, warp_hrep_from_signs)


def independent_vertices(n, edges):
    return np.array([bits for bits in itertools.product((0, 1), repeat=n)
                     if all(not (bits[u] and bits[v]) for u, v in edges)], dtype=float)


def support_validation(hrep, seed=381, directions=30):
    vertices = independent_vertices(hrep.n_vertices, hrep.edges)
    if hrep.budget_blocks:
        vertices = np.array([v for v in vertices
                             if all(sum(v[list(block)]) == 1 for block in hrep.budget_blocks)])
    for vertex in vertices:
        assert hrep.contains([int(x) for x in vertex])
    rng = np.random.default_rng(seed)
    probes = np.vstack([np.eye(hrep.n_vertices), -np.eye(hrep.n_vertices),
                        rng.normal(size=(directions, hrep.n_vertices))])
    A = np.array([r.coeff for r in hrep.rows], dtype=float)
    b = np.array([r.rhs for r in hrep.rows], dtype=float)
    for objective in probes:
        result = linprog(-objective, A_ub=A, b_ub=b, bounds=[(None, None)] * hrep.n_vertices,
                         method="highs")
        assert result.success, result.message
        assert -result.fun == pytest.approx(np.max(vertices @ objective), abs=2e-8)
    return len(vertices)


def cycle(n):
    return [(i, (i + 1) % n) for i in range(n)]


def canonical_signs():
    budgets = [j for j in range(3) for _ in range(4)]
    signs = []
    for j in range(3):
        for pattern in itertools.product((True, False), repeat=2):
            row = [False] * 3
            others = [k for k in range(3) if k != j]
            for k, sign in zip(others, pattern):
                row[k] = sign
            signs.append(row)
    return budgets, signs


def test_primitive_normalization_never_flips_an_inequality():
    assert primitive_row(Row((-4, 2), 6)) == Row((-2, 1), 3)
    assert primitive_row(Row((4, -2), -6)) == Row((2, -1), -3)
    with pytest.raises(TypeError):
        primitive_row(Row((Q(1, 2),), 1))


def test_empty_singleton_and_edgeless_graphs_are_exact_boxes():
    empty = stable_set_hrep(0, [])
    assert empty.contains([]) and empty.rows == ()
    for n in range(1, 5):
        hrep = stable_set_hrep(n, [])
        assert len(hrep.rows) == 2 * n
        assert support_validation(hrep, directions=8) == 2 ** n
        assert verify_arithmetic_provenance(hrep)["arithmetic_audit_passed"]


def test_K3_recovers_the_missing_three_event_constraint():
    hrep = stable_set_hrep(3, [(0, 1), (0, 2), (1, 2)])
    assert hrep.inequality([1, 1, 1], 1)
    assert not hrep.contains([Q(1, 2)] * 3)
    assert hrep.contains([Q(1, 3)] * 3)
    assert support_validation(hrep) == 4
    assert all(node["universal_vertex"] for node in hrep.audit["recursion_nodes"])
    assert hrep.audit["fm_pair_products"] == 0
    assert verify_arithmetic_provenance(hrep)["arithmetic_audit_passed"]


def test_C5_recovers_the_odd_cycle_facet():
    hrep = stable_set_hrep(5, cycle(5))
    assert hrep.inequality([1] * 5, 2)
    assert not hrep.contains([Q(1, 2)] * 5)
    assert hrep.contains([Q(2, 5)] * 5)
    assert support_validation(hrep) == 11
    assert verify_arithmetic_provenance(hrep)["arithmetic_audit_passed"]


def test_wheel_recovers_the_essential_weight_two_facet():
    edges = [(0, v) for v in range(1, 6)] + [(i + 1, ((i + 1) % 5) + 1) for i in range(5)]
    hrep = stable_set_hrep(6, edges)
    assert hrep.inequality([2, 1, 1, 1, 1, 1], 2)
    assert not hrep.contains([Q(2, 5)] + [Q(3, 10)] * 5)
    # Half the hub-only point and half the uniform size-two rim mixture.
    assert hrep.contains([Q(1, 2)] + [Q(1, 5)] * 5)
    assert support_validation(hrep) == 12
    assert verify_arithmetic_provenance(hrep)["arithmetic_audit_passed"]


def test_deterministic_vertex_rule_and_cached_induced_subgraphs():
    edges = cycle(5)
    first, second = stable_set_hrep(5, edges), stable_set_hrep(5, reversed(edges))
    assert first.rows == second.rows
    assert first.audit["recursion_nodes"] == second.audit["recursion_nodes"]
    assert first.audit["recursion_nodes"][0]["vertex"] == 0
    assert first.audit["cache_hits"] > 0
    assert not first.audit["independent_set_enumeration"]
    assert not first.audit["warp_type_enumeration"]


def test_seeded_small_graph_supports_and_mixtures():
    rng = np.random.default_rng(90941)
    for n in range(2, 7):
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < .45]
        hrep = stable_set_hrep(n, edges)
        support_validation(hrep, seed=n * 112, directions=16)
        points = independent_vertices(n, hrep.edges)
        weights = rng.integers(1, 10, len(points))
        point = [sum(Q(int(weights[i]), int(weights.sum())) * int(v[j])
                     for i, v in enumerate(points)) for j in range(n)]
        assert hrep.contains(point)
        verify_arithmetic_provenance(hrep)


def test_budget_face_is_complete_transversals_not_arbitrary_independent_sets():
    base = stable_set_hrep(4, [(0, 1), (2, 3), (0, 2)])
    face = complete_budget_face(base, [[0, 1], [2, 3]])
    assert not face.contains([0, 0, 0, 0])
    assert face.contains([Q(1, 2)] * 4)
    assert support_validation(face) == 3
    verify_arithmetic_provenance(face)
    with pytest.raises(ValueError, match="clique"):
        complete_budget_face(base, [[0, 3], [1, 2]])
    with pytest.raises(ValueError, match="partition"):
        complete_budget_face(base, [[0, 1], [1, 2, 3]])


def test_zero_selected_and_zero_omitted_perspective_masses():
    hrep = stable_set_hrep(4, [(0, 1), (1, 2), (2, 3)])
    # The deterministic max-degree pivot is1 and its selected child is{3}.
    assert hrep.audit["recursion_nodes"][0]["vertex"] == 1
    assert hrep.audit["recursion_nodes"][0]["selected_mask"] == 1 << 3
    # x_v=0 forces the selected-child y=0. x_v=1 forces the omitted child0.
    assert hrep.contains([Q(1, 2), 0, Q(1, 2), Q(1, 2)])
    assert hrep.contains([0, 1, 0, Q(3, 4)])
    assert not hrep.contains([Q(1, 10), 1, 0, 0])
    assert not hrep.contains([0, 0, 1, Q(1, 10)])
    audit = verify_arithmetic_provenance(hrep)
    assert audit["rules"]["selected_perspective"] > 0
    assert audit["rules"]["positive_sum"] > 0


def test_two_supported_events_per_budget_reject_the_Shirai_counterexample():
    # This is the E/F-supported face of the actual full12-patch geometry.
    budgets = [0, 0, 1, 1, 2, 2]
    signs = [[False, True, True], [False, False, False],
             [True, False, True], [False, False, False],
             [True, True, False], [False, False, False]]
    hrep = warp_hrep_from_signs(budgets, signs)
    assert hrep.inequality([1, 0, 1, 0, 1, 0], 1)
    assert not hrep.contains([Q(1, 2)] * 6)
    assert hrep.contains([Q(1, 3), Q(2, 3)] * 3)
    assert not hrep.contains([Q(1, 3), Q(1, 3)] * 3)
    assert support_validation(hrep) == 4
    verify_arithmetic_provenance(hrep)


def test_full_twelve_patch_adapter_raises_a_bounded_guard_not_partial_output():
    budgets, signs = canonical_signs()
    with pytest.raises(GenerationLimit):
        warp_hrep_from_signs(budgets, signs,
                             limits=Limits(max_rows=100, max_pair_products=1000, max_seconds=2))


def test_invalid_graph_signs_and_limit_failures_never_return_partial_success():
    with pytest.raises(ValueError, match="loops"):
        stable_set_hrep(2, [(0, 0)])
    with pytest.raises(ValueError, match="vertex indices"):
        stable_set_hrep(2, [(0, 2)])
    with pytest.raises(GenerationLimit):
        stable_set_hrep(5, cycle(5), limits=Limits(max_rows=3))
    with pytest.raises(GenerationLimit, match="pair-product"):
        stable_set_hrep(5, cycle(5), limits=Limits(max_pair_products=1))
    with pytest.raises(ValueError, match="own-budget"):
        warp_hrep_from_signs([0, 1], [[True, True], [True, False]])
    with pytest.raises(TypeError, match="booleans"):
        warp_hrep_from_signs([0, 1], [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="not recorded"):
        verify_arithmetic_provenance(stable_set_hrep(2, [(0, 1)], provenance=False))


def test_producer_contains_no_independent_set_or_type_enumeration():
    source = inspect.getsource(rpm_hrep._Producer)
    assert "itertools" not in source
    assert "enumerate_types" not in source
    assert "product((0, 1)" not in source
