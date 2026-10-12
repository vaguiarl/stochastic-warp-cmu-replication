"""Projection certificates do not rely on supplied type completeness."""

from types import SimpleNamespace

import numpy as np
import pytest

import rpm_projection_audit as audit_module
from rpm_projection_audit import ConflictGraphProjectionAuditor, ProjectionAuditError


OFFSETS = [0, 2, 4, 6]
CONFLICTS = [(0, 2), (0, 4), (2, 4)]
TYPES = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
MEAN = np.tile([.25, .75], 3)


def test_clique_projection_has_independent_dual_certificate():
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES, mean=MEAN)
    target = np.full(6, .5)
    q = np.tile([1 / 3, 2 / 3], 3)
    result = auditor.audit(target, q, weights=[0, 1 / 3, 1 / 3, 1 / 3])
    assert result["audit_passed"]
    assert result["milp_dual_bound_used"]
    assert result["milp_optimal"]
    assert not result["support_uses_type_enumeration"]
    assert result["global_kkt_gap_upper"] < 1e-12
    assert result["squared_distance"] == pytest.approx(1 / 6)


def test_tightened_affine_support_transform_is_required():
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES)
    tau = .1
    free = np.array([0, 1 / 3, 1 / 3, 1 / 3])
    weights = tau / 4 + (1 - tau) * free
    q = tau * MEAN + (1 - tau) * np.tile([1 / 3, 2 / 3], 3)
    result = auditor.audit(np.full(6, .5), q, weights=weights, tau=tau)
    assert result["audit_passed"]
    assert result["tightened_support_upper"] == pytest.approx(result["projected_normal_score"])
    assert result["unshifted_support_upper"] - result["projected_normal_score"] > 1e-3


def test_incomplete_primal_witness_cannot_hide_missing_maximizer():
    # The provided hull has only FFF. FEF is graph-admissible and maximizes n.
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES[:1])
    q = np.tile([0., 1.], 3)
    target = q.copy()
    target[2:4] = [1., 0.]
    with pytest.raises(ProjectionAuditError, match="nonzero projection KKT") as error:
        auditor.audit(target, q, weights=[1.])
    assert error.value.audit["global_kkt_gap_upper"] == pytest.approx(2.)


def test_conflicting_witness_is_rejected():
    with pytest.raises(ValueError, match="forbidden conflict"):
        ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=np.array([[0, 0, 1]]))


def test_nonnegative_and_tightening_weight_constraints_are_checked():
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES)
    q = np.tile([1 / 3, 2 / 3], 3)
    with pytest.raises(ProjectionAuditError, match="primal witness"):
        auditor.audit(np.full(6, .5), q, weights=[0, 1 / 3, 1 / 3, 1 / 3], tau=.2)
    with pytest.raises(ValueError, match="tau"):
        auditor.audit(q, q, weights=[0, 1 / 3, 1 / 3, 1 / 3], tau=1.)


def test_center_must_match_original_witness_mean():
    with pytest.raises(ValueError, match="tightening mean"):
        ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES, mean=np.full(6, .5))


def test_solver_limit_never_counts_as_a_certificate(monkeypatch):
    monkeypatch.setattr(audit_module, "milp", lambda *a, **k:
                        SimpleNamespace(status=1, success=False, message="Time limit reached",
                                        x=np.zeros(6), mip_dual_bound=-100.))
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES)
    with pytest.raises(ProjectionAuditError, match="optimal finite dual") as error:
        auditor.audit(np.full(6, .5), np.tile([1 / 3, 2 / 3], 3),
                      weights=[0, 1 / 3, 1 / 3, 1 / 3])
    assert not error.value.audit["milp_optimal"]


def test_incumbent_without_dual_bound_is_not_sufficient(monkeypatch):
    monkeypatch.setattr(audit_module, "milp", lambda *a, **k:
                        SimpleNamespace(status=0, success=True, message="Optimal",
                                        x=np.array([1, 0, 0, 1, 0, 1]), mip_dual_bound=None))
    auditor = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES)
    with pytest.raises(ProjectionAuditError, match="optimal finite dual"):
        auditor.audit(np.full(6, .5), np.tile([1 / 3, 2 / 3], 3),
                      weights=[0, 1 / 3, 1 / 3, 1 / 3])


def test_zero_normal_is_analytic_and_conflict_order_is_irrelevant():
    auditor = ConflictGraphProjectionAuditor(OFFSETS, [(2, 0), (0, 2)] + CONFLICTS,
                                            types=TYPES)
    result = auditor.audit(MEAN, MEAN, weights=np.full(4, .25))
    assert result["audit_passed"]
    assert not result["milp_called"]
    assert result["squared_objective_error_upper_bound"] == 0.


def test_prepared_witness_from_another_graph_is_rejected():
    first = ConflictGraphProjectionAuditor(OFFSETS, CONFLICTS, types=TYPES)
    other = ConflictGraphProjectionAuditor(OFFSETS, [], types=TYPES)
    with pytest.raises(ValueError, match="this conflict graph"):
        other.audit(MEAN, MEAN, weights=np.full(4, .25), mixture=first.mixture)


@pytest.mark.parametrize("offsets,conflicts", [
    ([1, 2, 4], []), ([0, 2, 2], []), ([0, 2, 4], [(1, 1)]),
    ([0, 2, 4], [(1, 4)]), ([0, 2, 4], [(1.5, 3)]),
])
def test_malformed_geometry_is_rejected(offsets, conflicts):
    with pytest.raises(ValueError):
        ConflictGraphProjectionAuditor(offsets, conflicts)
