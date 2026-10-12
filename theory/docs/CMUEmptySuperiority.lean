import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option autoImplicit false

/-!
# Scalar consumers for the actual-empty-CMU superiority bypass

Economic/model bridge supplied by the human argument:
R is the complete RUM probability polytope, S the product of budget simplices,
and B the conditional actual-empty-CMU inner polytope at a fixed positive cap.
Its explicitly witnessed one-budget responders and all-null type make
(1-beta)*z_R + beta*q_S feasible whenever the total mixture mass is at most one.
Here z_R and q_S are the projections of the same (possibly invalid) raw target
onto R and S. The squared residual identity bounds the B distance; B subset S
supplies the opposite distance ordering.

ALGEBRA VERIFIED: scalar mixture accounting, a squared-residual identity,
the supplied-distance-bound sandwich, and equality/strict-inequality null
equivalences. NOT FORMALIZED: polytope membership, projections, their existence,
the norm identity summed over all coordinates, the actual-empty CMU witnesses,
tangent cones, bootstrap validity, CF bias, or statistical significance.
Distance hypotheses below are explicit; they are not constructed in this file.
-/

namespace CMUEmptySuperiority

/-- One budget's active mass remains tau under the complete-plus-responder mix. -/
theorem mixture_response_accounting (tau beta : ℝ) :
    tau * (1 - beta) + tau * beta = tau := by ring

/-- The residual all-null mass completes the total latent mixture to one. -/
theorem mixture_total_accounting (J tau beta : ℝ) :
    tau * (1 - beta) + J * (tau * beta) +
      (1 - tau * (1 + (J - 1) * beta)) = 1 := by ring

/-- Scalar feasibility of the three kinds of mixture mass. Per-patch probability
normalization and actual availability of the corresponding types are supplied
by the explicit witness construction outside this file. -/
theorem mixture_masses_nonnegative (J tau beta : ℝ)
    (htau : 0 ≤ tau) (hbeta : 0 ≤ beta) (hbeta_upper : beta ≤ 1)
    (hcapacity : tau * (1 + (J - 1) * beta) ≤ 1) :
    0 ≤ tau * (1 - beta) ∧ 0 ≤ tau * beta ∧
      0 ≤ 1 - tau * (1 + (J - 1) * beta) := by
  exact ⟨mul_nonneg htau (sub_nonneg.mpr hbeta_upper),
    mul_nonneg htau hbeta, sub_nonneg.mpr hcapacity⟩

/-- Coordinatewise squared-norm identity; finite Euclidean norms sum this
identity over their coordinates in the human geometric argument. -/
theorem coordinate_squared_mixture_identity (p x y beta : ℝ) :
    (p - ((1 - beta) * x + beta * y)) ^ 2 =
      (1 - beta) * (p - x) ^ 2 + beta * (p - y) ^ 2 -
        beta * (1 - beta) * (x - y) ^ 2 := by ring

/-- The raw-target distance gap is between beta and one times the probability-
domain-excess gap, under the explicit geometric distance bounds. -/
theorem squared_distance_gain_sandwich (dR dS dB beta separation : ℝ)
    (hSB : dS ≤ dB)
    (hbeta : 0 < beta) (hbeta_upper : beta ≤ 1)
    (hseparation : 0 ≤ separation)
    (hmixture : dB ≤ (1 - beta) * dR + beta * dS -
      beta * (1 - beta) * separation) :
    beta * (dR - dS) ≤ dR - dB ∧ dR - dB ≤ dR - dS := by
  have hcorrection : 0 ≤ beta * (1 - beta) * separation :=
    mul_nonneg (mul_nonneg hbeta.le (sub_nonneg.mpr hbeta_upper)) hseparation
  constructor
  · nlinarith
  · linarith

/-- Terminal scalar null consumer. The target need not belong to S. -/
theorem actual_empty_gain_zero_iff_probability_excess_zero
    (dR dS dB beta separation : ℝ)
    (hSR : dS ≤ dR) (hSB : dS ≤ dB)
    (hbeta : 0 < beta) (hbeta_upper : beta ≤ 1)
    (hseparation : 0 ≤ separation)
    (hmixture : dB ≤ (1 - beta) * dR + beta * dS -
      beta * (1 - beta) * separation) :
    dR - dB = 0 ↔ dR - dS = 0 := by
  obtain ⟨hlower, hupper⟩ := squared_distance_gain_sandwich
    dR dS dB beta separation hSB hbeta hbeta_upper hseparation hmixture
  have hpsi : 0 ≤ dR - dS := sub_nonneg.mpr hSR
  constructor
  · intro hzero
    have hproduct : beta * (dR - dS) = 0 := by
      have hnonnegative := mul_nonneg hbeta.le hpsi
      rw [hzero] at hlower
      exact le_antisymm hlower hnonnegative
    exact (mul_eq_zero.mp hproduct).resolve_left (ne_of_gt hbeta)
  · intro hzero
    rw [hzero, mul_zero] at hlower
    rw [hzero] at hupper
    exact le_antisymm hupper hlower

theorem actual_empty_gain_positive_iff_probability_excess_positive
    (dR dS dB beta separation : ℝ)
    (hSB : dS ≤ dB)
    (hbeta : 0 < beta) (hbeta_upper : beta ≤ 1)
    (hseparation : 0 ≤ separation)
    (hmixture : dB ≤ (1 - beta) * dR + beta * dS -
      beta * (1 - beta) * separation) :
    0 < dR - dB ↔ 0 < dR - dS := by
  obtain ⟨hlower, hupper⟩ := squared_distance_gain_sandwich
    dR dS dB beta separation hSB hbeta hbeta_upper hseparation hmixture
  constructor
  · intro hphi
    exact lt_of_lt_of_le hphi hupper
  · intro hpsi
    exact lt_of_lt_of_le (mul_pos hbeta hpsi) hlower

#print axioms mixture_response_accounting
#print axioms mixture_total_accounting
#print axioms mixture_masses_nonnegative
#print axioms coordinate_squared_mixture_identity
#print axioms squared_distance_gain_sandwich
#print axioms actual_empty_gain_zero_iff_probability_excess_zero
#print axioms actual_empty_gain_positive_iff_probability_excess_positive

end CMUEmptySuperiority
