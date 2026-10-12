import spinoff.laws_of_demand.lean.RPMFourierMotzkin
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.GroupWithZero.Action.Pi

/-!
# Integer coefficients in the recursive observed-only H family

The finite-dimensional specialization below proves that the division-free
Fourier--Motzkin producer preserves integer coefficients. Each resulting
observable linear map is therefore a signed integer linear combination of the
observed coordinates. No integrality or completeness assumption is made
about the output: both are proved from the initial row coefficients.
-/

open scoped BigOperators
open LawsOfDemand.RPMFourierMotzkin

namespace LawsOfDemand.RPMIntegerH

variable {K : Type*} [DecidableEq K]

def basisVector (j : K) : K → ℝ := Pi.single j 1

/-- The latent coefficients and the observable standard-basis coefficients
of a row are integral. -/
def IntegralRow {n : ℕ} (r : Row n (K → ℝ)) : Prop :=
  (∀ i, ∃ z : ℤ, r.coefficient i = (z : ℝ)) ∧
  (∀ j, ∃ z : ℤ, r.observed (basisVector j) = (z : ℝ))

def IntegralRows {n : ℕ} (rows : Finset (Row n (K → ℝ))) : Prop :=
  ∀ r ∈ rows, IntegralRow r

theorem tailRow_integral {n : ℕ} {r : Row (n + 1) (K → ℝ)}
    (hr : IntegralRow r) : IntegralRow (tailRow r) := by
  exact ⟨fun i => hr.1 i.succ, hr.2⟩

theorem pairRow_integral {n : ℕ} {p q : Row (n + 1) (K → ℝ)}
    (hp : IntegralRow p) (hq : IntegralRow q) : IntegralRow (pairRow p q) := by
  obtain ⟨a, ha⟩ := hp.1 0
  obtain ⟨b, hb⟩ := hq.1 0
  constructor
  · intro i
    obtain ⟨u, hu⟩ := hp.1 i.succ
    obtain ⟨v, hv⟩ := hq.1 i.succ
    refine ⟨(-b) * u + a * v, ?_⟩
    simp only [pairRow, ha, hb, hu, hv, Int.cast_add, Int.cast_mul, Int.cast_neg]
  · intro j
    obtain ⟨u, hu⟩ := hp.2 j
    obtain ⟨v, hv⟩ := hq.2 j
    refine ⟨(-b) * u + a * v, ?_⟩
    simp only [pairRow, LinearMap.add_apply, LinearMap.smul_apply, smul_eq_mul,
      ha, hb, hu, hv, Int.cast_add, Int.cast_mul, Int.cast_neg]

/-- One step preserves integer coefficients, without any sign hypothesis
being needed for this algebraic preservation statement. -/
theorem eliminate_integral {n : ℕ} (rows : Finset (Row (n + 1) (K → ℝ)))
    (hr : IntegralRows rows) : IntegralRows (eliminate rows) := by
  classical
  intro r hmem
  rcases Finset.mem_union.mp hmem with hz | hp
  · obtain ⟨r₀, hr₀, rfl⟩ := Finset.mem_image.mp hz
    exact tailRow_integral (hr r₀ (Finset.mem_filter.mp hr₀).1)
  · obtain ⟨⟨p, q⟩, hpq, rfl⟩ := Finset.mem_image.mp hp
    obtain ⟨hp, hq⟩ := Finset.mem_product.mp hpq
    exact pairRow_integral (hr p (Finset.mem_filter.mp hp).1)
      (hr q (Finset.mem_filter.mp hq).1)

/-- Every row in the complete observed-only output has integer coefficients. -/
theorem eliminateAll_integral {n : ℕ} (rows : Finset (Row n (K → ℝ)))
    (hr : IntegralRows rows) : IntegralRows (eliminateAll rows) := by
  induction n with
  | zero => exact hr
  | succ n ih => exact ih (eliminate rows) (eliminate_integral rows hr)

variable [Fintype K]

/-- Standard-basis reconstruction for a real linear map on the finite
observation vector. -/
theorem linearMap_eq_basis_sum (h : (K → ℝ) →ₗ[ℝ] ℝ)
    (data : K → ℝ) :
    h data = ∑ j, h (basisVector j) * data j := by
  have hdata : (∑ j, data j • basisVector j) = data := by
    have hs : ∀ j, data j • basisVector j = Pi.single j (data j) := by
      intro j
      simpa only [basisVector, smul_eq_mul, mul_one] using
        (Pi.single_smul' j (data j) (1 : ℝ)).symm
    simp only [hs]
    exact Finset.univ_sum_single data
  calc
    h data = h (∑ j, data j • basisVector j) := congrArg h hdata.symm
    _ = ∑ j, data j * h (basisVector j) := by simp only [map_sum, map_smul, smul_eq_mul]
    _ = ∑ j, h (basisVector j) * data j := by
      apply Finset.sum_congr rfl
      intro j _
      exact mul_comm _ _

/-- Each generated H row is a signed integer linear combination of the
observed coordinate vector. -/
theorem observedRows_integer_weights {n : ℕ}
    (rows : Finset (Row n (K → ℝ))) (hr : IntegralRows rows) :
    ∀ h ∈ observedRows rows, ∃ w : K → ℤ,
      ∀ data, h data = ∑ j, (w j : ℝ) * data j := by
  classical
  intro h hh
  obtain ⟨r, hmem, rfl⟩ := Finset.mem_image.mp hh
  have hi := (eliminateAll_integral rows hr) r hmem
  choose w hw using hi.2
  refine ⟨w, fun data => ?_⟩
  rw [linearMap_eq_basis_sum]
  simp only [hw]

/-- The complete semantic projection and the signed integer-combination
representation hold together, derived solely from integral initial rows. -/
theorem integral_projection {n : ℕ} (rows : Finset (Row n (K → ℝ)))
    (hr : IntegralRows rows) :
    (∀ data, (∃ x : Fin n → ℝ, Satisfies rows x data) ↔
      ∀ h ∈ observedRows rows, 0 ≤ h data) ∧
    (∀ h ∈ observedRows rows, ∃ w : K → ℤ,
      ∀ data, h data = ∑ j, (w j : ℝ) * data j) := by
  exact ⟨fun data => exists_iff_observedRows rows data, observedRows_integer_weights rows hr⟩

#print axioms pairRow_integral
#print axioms eliminateAll_integral
#print axioms observedRows_integer_weights
#print axioms integral_projection

end LawsOfDemand.RPMIntegerH
