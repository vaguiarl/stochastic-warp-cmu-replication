import spinoff.laws_of_demand.lean.RPMGraphDual
import spinoff.laws_of_demand.lean.RPMIntegerH
import Mathlib.Data.Fintype.BigOperators

/-!
# Signed integer coefficients in the actual graph-generated RPM H rows

The initial graph constraints have integral normals and observable
coefficients. Division-free recursive elimination therefore yields signed
integer affine linear combinations of `(1, rho)`. This is proved for the
actual `rpmObservedRows` producer, with no hypothesis about its output.
-/

open scoped BigOperators
open LawsOfDemand.RPMFourierMotzkin LawsOfDemand.RPMIntegerH
open LawsOfDemand.RPMGraphDual

namespace LawsOfDemand.RPMGraphIntegerH

variable {V : Type*} [DecidableEq V]

theorem indicator_integer (s : Finset V) (v : V) :
    ∃ z : ℤ, indicator s v = (z : ℝ) := by
  by_cases h : v ∈ s
  · exact ⟨1, by simp [indicator, h]⟩
  · exact ⟨0, by simp [indicator, h]⟩

theorem direction_integer (positive : Bool) :
    ∃ z : ℤ, direction positive = (z : ℝ) := by
  cases positive
  · exact ⟨-1, by simp [direction]⟩
  · exact ⟨1, by simp [direction]⟩

theorem equationNormal_integer (G : SimpleGraph V) (e : Equation V) (s : Finset V) :
    ∃ z : ℤ, equationNormal G e s = (z : ℝ) := by
  classical
  rcases e with tag | v | ⟨u, v⟩
  · exact ⟨1, by simp [equationNormal]⟩
  · exact indicator_integer s v
  · by_cases h : G.Adj u v
    · obtain ⟨a, ha⟩ := indicator_integer s u
      obtain ⟨b, hb⟩ := indicator_integer s v
      exact ⟨a * b, by simp only [equationNormal, h, if_true, ha, hb, Int.cast_mul]⟩
    · exact ⟨0, by simp [equationNormal, h]⟩

theorem constraintNormal_integer (G : SimpleGraph V) (r : Constraint V) (s : Finset V) :
    ∃ z : ℤ, constraintNormal G r s = (z : ℝ) := by
  classical
  rcases r with s₀ | ⟨e, positive⟩
  · by_cases h : s = s₀
    · exact ⟨1, by simp [constraintNormal, h]⟩
    · exact ⟨0, by simp [constraintNormal, h]⟩
  · obtain ⟨a, ha⟩ := direction_integer positive
    obtain ⟨b, hb⟩ := equationNormal_integer G e s
    exact ⟨a * b, by simp only [constraintNormal, ha, hb, Int.cast_mul]⟩

theorem basisVector_integer (j k : Option V) :
    ∃ z : ℤ, basisVector j k = (z : ℝ) := by
  by_cases h : j = k
  · subst k
    exact ⟨1, by simp [basisVector]⟩
  · exact ⟨0, by simp [basisVector, Ne.symm h]⟩

theorem equationBound_basis_integer (e : Equation V) (j : Option V) :
    ∃ z : ℤ, equationBound e (basisVector j) = (z : ℝ) := by
  rcases e with tag | v | uv
  · simpa only [equationBound, LinearMap.proj_apply] using basisVector_integer j none
  · simpa only [equationBound, LinearMap.proj_apply] using basisVector_integer j (some v)
  · exact ⟨0, by simp [equationBound]⟩

theorem constraintBound_basis_integer (r : Constraint V) (j : Option V) :
    ∃ z : ℤ, constraintBound r (basisVector j) = (z : ℝ) := by
  rcases r with s | ⟨e, positive⟩
  · exact ⟨0, by simp [constraintBound]⟩
  · obtain ⟨a, ha⟩ := direction_integer positive
    obtain ⟨b, hb⟩ := equationBound_basis_integer e j
    exact ⟨a * b, by simp only [constraintBound, LinearMap.smul_apply,
      smul_eq_mul, ha, hb, Int.cast_mul]⟩

variable [Fintype V]

/-- Integrality is established on the actual initial graph constraints. -/
theorem rpmInitialRows_integral (G : SimpleGraph V) :
    IntegralRows (indexedRows (constraintNormal G) constraintBound
      (Fintype.equivFin (Finset V))) := by
  classical
  intro row hrow
  obtain ⟨r, _, rfl⟩ := Finset.mem_image.mp hrow
  constructor
  · intro k
    exact constraintNormal_integer G r ((Fintype.equivFin (Finset V)).symm k)
  · intro j
    obtain ⟨z, hz⟩ := constraintBound_basis_integer r j
    exact ⟨-z, by simp only [indexedRow, LinearMap.neg_apply, hz, Int.cast_neg]⟩

/-- Every coordinate coefficient of an actual generated H map is integral;
these are the coefficients at the constant coordinate `none` and at each
probability coordinate `some v`. -/
theorem rpmObservedRows_basis_integer (G : SimpleGraph V) :
    ∀ h ∈ rpmObservedRows G, ∀ j : Option V,
      ∃ z : ℤ, h (basisVector j) = (z : ℝ) := by
  classical
  intro h hh j
  obtain ⟨row, hrow, rfl⟩ := Finset.mem_image.mp hh
  exact ((eliminateAll_integral _ (rpmInitialRows_integral G)) row hrow).2 j

/-- Every actual geometry-generated H row is a signed integer affine
linear combination of the observed patch probabilities. -/
theorem rpmObservedRows_integer_affine (G : SimpleGraph V) :
    ∀ h ∈ rpmObservedRows G, ∃ w : Option V → ℤ,
      ∀ rho : V → ℝ, h (observedData rho) =
        (w none : ℝ) + ∑ v, (w (some v) : ℝ) * rho v := by
  intro h hh
  obtain ⟨w, hw⟩ := observedRows_integer_weights _ (rpmInitialRows_integral G) h hh
  refine ⟨w, fun rho => ?_⟩
  rw [hw, Fintype.sum_option]
  simp only [observedData, mul_one]

#print axioms rpmInitialRows_integral
#print axioms rpmObservedRows_basis_integer
#print axioms rpmObservedRows_integer_affine

end LawsOfDemand.RPMGraphIntegerH
