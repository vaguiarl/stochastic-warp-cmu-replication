import spinoff.laws_of_demand.lean.RPMGraphDual
import spinoff.laws_of_demand.lean.RPMGraphIntegerH
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.GroupWithZero.Action.Pi

set_option autoImplicit false

/-!
# Unit-right-hand-side observable RPM inequalities

The full finite RPM theorem in RPMGraphDual produces affine observable
rows, with a constant coordinate equal to one. Budget adding-up lets us
write the exact same test as Xi rho >= 1. The coefficient formula is
literal: evaluate a generated map on coordinate basis vectors, then add
its constant coefficient plus one to each event on one reference budget.

No row is assumed to be binary, nonnegative, facet-defining, or small.
This is a normalization of the verified finite H family, not a new
compactness theorem or a proof of the continuous CMU construction.
-/

namespace LawsOfDemand.RPMUnitH

open scoped BigOperators
open RPMGraphDual

variable {V T : Type*} [Fintype V] [DecidableEq V]
  [Fintype T] [DecidableEq T]

theorem linearMap_coordinate_sum {K : Type*} [Fintype K] [DecidableEq K]
    (h : (K → ℝ) →ₗ[ℝ] ℝ) (data : K → ℝ) :
    h data = ∑ k, h (Pi.single k 1) * data k := by
  have hdata : (∑ k, data k • Pi.single k (1 : ℝ)) = data := by
    have hs : ∀ k, data k • Pi.single k (1 : ℝ) = Pi.single k (data k) := by
      intro k
      simpa only [smul_eq_mul, mul_one] using
        (Pi.single_smul' k (data k) (1 : ℝ)).symm
    simp only [hs]
    exact Finset.univ_sum_single data
  calc
    h data = h (∑ k, data k • Pi.single k (1 : ℝ)) := congrArg h hdata.symm
    _ = ∑ k, data k * h (Pi.single k 1) := by
      simp only [map_sum, map_smul, smul_eq_mul]
    _ = ∑ k, h (Pi.single k 1) * data k := by
      apply Finset.sum_congr rfl
      intro k _
      exact mul_comm _ _

def unitCoefficient (budget : V → T) (t₀ : T)
    (h : Observable V →ₗ[ℝ] ℝ) (v : V) : ℝ :=
  h (Pi.single (some v) 1) +
    (h (Pi.single none 1) + 1) * (if budget v = t₀ then 1 else 0)

def unitRow (budget : V → T) (t₀ : T)
    (h : Observable V →ₗ[ℝ] ℝ) : (V → ℝ) →ₗ[ℝ] ℝ :=
  ∑ v, unitCoefficient budget t₀ h v • LinearMap.proj v

omit [Fintype T] in
theorem unitRow_apply (budget : V → T) (t₀ : T)
    (h : Observable V →ₗ[ℝ] ℝ) (rho : V → ℝ) :
    unitRow budget t₀ h rho = ∑ v, unitCoefficient budget t₀ h v * rho v := by
  simp [unitRow]

theorem observed_coordinate_sum (h : Observable V →ₗ[ℝ] ℝ) (rho : V → ℝ) :
    h (observedData rho) = h (Pi.single none 1) +
      ∑ v, h (Pi.single (some v) 1) * rho v := by
  rw [linearMap_coordinate_sum]
  simp [observedData, Fintype.sum_option]

omit [Fintype T] in
/-- Adding-up clears the affine constant without losing any inequality. -/
theorem unitRow_apply_of_addingUp (budget : V → T) (t₀ : T)
    (h : Observable V →ₗ[ℝ] ℝ) (rho : V → ℝ)
    (hadd : BudgetAddingUp budget rho) :
    unitRow budget t₀ h rho = h (observedData rho) + 1 := by
  rw [unitRow_apply]
  simp_rw [unitCoefficient]
  have hb :
      (∑ v, (h (Pi.single none 1) + 1) *
        (if budget v = t₀ then 1 else 0) * rho v) = h (Pi.single none 1) + 1 := by
    calc
      _ = (h (Pi.single none 1) + 1) *
          (∑ v, if budget v = t₀ then rho v else 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro v _
        split_ifs <;> simp
      _ = _ := by rw [hadd t₀, mul_one]
  calc
    _ = (∑ v, h (Pi.single (some v) 1) * rho v) +
        (∑ v, (h (Pi.single none 1) + 1) *
          (if budget v = t₀ then 1 else 0) * rho v) := by
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro v _
      ring
    _ = _ := by rw [hb, observed_coordinate_sum]; ring

noncomputable def unitRows (G : SimpleGraph V) (budget : V → T) (t₀ : T) :
    Finset ((V → ℝ) →ₗ[ℝ] ℝ) := by
  classical
  exact (rpmObservedRows G).image (unitRow budget t₀)

omit [Fintype T] in
theorem observed_H_iff_unit_H (G : SimpleGraph V) (budget : V → T) (t₀ : T)
    (rho : V → ℝ) (hadd : BudgetAddingUp budget rho) :
    (∀ h ∈ rpmObservedRows G, 0 ≤ h (observedData rho)) ↔
      (∀ r ∈ unitRows G budget t₀, 1 ≤ r rho) := by
  classical
  simp only [unitRows, Finset.forall_mem_image]
  constructor
  · intro hh h hmem
    rw [unitRow_apply_of_addingUp budget t₀ h rho hadd]
    linarith [hh h hmem]
  · intro hh h hmem
    have hu := hh hmem
    rw [unitRow_apply_of_addingUp budget t₀ h rho hadd] at hu
    linarith

omit [Fintype T] in
/-- Exact terminal matrix format: RPM iff adding-up and Xi rho >= 1.
The reference budget is explicit, and no latent weight remains on the RHS. -/
theorem rpm_iff_unit_H (G : SimpleGraph V) (budget : V → T)
    (hclique : BudgetCliques G budget) (t₀ : T) (rho : V → ℝ) :
    RPMMixture G budget rho ↔ BudgetAddingUp budget rho ∧
      (∀ r ∈ unitRows G budget t₀, 1 ≤ r rho) := by
  rw [rpm_iff_budgetAddingUp_and_observed_H G budget hclique]
  constructor
  · rintro ⟨hadd, hh⟩
    exact ⟨hadd, (observed_H_iff_unit_H G budget t₀ rho hadd).mp hh⟩
  · rintro ⟨hadd, hh⟩
    exact ⟨hadd, (observed_H_iff_unit_H G budget t₀ rho hadd).mpr hh⟩

omit [Fintype T] in
theorem unitCoefficient_integer (G : SimpleGraph V) (budget : V → T)
    (t₀ : T) (h : Observable V →ₗ[ℝ] ℝ) (hh : h ∈ rpmObservedRows G) (v : V) :
    ∃ z : ℤ, unitCoefficient budget t₀ h v = (z : ℝ) := by
  obtain ⟨a, ha⟩ := RPMGraphIntegerH.rpmObservedRows_basis_integer G h hh (some v)
  obtain ⟨c, hc⟩ := RPMGraphIntegerH.rpmObservedRows_basis_integer G h hh none
  change h (Pi.single (some v) 1) = (a : ℝ) at ha
  change h (Pi.single none 1) = (c : ℝ) at hc
  by_cases hb : budget v = t₀
  · refine ⟨a + c + 1, ?_⟩
    simp [unitCoefficient, hb, ha, hc, Int.cast_add]
    ring
  · exact ⟨a, by simp [unitCoefficient, hb, ha]⟩

omit [Fintype T] in
/-- Actual final Xi rows have integer probability-coordinate coefficients,
derived from the graph system rather than assumed of the output. -/
theorem unitRows_integer_coefficients (G : SimpleGraph V) (budget : V → T)
    (t₀ : T) : ∀ r ∈ unitRows G budget t₀, ∃ w : V → ℤ,
      ∀ rho, r rho = ∑ v, (w v : ℝ) * rho v := by
  classical
  intro r hr
  obtain ⟨h, hh, rfl⟩ := Finset.mem_image.mp hr
  choose w hw using unitCoefficient_integer G budget t₀ h hh
  exact ⟨w, fun rho => by rw [unitRow_apply]; simp only [hw]⟩

omit [Fintype T] in
/-- Price-sign terminal theorem in the literal Xi rho >= 1 format. -/
theorem price_sign_RPM_iff_unit_H (budget : V → T)
    (affordable : V → T → Prop) (t₀ : T) (rho : V → ℝ) :
    RPMMixture (patchConflictGraph budget affordable) budget rho ↔
      BudgetAddingUp budget rho ∧
        (∀ r ∈ unitRows (patchConflictGraph budget affordable) budget t₀, 1 ≤ r rho) :=
  rpm_iff_unit_H (patchConflictGraph budget affordable) budget
    (patchConflictGraph_budgetCliques budget affordable) t₀ rho

/-- A finite row index and literal coordinate formula for the displayed
matrix. Its entries depend on geometry and the reference budget, not rho. -/
abbrev MatrixRow (G : SimpleGraph V) (budget : V → T) (t₀ : T) :=
  {r // r ∈ unitRows G budget t₀}

def matrixCoefficient (G : SimpleGraph V) (budget : V → T) (t₀ : T)
    (r : MatrixRow G budget t₀) (v : V) : ℝ := r.val (Pi.single v 1)

omit [Fintype T] in
theorem matrixCoefficient_integer (G : SimpleGraph V) (budget : V → T) (t₀ : T)
    (r : MatrixRow G budget t₀) (v : V) :
    ∃ z : ℤ, matrixCoefficient G budget t₀ r v = (z : ℝ) := by
  obtain ⟨w, hw⟩ := unitRows_integer_coefficients G budget t₀ r.val r.property
  refine ⟨w v, ?_⟩
  simp [matrixCoefficient, hw, Pi.single_apply]

omit [Fintype T] in
/-- Unchanged final finite observable target, in literal matrix coordinates.
All its entries are proved integral by matrixCoefficient_integer. -/
theorem price_sign_RPM_iff_integer_matrix (budget : V → T)
    (affordable : V → T → Prop) (t₀ : T) (rho : V → ℝ) :
    RPMMixture (patchConflictGraph budget affordable) budget rho ↔
      BudgetAddingUp budget rho ∧
        (∀ r : MatrixRow (patchConflictGraph budget affordable) budget t₀,
          1 ≤ ∑ v, matrixCoefficient (patchConflictGraph budget affordable) budget t₀ r v * rho v) := by
  rw [price_sign_RPM_iff_unit_H budget affordable t₀]
  have hrows :
      (∀ r ∈ unitRows (patchConflictGraph budget affordable) budget t₀, 1 ≤ r rho) ↔
        (∀ r : MatrixRow (patchConflictGraph budget affordable) budget t₀,
          1 ≤ ∑ v, matrixCoefficient (patchConflictGraph budget affordable) budget t₀ r v * rho v) := by
    constructor
    · intro hh r
      have h := hh r.val r.property
      rw [linearMap_coordinate_sum] at h
      exact h
    · intro hh r hr
      have h := hh ⟨r, hr⟩
      rw [linearMap_coordinate_sum r rho]
      exact h
  exact and_congr_right fun _ => hrows

#print axioms unitRow_apply_of_addingUp
#print axioms observed_H_iff_unit_H
#print axioms rpm_iff_unit_H
#print axioms unitRows_integer_coefficients
#print axioms price_sign_RPM_iff_unit_H
#print axioms matrixCoefficient_integer
#print axioms price_sign_RPM_iff_integer_matrix

end LawsOfDemand.RPMUnitH
