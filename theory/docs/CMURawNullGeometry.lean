import Mathlib

set_option autoImplicit false

/-!
# A global geometric bound for the simplex-floor distance gap

For nonempty nested sets R ⊆ S in a real inner-product space, let q ∈ R
be a nearest point to the raw null target p in the convex set S. This file
proves, for every perturbation u, the GLOBAL deterministic inequality

  0 ≤ sqDistance (p+u) R - sqDistance (p+u) S ≤ ‖u‖².

The squared distance is explicitly the infimum of squared norm residuals.
Normal support is derived from Mathlib's convex-projection characterization,
not supplied as an unexplained squared-distance inequality. The raw p need
not lie in either set. Sampling/noise-tail and estimated-face calibration
theorems are not part of this geometric result.
-/

namespace CMURawNullGeometry

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def squaredDistance (x : E) (C : Set E) : ℝ :=
  sInf ((fun y : E => ‖x - y‖ ^ 2) '' C)

theorem residual_values_bddBelow (x : E) (C : Set E) :
    BddBelow ((fun y : E => ‖x - y‖ ^ 2) '' C) := by
  refine ⟨0, ?_⟩
  rintro z ⟨y, hy, rfl⟩
  exact sq_nonneg ‖x - y‖

theorem squaredDistance_le_candidate (x q : E) (C : Set E) (hq : q ∈ C) :
    squaredDistance x C ≤ ‖x - q‖ ^ 2 := by
  exact csInf_le (residual_values_bddBelow x C) ⟨q, hq, rfl⟩

theorem squaredDistance_nonnegative (x : E) (C : Set E) (hne : C.Nonempty) :
    0 ≤ squaredDistance x C := by
  apply le_csInf (hne.image _)
  rintro z ⟨y, hy, rfl⟩
  exact sq_nonneg ‖x - y‖

theorem squaredDistance_antitone (x : E) (R S : Set E)
    (hR : R.Nonempty) (hRS : R ⊆ S) :
    squaredDistance x S ≤ squaredDistance x R := by
  apply csInf_le_csInf (residual_values_bddBelow x S) (hR.image _)
  rintro z ⟨y, hy, rfl⟩
  exact ⟨y, hRS hy, rfl⟩

theorem normal_affine_bound (p q u : E) (S : Set E)
    (hN : ∀ y ∈ S, 0 ≤ inner (𝕜 := ℝ) (p - q) (q - y)) :
    ∀ y ∈ S, ‖p - q‖ ^ 2 + 2 * inner (𝕜 := ℝ) (p - q) u ≤
      ‖p + u - y‖ ^ 2 := by
  intro y hy
  have hvec : p + u - y = (p - q) + (u + (q - y)) := by abel
  rw [hvec, norm_add_sq_real, inner_add_right]
  have hn := hN y hy
  nlinarith [sq_nonneg ‖u + (q - y)‖]

theorem squaredDistance_normal_lower (p q u : E) (S : Set E)
    (hS : S.Nonempty)
    (hN : ∀ y ∈ S, 0 ≤ inner (𝕜 := ℝ) (p - q) (q - y)) :
    ‖p - q‖ ^ 2 + 2 * inner (𝕜 := ℝ) (p - q) u ≤ squaredDistance (p + u) S := by
  apply le_csInf (hS.image _)
  rintro z ⟨y, hy, rfl⟩
  exact normal_affine_bound p q u S hN y hy

theorem common_normal_gap_bound (p q u : E) (R S : Set E)
    (hq : q ∈ R) (hRS : R ⊆ S)
    (hN : ∀ y ∈ S, 0 ≤ inner (𝕜 := ℝ) (p - q) (q - y)) :
    0 ≤ squaredDistance (p + u) R - squaredDistance (p + u) S ∧
      squaredDistance (p + u) R - squaredDistance (p + u) S ≤ ‖u‖ ^ 2 := by
  have hlo := squaredDistance_antitone (p + u) R S ⟨q, hq⟩ hRS
  have hup := squaredDistance_le_candidate (p + u) q R hq
  have hsl := squaredDistance_normal_lower p q u S ⟨q, hRS hq⟩ hN
  have hvec : p + u - q = (p - q) + u := by abel
  rw [hvec, norm_add_sq_real] at hup
  constructor <;> linarith

/-- Terminal geometric consumer: the minimizer's normal condition is now
derived, using convexity of S and its actual norm-minimization premise.
No differentiability, finite dimension or membership of p in S is needed.
-/
theorem raw_common_projection_gap_bound (p q u : E) (R S : Set E)
    (hq : q ∈ R) (hRS : R ⊆ S) (hconvex : Convex ℝ S)
    (hmin : ‖p - q‖ = ⨅ y : S, ‖p - y‖) :
    0 ≤ squaredDistance (p + u) R - squaredDistance (p + u) S ∧
      squaredDistance (p + u) R - squaredDistance (p + u) S ≤ ‖u‖ ^ 2 := by
  have hn := (norm_eq_iInf_iff_real_inner_le_zero hconvex (hRS hq)).mp hmin
  apply common_normal_gap_bound p q u R S hq hRS
  intro y hy
  have hyN := hn y hy
  have hvec : q - y = -(y - q) := by abel
  rw [hvec, inner_neg_right]
  linarith

theorem common_projection_is_raw_null (p q : E) (R S : Set E)
    (hq : q ∈ R) (hRS : R ⊆ S) (hconvex : Convex ℝ S)
    (hmin : ‖p - q‖ = ⨅ y : S, ‖p - y‖) :
    squaredDistance p R - squaredDistance p S = 0 := by
  have hb := raw_common_projection_gap_bound p q 0 R S hq hRS hconvex hmin
  simp only [add_zero, norm_zero, zero_pow (by decide : 2 ≠ 0)] at hb
  exact le_antisymm hb.2 hb.1

#print axioms squaredDistance_le_candidate
#print axioms squaredDistance_antitone
#print axioms common_normal_gap_bound
#print axioms raw_common_projection_gap_bound
#print axioms common_projection_is_raw_null

end CMURawNullGeometry
