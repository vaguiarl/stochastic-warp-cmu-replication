import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option autoImplicit false

/-!
# Two-dimensional algebra for a sufficient empty-majority-CMU certificate

Economic consumer: at an interior point of a three-good budget frontier,
represent each voter's derivative on a fixed rational tangent basis by a
two-dimensional covector. A nonzero determinant supplies one direction with
positive derivatives for both voters. Three nonzero covectors cannot all be
pairwise negatively proportional; hence a supplied pair-opposition condition
forces some voter's covector to vanish.

ALGEBRA VERIFIED only. These declarations do not prove derivatives of shifted
log utilities, strict concavity, componentwise monotonicity, coherence,
skewness, strict-star concavity, feasibility of a small tangent move, the
classification of pairs with no common improvement, boundary exclusion,
identification of zero covectors with individual maximizers, or emptiness of
an entire budget. Those are explicit analytic/economic bridges, not hidden
conclusions of this file. Exact rational numerical certificates specialize
these scalar real identities.
-/

namespace CMUEmptyDemandGeometry

abbrev Covector2 := ℝ × ℝ

noncomputable def dot2 (v d : Covector2) : ℝ := v.1 * d.1 + v.2 * d.2

noncomputable def det2 (v w : Covector2) : ℝ := v.1 * w.2 - v.2 * w.1

noncomputable def improvingDirection (v w : Covector2) : Covector2 :=
  ((w.2 - v.2) / det2 v w, (v.1 - w.1) / det2 v w)

/-- The explicit direction solves both linear derivative equations exactly. -/
theorem determinant_direction_identity (v w : Covector2)
    (hdet : det2 v w ≠ 0) :
    dot2 v (improvingDirection v w) = 1 ∧
      dot2 w (improvingDirection v w) = 1 := by
  dsimp only [dot2, improvingDirection, det2] at *
  constructor
  · calc
      v.1 * ((w.2 - v.2) / (v.1 * w.2 - v.2 * w.1)) +
          v.2 * ((v.1 - w.1) / (v.1 * w.2 - v.2 * w.1)) =
          (v.1 * (w.2 - v.2) + v.2 * (v.1 - w.1)) /
            (v.1 * w.2 - v.2 * w.1) := by ring
      _ = (v.1 * w.2 - v.2 * w.1) / (v.1 * w.2 - v.2 * w.1) := by
        congr 1
        ring
      _ = 1 := div_self hdet
  · calc
      w.1 * ((w.2 - v.2) / (v.1 * w.2 - v.2 * w.1)) +
          w.2 * ((v.1 - w.1) / (v.1 * w.2 - v.2 * w.1)) =
          (w.1 * (w.2 - v.2) + w.2 * (v.1 - w.1)) /
            (v.1 * w.2 - v.2 * w.1) := by ring
      _ = (v.1 * w.2 - v.2 * w.1) / (v.1 * w.2 - v.2 * w.1) := by
        congr 1
        ring
      _ = 1 := div_self hdet

/-- Named consumer: independent covectors have a joint positive direction. -/
theorem determinant_implies_joint_positive_direction (v w : Covector2)
    (hdet : det2 v w ≠ 0) :
    ∃ d : Covector2, 0 < dot2 v d ∧ 0 < dot2 w d := by
  obtain ⟨hv, hw⟩ := determinant_direction_identity v w hdet
  exact ⟨improvingDirection v w, by rw [hv]; norm_num, by rw [hw]; norm_num⟩

def NegativelyProportional (v w : Covector2) : Prop :=
  ∃ a : ℝ, 0 < a ∧ w.1 = -a * v.1 ∧ w.2 = -a * v.2

/-- Two nonzero negative multiples of the same vector cannot themselves be
negative multiples. -/
theorem no_three_nonzero_pairwise_opposed (v w z : Covector2)
    (hv : v ≠ (0, 0)) (hvw : NegativelyProportional v w)
    (hvz : NegativelyProportional v z) :
    ¬ NegativelyProportional w z := by
  obtain ⟨a, ha, hw1, hw2⟩ := hvw
  obtain ⟨b, hb, hz1, hz2⟩ := hvz
  rintro ⟨c, hc, hop1, hop2⟩
  have hcoef : 0 < b + c * a := add_pos hb (mul_pos hc ha)
  have hvcoord : v.1 ≠ 0 ∨ v.2 ≠ 0 := by
    by_cases h1 : v.1 = 0
    · right
      intro h2
      exact hv (Prod.ext h1 h2)
    · exact Or.inl h1
  rcases hvcoord with h1 | h2
  · have heq : -b * v.1 = -c * (-a * v.1) := by
      simpa only [hw1, hz1] using hop1
    have hzero : (b + c * a) * v.1 = 0 := by nlinarith [heq]
    exact (mul_ne_zero (ne_of_gt hcoef) h1) hzero
  · have heq : -b * v.2 = -c * (-a * v.2) := by
      simpa only [hw2, hz2] using hop2
    have hzero : (b + c * a) * v.2 = 0 := by nlinarith [heq]
    exact (mul_ne_zero (ne_of_gt hcoef) h2) hzero

/-- Abstract core-candidate consumer. Pair opposition is supplied explicitly;
the analytic implication from absence of local majority improvement to this
pair condition is not proved here. -/
theorem pair_opposition_forces_a_zero_covector (v w z : Covector2)
    (hvw : v ≠ (0, 0) → w ≠ (0, 0) → NegativelyProportional v w)
    (hvz : v ≠ (0, 0) → z ≠ (0, 0) → NegativelyProportional v z)
    (hwz : w ≠ (0, 0) → z ≠ (0, 0) → NegativelyProportional w z) :
    v = (0, 0) ∨ w = (0, 0) ∨ z = (0, 0) := by
  by_cases hv : v = (0, 0)
  · exact Or.inl hv
  by_cases hw : w = (0, 0)
  · exact Or.inr (Or.inl hw)
  by_cases hz : z = (0, 0)
  · exact Or.inr (Or.inr hz)
  exact False.elim
    (no_three_nonzero_pairwise_opposed v w z hv (hvw hv hw) (hvz hv hz) (hwz hw hz))

#print axioms determinant_direction_identity
#print axioms determinant_implies_joint_positive_direction
#print axioms no_three_nonzero_pairwise_opposed
#print axioms pair_opposition_forces_a_zero_covector

end CMUEmptyDemandGeometry
