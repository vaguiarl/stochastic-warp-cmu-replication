import Mathlib

set_option autoImplicit false

/-!
# Minimal consumers for CMU restrictiveness and calibration experiments

Verified here: scalar quadratic ray bounds, blend-residual algebra, weighted
aggregation, rejection-event inclusion/measure transfer, and a concrete
nonconvexity witness for the analytic positive-quadrant/diagonal-ray gap.

NOT formalized here: continuous CMU witnesses, projection existence, support
duality, the geometric distance inequalities supplied to the scalar consumer,
Gaussian/survey/CF sampling validity, exchangeability/rank calibration, or an
estimated-face GMS theorem. None of these is silently assumed as an axiom.
The corresponding explicit hypotheses below mark the remaining bridges.
-/

namespace CMULocalCalibration

/-- Blend with the valid target itself: the coordinate residual shrinks. -/
theorem blend_residual_square (p q beta : ℝ) :
    (p - ((1 - beta) * q + beta * p)) ^ 2 =
      (1 - beta) ^ 2 * (p - q) ^ 2 := by
  ring

/-- Restrict a support-dual objective to the nonnegative ray t*v. -/
theorem ray_quadratic_le (a v t : ℝ) (hv : 0 < v) (ht : 0 ≤ t) :
    2 * t * a - t ^ 2 * v ≤ (max 0 a) ^ 2 / v := by
  by_cases ha : 0 ≤ a
  · rw [max_eq_right ha]
    apply (le_div_iff₀ hv).2
    nlinarith [sq_nonneg (t * v - a)]
  · have ha' : a ≤ 0 := le_of_not_ge ha
    rw [max_eq_left ha']
    simp only [zero_pow (by decide : 2 ≠ 0), zero_div]
    have hta : t * a ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ht ha'
    have htv : 0 ≤ t ^ 2 * v := mul_nonneg (sq_nonneg t) hv.le
    nlinarith

theorem ray_optimum_attained (a v : ℝ) (hv : 0 < v) (ha : 0 ≤ a) :
    2 * (a / v) * a - (a / v) ^ 2 * v = (max 0 a) ^ 2 / v := by
  rw [max_eq_right ha]
  field_simp
  ring

/-- An exact ray bound is a lower loss bound once support duality is supplied.
This is a deterministic scalar consumer, not a new duality theorem. -/
theorem ray_loss_lower_bound (a v loss : ℝ) (hv : 0 < v)
    (hdual : ∀ t : ℝ, 0 ≤ t → 2 * t * a - t ^ 2 * v ≤ loss) :
    (max 0 a) ^ 2 / v ≤ loss := by
  by_cases ha : 0 ≤ a
  · rw [← ray_optimum_attained a v hv ha]
    exact hdual (a / v) (div_nonneg ha hv.le)
  · rw [max_eq_left (le_of_not_ge ha)]
    simpa using hdual 0 (le_refl 0)

/-- The raw-null geometric proof supplies these two distance bounds. The raw
target need not be a probability vector; its common projection is the anchor.
Subtracting the simplex floor leaves at most the squared perturbation norm. -/
theorem raw_null_gap_le_noise (dR dS normalSq cross noiseSq : ℝ)
    (hR : dR ≤ normalSq + 2 * cross + noiseSq)
    (hS : normalSq + 2 * cross ≤ dS) :
    dR - dS ≤ noiseSq := by
  linarith

theorem raw_null_gap_interval (dR dS normalSq cross noiseSq : ℝ)
    (hnesting : dS ≤ dR)
    (hR : dR ≤ normalSq + 2 * cross + noiseSq)
    (hS : normalSq + 2 * cross ≤ dS) :
    0 ≤ dR - dS ∧ dR - dS ≤ noiseSq := by
  exact ⟨sub_nonneg.mpr hnesting,
    raw_null_gap_le_noise dR dS normalSq cross noiseSq hR hS⟩

/-- Pointwise domination, not a Gaussian or bootstrap validity theorem. -/
theorem rejection_event_subset {Ω : Type*} (stat noise : Ω → ℝ)
    (critical : ℝ) (hbound : ∀ ω, stat ω ≤ noise ω) :
    {ω | critical < stat ω} ⊆ {ω | critical < noise ω} := by
  intro ω hω
  exact lt_of_lt_of_le hω (hbound ω)

theorem rejection_measure_le {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) (stat noise : Ω → ℝ) (critical : ℝ)
    (hbound : ∀ ω, stat ω ≤ noise ω) :
    μ {ω | critical < stat ω} ≤ μ {ω | critical < noise ω} := by
  exact MeasureTheory.measure_mono (rejection_event_subset stat noise critical hbound)

/-- Terminal size-transfer consumer. The noise-tail guarantee is an explicit
external premise; this file does not establish it for the UK estimator. -/
theorem rejection_control_transfer {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) (stat noise : Ω → ℝ)
    (critical : ℝ) (alpha : ENNReal)
    (hbound : ∀ ω, stat ω ≤ noise ω)
    (htail : μ {ω | critical < noise ω} ≤ alpha) :
    μ {ω | critical < stat ω} ≤ alpha := by
  exact (rejection_measure_le μ stat noise critical hbound).trans htail

/-- Domination also preserves the direction of a finite Monte Carlo tail.
Distributional exchangeability and its probability guarantee remain external.
-/
theorem tail_count_monotone {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (draw : ι → ℝ) (stat noise : ℝ) (h : stat ≤ noise) :
    (s.filter (fun i => noise ≤ draw i)).card ≤
      (s.filter (fun i => stat ≤ draw i)).card := by
  classical
  apply Finset.card_le_card
  intro i hi
  obtain ⟨his, hidraw⟩ := Finset.mem_filter.mp hi
  exact Finset.mem_filter.mpr ⟨his, h.trans hidraw⟩

theorem weighted_interval {ι : Type*} (s : Finset ι)
    (w lower loss upper : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i)
    (hlo : ∀ i ∈ s, lower i ≤ loss i)
    (hhi : ∀ i ∈ s, loss i ≤ upper i) :
    (∑ i ∈ s, w i * lower i) ≤ (∑ i ∈ s, w i * loss i) ∧
      (∑ i ∈ s, w i * loss i) ≤ (∑ i ∈ s, w i * upper i) := by
  constructor
  · exact Finset.sum_le_sum (fun i hi => mul_le_mul_of_nonneg_left (hlo i hi) (hw i hi))
  · exact Finset.sum_le_sum (fun i hi => mul_le_mul_of_nonneg_left (hhi i hi) (hw i hi))

/-- Closed-form cone gap for S=positive quadrant and R=positive diagonal ray.
The identification with cone projections is supplied by the human derivation.
-/
noncomputable def diagonalConeGap (x y : ℝ) : ℝ :=
  (max 0 x) ^ 2 + (max 0 y) ^ 2 - (max 0 (x + y)) ^ 2 / 2

theorem nonconvex_midpoint_witness :
    (diagonalConeGap 1 1 + diagonalConeGap (-2) 0) / 2 = 0 ∧
      diagonalConeGap (-(1 / 2)) (1 / 2) = 1 / 4 := by
  norm_num [diagonalConeGap]

#print axioms blend_residual_square
#print axioms ray_quadratic_le
#print axioms ray_optimum_attained
#print axioms ray_loss_lower_bound
#print axioms raw_null_gap_le_noise
#print axioms raw_null_gap_interval
#print axioms rejection_control_transfer
#print axioms tail_count_monotone
#print axioms weighted_interval
#print axioms nonconvex_midpoint_witness

end CMULocalCalibration
