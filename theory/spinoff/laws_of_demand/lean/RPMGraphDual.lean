import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.LinearAlgebra.Pi
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import spinoff.laws_of_demand.lean.RPMFourierMotzkin

set_option autoImplicit false

/-!
# Finite-mixture bridge for the graph-generated RPM dual

The graph is the actual finite patch conflict graph. Independent selections
are finite vertex sets, and mixture weights range over all finite vertex sets.
No stable-set H-characterization, support oracle, or integer-hull theorem is
assumed. The principal consumer proves that budget adding-up selects exactly
the complete-choice face of the stable-mixture model.

The second consumer proves the universal-vertex lifting rule directly for
independent selections and their finite mixtures. It explains, for example,
why lifting the five-cycle row of bound two assigns coefficient two to a
universal wheel hub.

Compile from the workspace root (after the elimination module is built):
`lake env sh -c 'LEAN_PATH=".:$LEAN_PATH" lean -DwarningAsError=true
  spinoff/laws_of_demand/lean/RPMGraphDual.lean'`.
-/

namespace LawsOfDemand.RPMGraphDual

open scoped BigOperators
attribute [local instance] Classical.propDecidable

variable {V T : Type*} [Fintype V] [DecidableEq V]
  [DecidableEq T]

def Independent (G : SimpleGraph V) (s : Finset V) : Prop :=
  ∀ u ∈ s, ∀ v ∈ s, ¬ G.Adj u v

def indicator (s : Finset V) (v : V) : ℝ := if v ∈ s then 1 else 0

def marginal (mu : Finset V → ℝ) (v : V) : ℝ :=
  ∑ s, mu s * indicator s v

def StableWeights (G : SimpleGraph V) (mu : Finset V → ℝ) : Prop :=
  (∀ s, 0 ≤ mu s) ∧ (∑ s, mu s = 1) ∧
    (∀ s, mu s ≠ 0 → Independent G s)

def StableMixture (G : SimpleGraph V) (rho : V → ℝ) : Prop :=
  ∃ mu : Finset V → ℝ, StableWeights G mu ∧ marginal mu = rho

def blockCard (budget : V → T) (s : Finset V) (t : T) : ℕ :=
  (s.filter fun v => budget v = t).card

def Complete (budget : V → T) (s : Finset V) : Prop :=
  ∀ t, blockCard budget s t = 1

def BudgetAddingUp (budget : V → T) (rho : V → ℝ) : Prop :=
  ∀ t, (∑ v, if budget v = t then rho v else 0) = 1

def RPMMixture (G : SimpleGraph V) (budget : V → T) (rho : V → ℝ) : Prop :=
  ∃ mu : Finset V → ℝ, StableWeights G mu ∧
    (∀ s, mu s ≠ 0 → Complete budget s) ∧ marginal mu = rho

def BudgetCliques (G : SimpleGraph V) (budget : V → T) : Prop :=
  ∀ u v, u ≠ v → budget u = budget v → G.Adj u v

omit [Fintype V] in
theorem independent_blockCard_le_one (G : SimpleGraph V) (budget : V → T)
    (hclique : BudgetCliques G budget) (s : Finset V)
    (hs : Independent G s) (t : T) : blockCard budget s t ≤ 1 := by
  apply Finset.card_le_one.mpr
  intro u hu v hv
  obtain ⟨hus, hut⟩ := Finset.mem_filter.mp hu
  obtain ⟨hvs, hvt⟩ := Finset.mem_filter.mp hv
  by_contra hne
  exact hs u hus v hvs (hclique u v hne (hut.trans hvt.symm))

theorem indicator_budget_sum (budget : V → T) (s : Finset V) (t : T) :
    (∑ v, if budget v = t then indicator s v else 0) =
      (blockCard budget s t : ℝ) := by
  classical
  have hsum :
      (∑ v, if budget v = t then indicator s v else 0) =
        ∑ v, if budget v = t ∧ v ∈ s then (1 : ℝ) else 0 := by
    apply Finset.sum_congr rfl
    intro v hv
    by_cases hs : v ∈ s <;> by_cases ht : budget v = t <;> simp [indicator, hs, ht]
  have hfilter : Finset.univ.filter (fun v => budget v = t ∧ v ∈ s) =
      s.filter (fun v => budget v = t) := by
    ext v
    simp [and_comm]
  rw [hsum]
  simp only [Finset.sum_boole]
  rw [hfilter]
  rfl

theorem marginal_budget_sum (budget : V → T) (mu : Finset V → ℝ) (t : T) :
    (∑ v, if budget v = t then marginal mu v else 0) =
      ∑ s, mu s * (blockCard budget s t : ℝ) := by
  classical
  calc
    (∑ v, if budget v = t then marginal mu v else 0) =
        ∑ v, ∑ s, mu s * (if budget v = t then indicator s v else 0) := by
      apply Finset.sum_congr rfl
      intro v hv
      by_cases ht : budget v = t <;> simp [ht, marginal]
    _ = ∑ s, ∑ v, mu s * (if budget v = t then indicator s v else 0) :=
      Finset.sum_comm
    _ = ∑ s, mu s * (blockCard budget s t : ℝ) := by
      apply Finset.sum_congr rfl
      intro s hs
      rw [← Finset.mul_sum, indicator_budget_sum]

/-- Saturation of every budget marginal forces every positive-mass stable
selection to choose once from every budget. -/
theorem positive_mass_selection_complete (G : SimpleGraph V) (budget : V → T)
    (hclique : BudgetCliques G budget) (mu : Finset V → ℝ)
    (hmu : StableWeights G mu) (hadd : BudgetAddingUp budget (marginal mu)) :
    ∀ s, mu s ≠ 0 → Complete budget s := by
  intro s hmass t
  have hmean : (∑ r, mu r * (blockCard budget r t : ℝ)) = 1 := by
    rw [← marginal_budget_sum]
    exact hadd t
  have hnonneg : ∀ r : Finset V,
      0 ≤ mu r * (1 - (blockCard budget r t : ℝ)) := by
    intro r
    by_cases hz : mu r = 0
    · simp [hz]
    · have hcard := independent_blockCard_le_one G budget hclique r (hmu.2.2 r hz) t
      have hcard_real : (blockCard budget r t : ℝ) ≤ 1 := by exact_mod_cast hcard
      exact mul_nonneg (hmu.1 r) (sub_nonneg.mpr hcard_real)
  have hzero : (∑ r, mu r * (1 - (blockCard budget r t : ℝ))) = 0 := by
    simp_rw [mul_sub, mul_one]
    rw [Finset.sum_sub_distrib, hmu.2.1, hmean]
    ring
  have hs_zero : mu s * (1 - (blockCard budget s t : ℝ)) = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun r _ => hnonneg r)).mp hzero s (Finset.mem_univ s)
  have hcard_real : (blockCard budget s t : ℝ) = 1 := by
    have := (mul_eq_zero.mp hs_zero).resolve_left hmass
    linarith
  exact_mod_cast hcard_real

theorem complete_support_budgetAddingUp (G : SimpleGraph V) (budget : V → T)
    (mu : Finset V → ℝ) (hmu : StableWeights G mu)
    (hcomplete : ∀ s, mu s ≠ 0 → Complete budget s) :
    BudgetAddingUp budget (marginal mu) := by
  intro t
  rw [marginal_budget_sum]
  calc
    (∑ s, mu s * (blockCard budget s t : ℝ)) = ∑ s, mu s := by
      apply Finset.sum_congr rfl
      intro s hs
      by_cases hz : mu s = 0
      · simp [hz]
      · rw [hcomplete s hz t]
        simp
    _ = 1 := hmu.2.1

/-- Named RPM consumer: the complete-choice face of actual stable mixtures.
This equivalence does not assume an H-representation of STAB. -/
theorem rpm_iff_stableMixture_and_budgetAddingUp
    (G : SimpleGraph V) (budget : V → T) (hclique : BudgetCliques G budget)
    (rho : V → ℝ) :
    RPMMixture G budget rho ↔ StableMixture G rho ∧ BudgetAddingUp budget rho := by
  constructor
  · rintro ⟨mu, hmu, hcomplete, hrho⟩
    exact ⟨⟨mu, hmu, hrho⟩, hrho ▸ complete_support_budgetAddingUp G budget mu hmu hcomplete⟩
  · rintro ⟨⟨mu, hmu, hrho⟩, hadd⟩
    refine ⟨mu, hmu, positive_mass_selection_complete G budget hclique mu hmu ?_, hrho⟩
    exact hrho.symm ▸ hadd

def score (a : V → ℝ) (s : Finset V) : ℝ :=
  ∑ v, a v * indicator s v

def dot (a rho : V → ℝ) : ℝ := ∑ v, a v * rho v

theorem score_marginal (a : V → ℝ) (mu : Finset V → ℝ) :
    dot a (marginal mu) = ∑ s, mu s * score a s := by
  classical
  unfold dot marginal score
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro s hs
  apply Finset.sum_congr rfl
  intro v hv
  ring

theorem stableWeights_score_bound (G : SimpleGraph V) (mu : Finset V → ℝ)
    (hmu : StableWeights G mu) (a : V → ℝ) (b : ℝ)
    (hbound : ∀ s, Independent G s → score a s ≤ b) :
    dot a (marginal mu) ≤ b := by
  rw [score_marginal]
  calc
    (∑ s, mu s * score a s) ≤ ∑ s, mu s * b := by
      apply Finset.sum_le_sum
      intro s hs
      by_cases hz : mu s = 0
      · simp [hz]
      · exact mul_le_mul_of_nonneg_left (hbound s (hmu.2.2 s hz)) (hmu.1 s)
    _ = b := by rw [← Finset.sum_mul, hmu.2.1, one_mul]

def UniversalVertex (G : SimpleGraph V) (v : V) : Prop :=
  ∀ u, u ≠ v → G.Adj v u

def liftedWeight (a : V → ℝ) (b : ℝ) (v : V) : V → ℝ :=
  fun u => if u = v then b else a u

omit [Fintype V] in
theorem independent_contains_universal_eq_singleton (G : SimpleGraph V) (v : V)
    (hv : UniversalVertex G v) (s : Finset V) (hs : Independent G s)
    (hvs : v ∈ s) : s = {v} := by
  apply Finset.ext
  intro u
  constructor
  · intro hus
    have huv : u = v := by
      by_contra hne
      exact hs v hvs u hus (hv u hne)
    simp [huv]
  · intro hu
    simpa using (Finset.mem_singleton.mp hu) ▸ hvs

/-- Direct graph lifting, proved on actual independent selections. The
child bound is required only on selections omitting the universal vertex. -/
theorem universal_vertex_lift_selection (G : SimpleGraph V) (v : V)
    (hv : UniversalVertex G v) (a : V → ℝ) (b : ℝ)
    (hbase : ∀ s, Independent G s → v ∉ s → score a s ≤ b)
    (s : Finset V) (hs : Independent G s) :
    score (liftedWeight a b v) s ≤ b := by
  classical
  by_cases hvs : v ∈ s
  · rw [independent_contains_universal_eq_singleton G v hv s hs hvs]
    simp [score, indicator, liftedWeight]
  · have heq : score (liftedWeight a b v) s = score a s := by
      apply Finset.sum_congr rfl
      intro u hu
      by_cases huv : u = v
      · simp [indicator, liftedWeight, huv, hvs]
      · simp [liftedWeight, huv]
    rw [heq]
    exact hbase s hs hvs

theorem stableMixture_universal_vertex_lift (G : SimpleGraph V) (v : V)
    (hv : UniversalVertex G v) (a : V → ℝ) (b : ℝ)
    (hbase : ∀ s, Independent G s → v ∉ s → score a s ≤ b)
    (rho : V → ℝ) (hrho : StableMixture G rho) :
    dot (liftedWeight a b v) rho ≤ b := by
  obtain ⟨mu, hmu, rfl⟩ := hrho
  exact stableWeights_score_bound G mu hmu (liftedWeight a b v) b
    (universal_vertex_lift_selection G v hv a b hbase)

/-- Named RPM row consumer, reached through the complete-choice face bridge. -/
theorem rpm_universal_vertex_lift (G : SimpleGraph V) (budget : V → T)
    (hclique : BudgetCliques G budget) (v : V) (hv : UniversalVertex G v)
    (a : V → ℝ) (b : ℝ)
    (hbase : ∀ s, Independent G s → v ∉ s → score a s ≤ b)
    (rho : V → ℝ) (hrho : RPMMixture G budget rho) :
    dot (liftedWeight a b v) rho ≤ b := by
  exact stableMixture_universal_vertex_lift G v hv a b hbase rho
    ((rpm_iff_stableMixture_and_budgetAddingUp G budget hclique rho).mp hrho).1

/-! ## Unfiltered candidate table and its explicit observed-only H system

The latent index is ALL `Finset V`, not a list of independent or WARP types.
Each conflict edge supplies a zero pair moment. Positivity makes these local
equations force stable support. Observed budget adding-up then forces full
choice. The final rows are the literal output of the proved finite elimination
algorithm applied to the fixed coefficient system below.
-/

def pairMoment (mu : Finset V → ℝ) (u v : V) : ℝ :=
  ∑ s, (indicator s u * indicator s v) * mu s

def ZeroConflictMoments (G : SimpleGraph V) (mu : Finset V → ℝ) : Prop :=
  ∀ u v, G.Adj u v → pairMoment mu u v = 0

def LocalWeights (G : SimpleGraph V) (mu : Finset V → ℝ) : Prop :=
  (∀ s, 0 ≤ mu s) ∧ (∑ s, mu s = 1) ∧ ZeroConflictMoments G mu

def PrimitiveJointLaw (G : SimpleGraph V) (rho : V → ℝ) : Prop :=
  ∃ mu : Finset V → ℝ, LocalWeights G mu ∧ marginal mu = rho

omit [Fintype V] in
theorem indicator_nonnegative (s : Finset V) (v : V) : 0 ≤ indicator s v := by
  by_cases hv : v ∈ s <;> simp [indicator, hv]

theorem stableWeights_iff_localWeights (G : SimpleGraph V) (mu : Finset V → ℝ) :
    StableWeights G mu ↔ LocalWeights G mu := by
  constructor
  · intro hmu
    refine ⟨hmu.1, hmu.2.1, ?_⟩
    intro u v huv
    apply Finset.sum_eq_zero
    intro s hs
    by_cases hz : mu s = 0
    · simp [hz]
    · by_cases hu : u ∈ s
      · have hv : v ∉ s := fun hv => hmu.2.2 s hz u hu v hv huv
        simp [indicator, hv]
      · simp [indicator, hu]
  · intro hmu
    refine ⟨hmu.1, hmu.2.1, ?_⟩
    intro s hmass u hu v hv huv
    have hnonneg : ∀ r : Finset V,
        0 ≤ (indicator r u * indicator r v) * mu r := fun r =>
      mul_nonneg (mul_nonneg (indicator_nonnegative r u) (indicator_nonnegative r v))
        (hmu.1 r)
    have hterm : (indicator s u * indicator s v) * mu s = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun r _ => hnonneg r)).mp
        (hmu.2.2 u v huv) s (Finset.mem_univ s)
    exact hmass (by simpa [indicator, hu, hv] using hterm)

theorem stableMixture_iff_primitiveJointLaw (G : SimpleGraph V) (rho : V → ℝ) :
    StableMixture G rho ↔ PrimitiveJointLaw G rho := by
  simp only [StableMixture, PrimitiveJointLaw, stableWeights_iff_localWeights]

theorem rpm_iff_primitiveJointLaw_and_budgetAddingUp
    (G : SimpleGraph V) (budget : V → T) (hclique : BudgetCliques G budget)
    (rho : V → ℝ) :
    RPMMixture G budget rho ↔ PrimitiveJointLaw G rho ∧ BudgetAddingUp budget rho := by
  rw [rpm_iff_stableMixture_and_budgetAddingUp G budget hclique,
    stableMixture_iff_primitiveJointLaw]

abbrev Observable (V : Type*) := Option V → ℝ
abbrev Equation (V : Type*) := Unit ⊕ (V ⊕ (V × V))
abbrev Constraint (V : Type*) := Finset V ⊕ (Equation V × Bool)

def observedData (rho : V → ℝ) : Observable V :=
  fun i => match i with | none => 1 | some v => rho v

noncomputable def equationNormal (G : SimpleGraph V) : Equation V → Finset V → ℝ
  | .inl _, _ => 1
  | .inr (.inl v), s => indicator s v
  | .inr (.inr (u, v)), s =>
      if G.Adj u v then indicator s u * indicator s v else 0

def equationBound : Equation V → Observable V →ₗ[ℝ] ℝ
  | .inl _ => LinearMap.proj none
  | .inr (.inl v) => LinearMap.proj (some v)
  | .inr (.inr _) => 0

def direction (positive : Bool) : ℝ := if positive then 1 else -1

noncomputable def constraintNormal (G : SimpleGraph V) : Constraint V → Finset V → ℝ
  | .inl s₀, s => if s = s₀ then 1 else 0
  | .inr (e, positive), s => direction positive * equationNormal G e s

def constraintBound : Constraint V → Observable V →ₗ[ℝ] ℝ
  | .inl _ => 0
  | .inr (e, positive) => direction positive • equationBound e

theorem constraints_iff_nonnegative_equations (G : SimpleGraph V)
    (mu : Finset V → ℝ) (data : Observable V) :
    (∀ r : Constraint V, constraintBound r data ≤ ∑ s, constraintNormal G r s * mu s) ↔
      (∀ s, 0 ≤ mu s) ∧
        (∀ e : Equation V, equationBound e data = ∑ s, equationNormal G e s * mu s) := by
  classical
  constructor
  · intro h
    constructor
    · intro s
      simpa [constraintBound, constraintNormal] using h (.inl s)
    · intro e
      have hp := h (.inr (e, true))
      have hn := h (.inr (e, false))
      simp [constraintBound, constraintNormal, direction] at hp
      simp [constraintBound, constraintNormal, direction, neg_mul,
        Finset.sum_neg_distrib] at hn
      linarith
  · rintro ⟨hnonneg, heq⟩ r
    rcases r with s | ⟨e, positive⟩
    · simpa [constraintBound, constraintNormal] using hnonneg s
    · cases positive <;>
        simp [constraintBound, constraintNormal, direction, neg_mul,
          Finset.sum_neg_distrib, ← heq e]

theorem equations_iff_localWeights_and_marginal (G : SimpleGraph V)
    (mu : Finset V → ℝ) (rho : V → ℝ) :
    ((∀ s, 0 ≤ mu s) ∧
      (∀ e : Equation V, equationBound e (observedData rho) =
        ∑ s, equationNormal G e s * mu s)) ↔
      LocalWeights G mu ∧ marginal mu = rho := by
  classical
  constructor
  · rintro ⟨hnonneg, heq⟩
    refine ⟨⟨hnonneg, ?_, ?_⟩, ?_⟩
    · simpa [equationBound, equationNormal, observedData] using (heq (.inl ())).symm
    · intro u v huv
      simpa [equationBound, equationNormal, huv, pairMoment] using
        (heq (.inr (.inr (u, v)))).symm
    · funext v
      simpa [equationBound, equationNormal, observedData, marginal, mul_comm] using
        (heq (.inr (.inl v))).symm
  · rintro ⟨hmu, hrho⟩
    refine ⟨hmu.1, ?_⟩
    intro e
    rcases e with tag | v | ⟨u, v⟩
    · simpa [equationBound, equationNormal, observedData] using hmu.2.1.symm
    · simpa [equationBound, equationNormal, observedData, marginal, mul_comm] using
        (congrFun hrho v).symm
    · by_cases huv : G.Adj u v
      · simpa [equationBound, equationNormal, huv, pairMoment] using
          (hmu.2.2 u v huv).symm
      · simp [equationBound, equationNormal, huv]

theorem primitiveJointLaw_iff_constraints (G : SimpleGraph V) (rho : V → ℝ) :
    PrimitiveJointLaw G rho ↔
      ∃ mu : Finset V → ℝ,
        ∀ r : Constraint V, constraintBound r (observedData rho) ≤
          ∑ s, constraintNormal G r s * mu s := by
  simp only [PrimitiveJointLaw, constraints_iff_nonnegative_equations,
    equations_iff_localWeights_and_marginal]

/-- A literal geometry-only finite H family. It does not enumerate admissible
WARP columns, optimize support functionals, or assert existence of a hull. -/
noncomputable def rpmObservedRows (G : SimpleGraph V) :
    Finset (Observable V →ₗ[ℝ] ℝ) :=
  RPMFourierMotzkin.observedRows
    (RPMFourierMotzkin.indexedRows (constraintNormal G) constraintBound
      (Fintype.equivFin (Finset V)))

theorem primitiveJointLaw_iff_observed_H (G : SimpleGraph V) (rho : V → ℝ) :
    PrimitiveJointLaw G rho ↔ ∀ h ∈ rpmObservedRows G, 0 ≤ h (observedData rho) := by
  rw [primitiveJointLaw_iff_constraints]
  exact RPMFourierMotzkin.indexed_exists_iff_observedRows
    (constraintNormal G) constraintBound (Fintype.equivFin (Finset V)) (observedData rho)

/-- End-to-end observable RPM H consumer. Budget adding-up is explicit in
the result, rather than a hidden feasibility or existence assumption. -/
theorem rpm_iff_budgetAddingUp_and_observed_H
    (G : SimpleGraph V) (budget : V → T) (hclique : BudgetCliques G budget)
    (rho : V → ℝ) :
    RPMMixture G budget rho ↔
      BudgetAddingUp budget rho ∧
        (∀ h ∈ rpmObservedRows G, 0 ≤ h (observedData rho)) := by
  rw [rpm_iff_primitiveJointLaw_and_budgetAddingUp G budget hclique,
    primitiveJointLaw_iff_observed_H, and_comm]

/-- The canonical observed strict-price-sign conflict graph: same-budget
events conflict, as do distinct mutually affordable cross-budget events. -/
def patchConflictGraph (budget : V → T) (affordable : V → T → Prop) : SimpleGraph V where
  Adj u v := u ≠ v ∧ (budget u = budget v ∨
    (affordable u (budget v) ∧ affordable v (budget u)))
  symm := by
    constructor
    intro u v h
    refine ⟨h.1.symm, ?_⟩
    rcases h.2 with hb | ⟨hu, hv⟩
    · exact Or.inl hb.symm
    · exact Or.inr ⟨hv, hu⟩
  loopless := by
    constructor
    intro u h
    exact h.1 rfl

omit [Fintype V] [DecidableEq V] [DecidableEq T] in
theorem patchConflictGraph_budgetCliques (budget : V → T)
    (affordable : V → T → Prop) : BudgetCliques (patchConflictGraph budget affordable) budget := by
  intro u v hne hb
  exact ⟨hne, Or.inl hb⟩

/-- Full finite RPM characterization specialized to actual budget labels and
strict affordability signatures, with no graph-clique premise left to supply. -/
theorem price_sign_RPM_iff_explicit_H (budget : V → T)
    (affordable : V → T → Prop) (rho : V → ℝ) :
    RPMMixture (patchConflictGraph budget affordable) budget rho ↔
      BudgetAddingUp budget rho ∧
        (∀ h ∈ rpmObservedRows (patchConflictGraph budget affordable),
          0 ≤ h (observedData rho)) :=
  rpm_iff_budgetAddingUp_and_observed_H (patchConflictGraph budget affordable) budget
    (patchConflictGraph_budgetCliques budget affordable) rho

#print axioms independent_blockCard_le_one
#print axioms positive_mass_selection_complete
#print axioms rpm_iff_stableMixture_and_budgetAddingUp
#print axioms universal_vertex_lift_selection
#print axioms stableMixture_universal_vertex_lift
#print axioms rpm_universal_vertex_lift
#print axioms stableWeights_iff_localWeights
#print axioms primitiveJointLaw_iff_observed_H
#print axioms rpm_iff_budgetAddingUp_and_observed_H
#print axioms price_sign_RPM_iff_explicit_H

end LawsOfDemand.RPMGraphDual
