import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Module.LinearMap.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Exact finite Fourier--Motzkin projection

The generic algebraic projection proofs below are reused, with the namespace
changed and the choice-specific sections removed, from the user's local files
`entanglement_bell_inequalities/BellChoice/RecursiveElimination.lean` and
`entanglement_bell_inequalities/BellChoice/RecursiveHSystem.lean`.

Zero rows are retained and every positive-negative pair is combined using
nonnegative weights. The generated coefficients contain no division. The
finite recursion is semantically exact, including empty systems and empty
sign classes. Real-coefficient Finsets use classical equality; this file
does not claim an executable rational-data implementation or facet minimality.
-/

open scoped BigOperators

namespace LawsOfDemand.RPMFourierMotzkin

theorem pair_combination_identity (ap an kp kn t : ℝ) :
    (-an) * (ap * t + kp) + ap * (an * t + kn) =
      (-an) * kp + ap * kn := by
  ring

theorem positive_constraint_iff (a k t : ℝ) (ha : 0 < a) :
    0 ≤ a * t + k ↔ -k / a ≤ t := by
  rw [div_le_iff₀ ha]
  constructor <;> intro h <;> nlinarith

theorem negative_constraint_iff (a k t : ℝ) (ha : a < 0) :
    0 ≤ a * t + k ↔ t ≤ k / (-a) := by
  rw [le_div_iff₀ (neg_pos.mpr ha)]
  constructor <;> intro h <;> nlinarith

theorem pair_condition_iff (ap an kp kn : ℝ) (hp : 0 < ap) (hn : an < 0) :
    0 ≤ (-an) * kp + ap * kn ↔ -kp / ap ≤ kn / (-an) := by
  rw [div_le_div_iff₀ hp (neg_pos.mpr hn)]
  constructor <;> intro h <;> nlinarith

/-- Exact one-coordinate elimination in nonnegative-row convention. -/
theorem one_scalar_elimination_iff {R : Type*} [Fintype R]
    (alpha residual : R → ℝ) :
    (∃ t : ℝ, ∀ r, 0 ≤ alpha r * t + residual r) ↔
      (∀ r, alpha r = 0 → 0 ≤ residual r) ∧
      (∀ p n, 0 < alpha p → alpha n < 0 →
        0 ≤ (-alpha n) * residual p + alpha p * residual n) := by
  classical
  constructor
  · rintro ⟨t, ht⟩
    constructor
    · intro r hr
      simpa only [hr, zero_mul, zero_add] using ht r
    · intro p n hp hn
      have hcomb := add_nonneg
        (mul_nonneg (le_of_lt (neg_pos.mpr hn)) (ht p))
        (mul_nonneg (le_of_lt hp) (ht n))
      rw [pair_combination_identity] at hcomb
      exact hcomb
  · rintro ⟨hzero, hpair⟩
    let pos : Finset R := Finset.univ.filter (fun r => 0 < alpha r)
    let neg : Finset R := Finset.univ.filter (fun r => alpha r < 0)
    by_cases hpos : pos.Nonempty
    · let t : ℝ := pos.sup' hpos (fun r => -residual r / alpha r)
      refine ⟨t, ?_⟩
      intro r
      rcases lt_trichotomy (alpha r) 0 with hr | hr | hr
      · apply (negative_constraint_iff (alpha r) (residual r) t hr).mpr
        apply Finset.sup'_le hpos
        intro p hp
        have hp' : 0 < alpha p := (Finset.mem_filter.mp hp).2
        exact (pair_condition_iff (alpha p) (alpha r) (residual p)
          (residual r) hp' hr).mp (hpair p r hp' hr)
      · simpa only [hr, zero_mul, zero_add] using hzero r hr
      · apply (positive_constraint_iff (alpha r) (residual r) t hr).mpr
        exact Finset.le_sup' (fun p => -residual p / alpha p)
          (Finset.mem_filter.mpr ⟨Finset.mem_univ r, hr⟩)
    · have hnotpos (r : R) : ¬0 < alpha r := by
        intro hr
        exact hpos ⟨r, Finset.mem_filter.mpr ⟨Finset.mem_univ r, hr⟩⟩
      by_cases hneg : neg.Nonempty
      · let t : ℝ := neg.inf' hneg (fun r => residual r / (-alpha r))
        refine ⟨t, ?_⟩
        intro r
        rcases lt_trichotomy (alpha r) 0 with hr | hr | hr
        · apply (negative_constraint_iff (alpha r) (residual r) t hr).mpr
          exact Finset.inf'_le (fun n => residual n / (-alpha n))
            (Finset.mem_filter.mpr ⟨Finset.mem_univ r, hr⟩)
        · simpa only [hr, zero_mul, zero_add] using hzero r hr
        · exact (hnotpos r hr).elim
      · refine ⟨0, ?_⟩
        intro r
        rcases lt_trichotomy (alpha r) 0 with hr | hr | hr
        · exact (hneg ⟨r, Finset.mem_filter.mpr ⟨Finset.mem_univ r, hr⟩⟩).elim
        · simpa only [hr, zero_mul, zero_add] using hzero r hr
        · exact (hnotpos r hr).elim

/-- The same theorem in the usual weak-inequality convention `aᵢ t ≤ bᵢ`.
The right side is an explicit zero-row and positive-negative pair family. -/
theorem scalar_le_elimination_iff {R : Type*} [Fintype R] (a b : R → ℝ) :
    (∃ t : ℝ, ∀ i, a i * t ≤ b i) ↔
      (∀ i, a i = 0 → 0 ≤ b i) ∧
      (∀ p n, 0 < a p → a n < 0 →
        0 ≤ (-a n) * b p + a p * b n) := by
  have hs := one_scalar_elimination_iff (fun i => -a i) b
  constructor
  · rintro ⟨t, ht⟩
    have hh := hs.mp ⟨t, fun i => by nlinarith [ht i]⟩
    constructor
    · intro i hi
      exact hh.1 i (by simp [hi])
    · intro p n hp hn
      have h := hh.2 n p (neg_pos.mpr hn) (neg_lt_zero.mpr hp)
      simpa only [neg_neg, add_comm] using h
  · rintro ⟨hz, hp⟩
    have hh :
        (∀ i, -a i = 0 → 0 ≤ b i) ∧
        (∀ p n, 0 < -a p → -a n < 0 →
          0 ≤ (-(-a n)) * b p + (-a p) * b n) := by
      constructor
      · intro i hi
        exact hz i (neg_eq_zero.mp hi)
      · intro p n hpp hnn
        have h := hp n p (neg_lt_zero.mp hnn) (neg_pos.mp hpp)
        simpa only [neg_neg, add_comm] using h
    obtain ⟨t, ht⟩ := hs.mpr hh
    exact ⟨t, fun i => by nlinarith [ht i]⟩

/-- The usual affine-row convention, with the generated remaining-variable
coefficients and right-hand sides written explicitly. -/
theorem affine_le_elimination_iff {R V : Type*} [Fintype R] [Fintype V]
    (a : R → ℝ) (c : R → V → ℝ) (b : R → ℝ) (x : V → ℝ) :
    (∃ t : ℝ, ∀ i, a i * t + (∑ v, c i v * x v) ≤ b i) ↔
      (∀ i, a i = 0 → (∑ v, c i v * x v) ≤ b i) ∧
      (∀ p n, 0 < a p → a n < 0 →
        (∑ v, ((-a n) * c p v + a p * c n v) * x v) ≤
          (-a n) * b p + a p * b n) := by
  have hs := scalar_le_elimination_iff a (fun i => b i - ∑ v, c i v * x v)
  have hsum (p n : R) :
      (∑ v, ((-a n) * c p v + a p * c n v) * x v) =
        (-a n) * (∑ v, c p v * x v) + a p * (∑ v, c n v * x v) := by
    simp [add_mul, mul_assoc, Finset.sum_add_distrib, ← Finset.mul_sum]
  constructor
  · rintro ⟨t, ht⟩
    have hh := hs.mp ⟨t, fun i => by linarith [ht i]⟩
    constructor
    · intro i hi
      linarith [hh.1 i hi]
    · intro p n hp hn
      rw [hsum]
      nlinarith [hh.2 p n hp hn]
  · rintro ⟨hz, hp⟩
    have hh :
        (∀ i, a i = 0 → 0 ≤ b i - ∑ v, c i v * x v) ∧
        (∀ p n, 0 < a p → a n < 0 →
          0 ≤ (-a n) * (b p - ∑ v, c p v * x v) +
            a p * (b n - ∑ v, c n v * x v)) := by
      constructor
      · intro i hi
        linarith [hz i hi]
      · intro p n hpp hnn
        have h := hp p n hpp hnn
        rw [hsum] at h
        nlinarith
    obtain ⟨t, ht⟩ := hs.mpr hh
    exact ⟨t, fun i => by linarith [ht i]⟩

attribute [local instance] Classical.propDecidable

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- A row is nonnegative when its latent dot product plus its observable
linear expression is nonnegative. -/
structure Row (n : ℕ) (E : Type*) [AddCommGroup E] [Module ℝ E] where
  coefficient : Fin n → ℝ
  observed : E →ₗ[ℝ] ℝ

def Row.value {n : ℕ} (r : Row n E) (x : Fin n → ℝ) (data : E) : ℝ :=
  (∑ i, r.coefficient i * x i) + r.observed data

def Satisfies {n : ℕ} (rows : Finset (Row n E)) (x : Fin n → ℝ) (data : E) : Prop :=
  ∀ r ∈ rows, 0 ≤ r.value x data

def tailRow {n : ℕ} (r : Row (n + 1) E) : Row n E :=
  ⟨fun i => r.coefficient i.succ, r.observed⟩

/-- Nonnegative weights cancel the eliminated coefficient. The same explicit
transformation is applied to every remaining coefficient and observable map. -/
def pairRow {n : ℕ} (p q : Row (n + 1) E) : Row n E :=
  ⟨fun i => (-q.coefficient 0) * p.coefficient i.succ +
      p.coefficient 0 * q.coefficient i.succ,
    (-q.coefficient 0) • p.observed + p.coefficient 0 • q.observed⟩

/-- Literal finite row generation, independent of the observed data. -/
noncomputable def eliminate {n : ℕ} (rows : Finset (Row (n + 1) E)) :
    Finset (Row n E) :=
  ((rows.filter fun r => r.coefficient 0 = 0).image tailRow) ∪
    (((rows.filter fun r => 0 < r.coefficient 0) ×ˢ
      (rows.filter fun r => r.coefficient 0 < 0)).image fun pq => pairRow pq.1 pq.2)

theorem value_cons {n : ℕ} (r : Row (n + 1) E) (x : Fin n → ℝ)
    (data : E) (t : ℝ) :
    r.value (Fin.cons t x) data = r.coefficient 0 * t + (tailRow r).value x data := by
  simp [Row.value, tailRow, Fin.sum_univ_succ, add_assoc]

theorem pairRow_value {n : ℕ} (p q : Row (n + 1) E) (x : Fin n → ℝ) (data : E) :
    (pairRow p q).value x data =
      (-q.coefficient 0) * (tailRow p).value x data +
        p.coefficient 0 * (tailRow q).value x data := by
  simp [Row.value, pairRow, tailRow, add_mul, mul_assoc, Finset.sum_add_distrib,
    ← Finset.mul_sum]
  ring

/-- The generated rows exactly characterize existence of the eliminated
coordinate, with every remaining coordinate held fixed. -/
theorem eliminate_exact {n : ℕ} (rows : Finset (Row (n + 1) E))
    (x : Fin n → ℝ) (data : E) :
    (∃ t : ℝ, Satisfies rows (Fin.cons t x) data) ↔ Satisfies (eliminate rows) x data := by
  classical
  have scalar := one_scalar_elimination_iff
    (fun r : rows => r.val.coefficient 0) (fun r : rows => (tailRow r.val).value x data)
  constructor
  · rintro ⟨t, ht⟩
    have hs := scalar.mp ⟨t, fun r => by simpa only [value_cons] using ht r.val r.property⟩
    intro r hr
    rcases Finset.mem_union.mp hr with hz | hp
    · obtain ⟨r₀, hr₀, rfl⟩ := Finset.mem_image.mp hz
      exact hs.1 ⟨r₀, (Finset.mem_filter.mp hr₀).1⟩ (Finset.mem_filter.mp hr₀).2
    · obtain ⟨⟨p, q⟩, hpq, rfl⟩ := Finset.mem_image.mp hp
      obtain ⟨hp, hq⟩ := Finset.mem_product.mp hpq
      rw [pairRow_value]
      exact hs.2 ⟨p, (Finset.mem_filter.mp hp).1⟩ ⟨q, (Finset.mem_filter.mp hq).1⟩
        (Finset.mem_filter.mp hp).2 (Finset.mem_filter.mp hq).2
  · intro ht
    have hs :
        (∀ r : rows, r.val.coefficient 0 = 0 → 0 ≤ (tailRow r.val).value x data) ∧
        (∀ p q : rows, 0 < p.val.coefficient 0 → q.val.coefficient 0 < 0 →
          0 ≤ (-q.val.coefficient 0) * (tailRow p.val).value x data +
            p.val.coefficient 0 * (tailRow q.val).value x data) := by
      constructor
      · intro r hr
        exact ht (tailRow r.val) (Finset.mem_union_left _
          (Finset.mem_image.mpr ⟨r.val, Finset.mem_filter.mpr ⟨r.property, hr⟩, rfl⟩))
      · intro p q hp hq
        rw [← pairRow_value]
        apply ht
        exact Finset.mem_union_right _ (Finset.mem_image.mpr ⟨(p.val, q.val),
          Finset.mem_product.mpr ⟨Finset.mem_filter.mpr ⟨p.property, hp⟩,
            Finset.mem_filter.mpr ⟨q.property, hq⟩⟩, rfl⟩)
    obtain ⟨t, ht⟩ := scalar.mpr hs
    refine ⟨t, fun r hr => ?_⟩
    simpa only [value_cons] using ht ⟨r, hr⟩

/-- The recursion removes one latent coordinate at each call. -/
noncomputable def eliminateAll : {n : ℕ} → Finset (Row n E) → Finset (Row 0 E)
  | 0, rows => rows
  | _ + 1, rows => eliminateAll (eliminate rows)

/-- Full finite projection: the output contains no latent coordinate. -/
theorem eliminateAll_exact {n : ℕ} (rows : Finset (Row n E)) (data : E) :
    (∃ x : Fin n → ℝ, Satisfies rows x data) ↔
      ∀ r ∈ eliminateAll rows, 0 ≤ r.observed data := by
  induction n with
  | zero =>
      simp only [eliminateAll, Satisfies, Row.value, Finset.univ_eq_empty,
        Finset.sum_empty, zero_add]
      exact ⟨fun ⟨_, h⟩ => h, fun h => ⟨Fin.elim0, h⟩⟩
  | succ n ih =>
      change (∃ x : Fin (n + 1) → ℝ, Satisfies rows x data) ↔
        ∀ r ∈ eliminateAll (eliminate rows), 0 ≤ r.observed data
      rw [← ih (eliminate rows)]
      constructor
      · rintro ⟨x, hx⟩
        refine ⟨Fin.tail x, (eliminate_exact rows (Fin.tail x) data).mp ⟨x 0, ?_⟩⟩
        simpa only [Fin.cons_self_tail] using hx
      · rintro ⟨x, hx⟩
        obtain ⟨t, ht⟩ := (eliminate_exact rows x data).mpr hx
        exact ⟨Fin.cons t x, ht⟩

/-- The literal finite observed-only family of output linear maps. -/
noncomputable def observedRows {n : ℕ} (rows : Finset (Row n E)) :
    Finset (E →ₗ[ℝ] ℝ) := (eliminateAll rows).image Row.observed

theorem exists_iff_observedRows {n : ℕ} (rows : Finset (Row n E)) (data : E) :
    (∃ x : Fin n → ℝ, Satisfies rows x data) ↔
      ∀ h ∈ observedRows rows, 0 ≤ h data := by
  rw [eliminateAll_exact]
  simp only [observedRows, Finset.forall_mem_image]

section IndexedSystem

variable {R V : Type*} [Fintype R] [Fintype V] {n : ℕ}

/-- Reindex any finite system `bound(data) ≤ normal · x` by the specified
elimination order. -/
def indexedRow (normal : R → V → ℝ) (bound : R → E →ₗ[ℝ] ℝ)
    (order : V ≃ Fin n) (r : R) : Row n E :=
  ⟨fun k => normal r (order.symm k), -bound r⟩

noncomputable def indexedRows (normal : R → V → ℝ) (bound : R → E →ₗ[ℝ] ℝ)
    (order : V ≃ Fin n) : Finset (Row n E) :=
  Finset.univ.image (indexedRow normal bound order)

omit [Fintype R] in
theorem indexedRow_value (normal : R → V → ℝ) (bound : R → E →ₗ[ℝ] ℝ)
    (order : V ≃ Fin n) (r : R) (x : Fin n → ℝ) (data : E) :
    (indexedRow normal bound order r).value x data =
      (∑ v, normal r v * x (order v)) - bound r data := by
  have hs := order.symm.sum_comp (fun v => normal r v * x (order v))
  simpa only [Row.value, indexedRow, LinearMap.neg_apply,
    Equiv.apply_symm_apply, sub_eq_add_neg] using congrArg (fun t => t - bound r data) hs

theorem indexedRows_satisfies (normal : R → V → ℝ) (bound : R → E →ₗ[ℝ] ℝ)
    (order : V ≃ Fin n) (x : Fin n → ℝ) (data : E) :
    Satisfies (indexedRows normal bound order) x data ↔
      ∀ r, bound r data ≤ ∑ v, normal r v * x (order v) := by
  simp only [Satisfies, indexedRows, Finset.forall_mem_image, Finset.mem_univ,
    true_implies, indexedRow_value, sub_nonneg]

/-- Exact finite observed-only characterization of any finite weak linear
system, for every coordinate order, with a literal recursively generated
output family. No projection-completeness assumption occurs in the signature. -/
theorem indexed_exists_iff_observedRows (normal : R → V → ℝ)
    (bound : R → E →ₗ[ℝ] ℝ) (order : V ≃ Fin n) (data : E) :
    (∃ x : V → ℝ, ∀ r, bound r data ≤ ∑ v, normal r v * x v) ↔
      ∀ h ∈ observedRows (indexedRows normal bound order), 0 ≤ h data := by
  rw [← exists_iff_observedRows]
  constructor
  · rintro ⟨x, hx⟩
    refine ⟨fun k => x (order.symm k), (indexedRows_satisfies _ _ _ _ _).mpr ?_⟩
    simpa only [Equiv.symm_apply_apply] using hx
  · rintro ⟨x, hx⟩
    exact ⟨fun v => x (order v), (indexedRows_satisfies _ _ _ _ _).mp hx⟩

end IndexedSystem

#print axioms one_scalar_elimination_iff
#print axioms scalar_le_elimination_iff
#print axioms affine_le_elimination_iff
#print axioms eliminate_exact
#print axioms eliminateAll_exact
#print axioms exists_iff_observedRows
#print axioms indexed_exists_iff_observedRows

end LawsOfDemand.RPMFourierMotzkin
