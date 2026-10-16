import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

set_option autoImplicit false
open scoped BigOperators

namespace BehavioralCoupling

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

noncomputable def overlap (p q : ι → ℝ) : ℝ := ∑ i, min (p i) (q i)

noncomputable def totalVariation (p q : ι → ℝ) : ℝ := (∑ i, |p i - q i|) / 2

lemma absolute_difference (a b : ℝ) : |a - b| = a + b - 2 * min a b := by
  rcases le_total a b with h | h
  · rw [min_eq_left h, abs_of_nonpos (sub_nonpos.mpr h)]
    ring
  · rw [min_eq_right h, abs_of_nonneg (sub_nonneg.mpr h)]
    ring

lemma overlap_identity (p q : ι → ℝ)
    (psum : ∑ i, p i = 1) (qsum : ∑ i, q i = 1) :
    overlap p q = 1 - totalVariation p q := by
  have hs : (∑ i, |p i - q i|) = 2 - 2 * overlap p q := by
    simp_rw [absolute_difference]
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
    rw [← Finset.mul_sum, psum, qsum]
    simp [overlap] <;> ring
  unfold totalVariation
  rw [hs]
  ring

omit [DecidableEq ι] in
lemma overlap_le_one (p q : ι → ℝ) (psum : ∑ i, p i = 1) : overlap p q ≤ 1 := by
  calc
    overlap p q ≤ ∑ i, p i := Finset.sum_le_sum (fun i _ => min_le_left (p i) (q i))
    _ = 1 := psum

lemma residuals_disjoint (a b : ℝ) : (a - min a b) * (b - min a b) = 0 := by
  rcases le_total a b with h | h
  · simp [min_eq_left h]
  · simp [min_eq_right h]

noncomputable def residualCoupling (p q : ι → ℝ) (i j : ι) : ℝ :=
  (if i = j then min (p i) (q i) else 0) +
    ((p i - min (p i) (q i)) * (q j - min (p j) (q j))) / (1 - overlap p q)

lemma residualCoupling_nonnegative (p q : ι → ℝ)
    (pnonneg : ∀ i, 0 ≤ p i) (qnonneg : ∀ i, 0 ≤ q i)
    (denominator : 0 < 1 - overlap p q) (i j : ι) :
    0 ≤ residualCoupling p q i j := by
  unfold residualCoupling
  apply add_nonneg
  · split_ifs
    · exact le_min (pnonneg i) (qnonneg i)
    · exact le_rfl
  · exact div_nonneg
      (mul_nonneg (sub_nonneg.mpr (min_le_left _ _))
        (sub_nonneg.mpr (min_le_right _ _))) (le_of_lt denominator)

lemma residualCoupling_row (p q : ι → ℝ) (qsum : ∑ j, q j = 1)
    (denominator : 0 < 1 - overlap p q) (i : ι) :
    (∑ j, residualCoupling p q i j) = p i := by
  have hq : (∑ j, (q j - min (p j) (q j))) = 1 - overlap p q := by
    rw [Finset.sum_sub_distrib, qsum]
    rfl
  unfold residualCoupling
  rw [Finset.sum_add_distrib]
  have hd : (∑ j, if i = j then min (p i) (q i) else 0) = min (p i) (q i) := by simp
  rw [hd, ← Finset.sum_div, ← Finset.mul_sum, hq]
  field_simp [ne_of_gt denominator]

lemma residualCoupling_column (p q : ι → ℝ) (psum : ∑ i, p i = 1)
    (denominator : 0 < 1 - overlap p q) (j : ι) :
    (∑ i, residualCoupling p q i j) = q j := by
  have hp : (∑ i, (p i - min (p i) (q i))) = 1 - overlap p q := by
    rw [Finset.sum_sub_distrib, psum]
    rfl
  unfold residualCoupling
  rw [Finset.sum_add_distrib]
  have hd : (∑ i, if i = j then min (p i) (q i) else 0) = min (p j) (q j) := by simp
  rw [hd, ← Finset.sum_div, ← Finset.sum_mul, hp]
  field_simp [ne_of_gt denominator]

lemma residualCoupling_diagonal (p q : ι → ℝ) :
    (∑ i, residualCoupling p q i i) = overlap p q := by
  simp [residualCoupling, residuals_disjoint, overlap]

/-- Explicit finite maximal coupling: nonnegative joint weights, both exact
marginals and diagonal probability equal to one minus total variation. -/
theorem finite_maximal_coupling (p q : ι → ℝ)
    (pnonneg : ∀ i, 0 ≤ p i) (qnonneg : ∀ i, 0 ≤ q i)
    (psum : ∑ i, p i = 1) (qsum : ∑ i, q i = 1) :
    ∃ c : ι → ι → ℝ,
      (∀ i j, 0 ≤ c i j) ∧
      (∀ i, ∑ j, c i j = p i) ∧
      (∀ j, ∑ i, c i j = q j) ∧
      (∑ i, c i i) = 1 - totalVariation p q := by
  by_cases full : overlap p q = 1
  · have hpzero : (∑ i, (p i - min (p i) (q i))) = 0 := by
      rw [Finset.sum_sub_distrib]
      change (∑ i, p i) - overlap p q = 0
      rw [psum, full]
      ring
    have hqzero : (∑ i, (q i - min (p i) (q i))) = 0 := by
      rw [Finset.sum_sub_distrib]
      change (∑ i, q i) - overlap p q = 0
      rw [qsum, full]
      ring
    have equal : ∀ i, p i = q i := by
      intro i
      have hp := (Finset.sum_eq_zero_iff_of_nonneg
        (fun j _ => sub_nonneg.mpr (min_le_left (p j) (q j)))).mp hpzero i (Finset.mem_univ i)
      have hq := (Finset.sum_eq_zero_iff_of_nonneg
        (fun j _ => sub_nonneg.mpr (min_le_right (p j) (q j)))).mp hqzero i (Finset.mem_univ i)
      linarith
    refine ⟨(fun i j => if i = j then p i else 0), ?_, ?_, ?_, ?_⟩
    · intro i j
      dsimp only
      split_ifs
      · exact pnonneg i
      · exact le_rfl
    · intro i
      simp
    · intro j
      simpa using equal j
    · simpa [psum, full] using overlap_identity p q psum qsum
  · have less : overlap p q < 1 := by
      rcases lt_or_eq_of_le (overlap_le_one p q psum) with h | h
      · exact h
      · exact False.elim (full h)
    have denominator : 0 < 1 - overlap p q := by linarith
    refine ⟨residualCoupling p q, ?_, ?_, ?_, ?_⟩
    · exact residualCoupling_nonnegative p q pnonneg qnonneg denominator
    · exact residualCoupling_row p q qsum denominator
    · exact residualCoupling_column p q psum denominator
    · rw [residualCoupling_diagonal]
      exact overlap_identity p q psum qsum

#print axioms finite_maximal_coupling

end BehavioralCoupling
