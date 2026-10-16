import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Linarith

set_option autoImplicit false
open scoped BigOperators

namespace BehavioralEvents

variable {Ω : Type*} [Fintype Ω]

lemma finite_absolute_sum (f : Ω → ℝ) (s : Finset Ω) :
    |∑ x ∈ s, f x| ≤ ∑ x ∈ s, |f x| := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert x s absent inductionHypothesis =>
    rw [Finset.sum_insert absent, Finset.sum_insert absent]
    exact (abs_add_le (f x) (∑ y ∈ s, f y)).trans
      (add_le_add_left inductionHypothesis |f x|)

/-- On a finite joint path space, equal observable events on matched paths can
only disagree on unmatched probability mass. No independence is assumed. -/
theorem event_difference_from_matched_paths
    (weight : Ω → ℝ) (eventA eventB matched : Ω → Bool)
    (nonnegative : ∀ x, 0 ≤ weight x)
    (normalized : ∑ x, weight x = 1)
    (eventsAgree : ∀ x, matched x = true → eventA x = eventB x) :
    |(∑ x, if eventA x then weight x else 0) -
      (∑ x, if eventB x then weight x else 0)| ≤
      1 - (∑ x, if matched x then weight x else 0) := by
  have pointwise : ∀ x,
      |(if eventA x then weight x else 0) - (if eventB x then weight x else 0)| ≤
      (if matched x then 0 else weight x) := by
    intro x
    cases hm : matched x with
    | false =>
      cases ha : eventA x <;> cases hb : eventB x <;>
        simp [hm, ha, hb, abs_of_nonneg (nonnegative x), nonnegative x]
    | true =>
      have agree := eventsAgree x hm
      simp [hm, agree]
  have complement :
      (∑ x, if matched x then 0 else weight x) +
        (∑ x, if matched x then weight x else 0) = 1 := by
    rw [← Finset.sum_add_distrib]
    calc
      (∑ x, ((if matched x then 0 else weight x) +
        (if matched x then weight x else 0))) = ∑ x, weight x := by
          apply Finset.sum_congr rfl
          intro x _
          cases matched x <;> simp
      _ = 1 := normalized
  have bound :
      |(∑ x, if eventA x then weight x else 0) -
        (∑ x, if eventB x then weight x else 0)| ≤
        ∑ x, if matched x then 0 else weight x := by
    rw [← Finset.sum_sub_distrib]
    exact (finite_absolute_sum
      (fun x => (if eventA x then weight x else 0) -
        (if eventB x then weight x else 0)) Finset.univ).trans
      (Finset.sum_le_sum (fun x _ => pointwise x))
  linarith

#print axioms event_difference_from_matched_paths

end BehavioralEvents
