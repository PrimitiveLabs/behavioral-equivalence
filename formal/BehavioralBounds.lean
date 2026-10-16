import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

set_option autoImplicit false

namespace BehavioralBounds

/-- Induction from a uniform conditional block-match recurrence.
The probabilistic coupling construction supplying these hypotheses is separate. -/
theorem match_probability_lower_bound
    (epsilon eta : ℝ) (matchProbability : ℕ → ℝ)
    (epsilon_le_one : epsilon ≤ 1)
    (initial : 1 - eta ≤ matchProbability 0)
    (step : ∀ n, (1 - epsilon) * matchProbability n ≤ matchProbability (n + 1)) :
    ∀ n, (1 - eta) * (1 - epsilon) ^ n ≤ matchProbability n := by
  intro n
  induction n with
  | zero => simpa using initial
  | succ n inductionHypothesis =>
    calc
      (1 - eta) * (1 - epsilon) ^ (n + 1) =
          (1 - epsilon) * ((1 - eta) * (1 - epsilon) ^ n) := by
            rw [pow_succ]
            ring
      _ ≤ (1 - epsilon) * matchProbability n :=
        mul_le_mul_of_nonneg_left inductionHypothesis (sub_nonneg.mpr epsilon_le_one)
      _ ≤ matchProbability (n + 1) := step n

/-- Observable event discrepancies are bounded once path matching and its
indicator-disagreement inequality have been established. -/
theorem finite_horizon_event_error
    (epsilon eta p q : ℝ) (matchProbability : ℕ → ℝ) (horizon : ℕ)
    (epsilon_le_one : epsilon ≤ 1)
    (initial : 1 - eta ≤ matchProbability 0)
    (step : ∀ n, (1 - epsilon) * matchProbability n ≤ matchProbability (n + 1))
    (eventDisagreement : |p - q| ≤ 1 - matchProbability horizon) :
    |p - q| ≤ 1 - (1 - eta) * (1 - epsilon) ^ horizon := by
  have matched := match_probability_lower_bound epsilon eta matchProbability
    epsilon_le_one initial step horizon
  linarith

/-- Robust-model uncertainty and a certified absolute abstraction discrepancy
compose by expansion and intersection with the probability range. -/
theorem uncertainty_interval_composition
    (p q lower upper delta : ℝ)
    (probability_nonnegative : 0 ≤ p) (probability_le_one : p ≤ 1)
    (lower_bound : lower ≤ q) (upper_bound : q ≤ upper)
    (discrepancy : |p - q| ≤ delta) :
    max 0 (lower - delta) ≤ p ∧ p ≤ min 1 (upper + delta) := by
  have sides := abs_le.mp discrepancy
  constructor
  · exact max_le probability_nonnegative (by linarith [sides.1])
  · exact le_min probability_le_one (by linarith [sides.2])

/-- A zero discrepancy allowance implies exact event-probability equality. -/
theorem zero_event_discrepancy (p q : ℝ) (bound : |p - q| ≤ 0) : p = q := by
  have zero : |p - q| = 0 := le_antisymm bound (abs_nonneg (p - q))
  exact sub_eq_zero.mp (abs_eq_zero.mp zero)

#print axioms match_probability_lower_bound
#print axioms finite_horizon_event_error
#print axioms uncertainty_interval_composition
#print axioms zero_event_discrepancy

end BehavioralBounds
