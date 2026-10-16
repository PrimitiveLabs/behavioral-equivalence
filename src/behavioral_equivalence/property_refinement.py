"""Property-directed latent refinement, with independent numeric acceptance."""

from fractions import Fraction
import numpy as np
from .core import Chain, Partition, check_partition, quotient
from .decision_model import decision_partition
from .property_bounds import LabelEvent, event_predictions, certify_properties


def property_guided_latent_refinement(
    parent: Chain,
    initial_blocks: Partition,
    events: tuple[LabelEvent, ...],
    tolerance: float,
    bits: int = 48,
) -> tuple[Chain, Partition, dict]:
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Finite probability-error tolerance in [0,1] required")
    if parent.size % 3 or parent.labels != tuple(
        label for _ in range(parent.size // 3) for label in ("pending", "negative", "positive")
    ):
        raise ValueError("Expected a 3K decision chain")
    check_partition(parent, decision_partition(initial_blocks))
    budget = Fraction.from_float(float(tolerance))
    origins = tuple(range(0, parent.size, 3))
    truth = event_predictions(parent, events)
    blocks = initial_blocks
    trace = []
    while True:
        lifted = decision_partition(blocks)
        abstract, _ = quotient(parent, lifted)
        mapping = {s: b for b, block in enumerate(lifted) for s in block}
        candidate = event_predictions(abstract, events)
        differences = np.abs(truth[:, origins] - candidate[:, [mapping[s] for s in origins]])
        event_index, origin_index = np.unravel_index(differences.argmax(), differences.shape)
        witness = origins[origin_index]
        screening_error = float(differences[event_index, origin_index])
        entry = {
            "latent_states": len(blocks),
            "screening_error": screening_error,
            "event_index": int(event_index),
            "witness_state": witness,
            "numeric_acceptance": "not attempted",
        }
        trace.append(entry)
        if screening_error <= tolerance or len(blocks) == parent.size // 3:
            certificate = certify_properties(parent, abstract, lifted, events, bits, origins)
            bounds = [
                Fraction(
                    int(r["error_bound"]["numerator_hex"], 16),
                    int(r["error_bound"]["denominator_hex"], 16),
                )
                for r in certificate["properties"]
            ]
            worst = max(range(len(bounds)), key=bounds.__getitem__)
            entry.update(
                numeric_acceptance="passed" if bounds[worst] <= budget else "refine",
                verified_max_error_for_display=float(bounds[worst]),
            )
            if bounds[worst] <= budget:
                return (
                    abstract,
                    blocks,
                    {
                        "certificate": certificate,
                        "refinement_trace": trace,
                        "parent_latent_states": parent.size // 3,
                        "reduced_latent_states": len(blocks),
                        "latent_compression_ratio": (parent.size // 3) / len(blocks),
                        "probability_error_budget": tolerance,
                        "scope": "specified finite property suite from pending prediction states; no unlisted-property, bisimulation or human-process guarantee",
                    },
                )
            witness = certificate["properties"][worst]["witness_state"]
        latent = witness // 3
        selected = next(i for i, block in enumerate(blocks) if latent in block)
        if len(blocks[selected]) == 1:
            nontrivial = [i for i, b in enumerate(blocks) if len(b) > 1]
            if not nontrivial:
                raise AssertionError("Identity quotient must preserve the specified properties")

            def spread(i):
                return float(np.ptp(truth[:, [3 * s for s in blocks[i]]], axis=1).max()), len(
                    blocks[i]
                )

            selected = max(nontrivial, key=spread)
            latent = blocks[selected][0]
        remainder = tuple(s for s in blocks[selected] if s != latent)
        blocks = blocks[:selected] + (remainder, (latent,)) + blocks[selected + 1 :]
