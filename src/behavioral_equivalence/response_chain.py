"""One-step labeled point chains for fixed-context binary response emissions."""

import numpy as np
from .core import Chain
from .response_refinement import validate_latent_partition


def binary_response_chain(probabilities, initial=None) -> Chain:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Finite nonempty binary response probability vector required")
    k = len(p)
    transitions = np.zeros((k + 2, k + 2))
    transitions[:k, k] = p
    transitions[:k, k + 1] = 1 - p
    transitions[k, k] = 1
    transitions[k + 1, k + 1] = 1
    mu = np.zeros(k + 2)
    prior = np.full(k, 1 / k) if initial is None else np.asarray(initial, dtype=float)
    if prior.shape != (k,):
        raise ValueError("One initial mass per pending state required")
    mu[:k] = prior
    return Chain(transitions, ("pending",) * k + ("positive", "negative"), mu)


def response_chain_pair(parent, abstract, blocks):
    parent = np.asarray(parent, dtype=float)
    abstract = np.asarray(abstract, dtype=float)
    if parent.ndim != 1 or abstract.shape != (len(blocks),):
        raise ValueError("Aligned parent and abstract response vectors required")
    validate_latent_partition(blocks, len(parent))
    concrete = binary_response_chain(parent)
    quotient = binary_response_chain(abstract, np.array([len(b) / len(parent) for b in blocks]))
    relation = blocks + ((len(parent),), (len(parent) + 1,))
    return concrete, quotient, relation
