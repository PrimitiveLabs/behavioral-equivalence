"""Seeded finite mechanism families with a known duplicated-state partition."""

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .core import Chain, Partition

GENERATOR_REVISION = "mechanism-parameterized-v3"

FAMILIES = ("preference", "belief", "trust", "patience", "memory")


@dataclass(frozen=True)
class World:
    chain: Chain
    partition: Partition
    family: str
    seed: int
    perturbation: float
    features: tuple[tuple[int, int], ...]
    parameters: tuple[tuple[str, int], ...]


def generate(family: str, size: int, seed: int, perturbation: float = 0) -> World:
    if family not in FAMILIES or size < 16 or size % 4:
        raise ValueError("Choose a supported family and size divisible by four, at least 16")
    if not 0 <= perturbation <= 1:
        raise ValueError("Perturbation must lie in [0,1]")
    rng = np.random.default_rng(seed)
    perturb_rng = np.random.default_rng(np.random.SeedSequence([seed, 4444]))
    mechanism_rng = np.random.default_rng(np.random.SeedSequence([seed, 1234]))
    terminal_budget = int(mechanism_rng.choice([12, 16, 20, 24]))
    switching_mass = int(mechanism_rng.choice([4, 8, 12, 16]))
    progress_mass = int(mechanism_rng.choice([8, 12, 16]))
    evidence_reliability_tenths = int(mechanism_rng.choice([6, 7, 8]))
    trust_up_quarters = int(mechanism_rng.choice([1, 2, 3]))
    parameters = (
        ("terminal_budget", terminal_budget),
        ("switching_mass", switching_mass),
        ("progress_mass", progress_mass),
        ("evidence_reliability_tenths", evidence_reliability_tenths),
        ("trust_up_quarters", trust_up_quarters),
    )
    base_size, copies = size // 4, 4
    active = base_size - 2
    base = np.zeros((base_size, base_size))
    features = tuple((s % 2, s // 2) for s in range(base_size))
    for s in range(active):
        # Integer masses sum to 64. Quantized Bayesian updates and all other
        # mechanism transitions therefore retain exact binary-rational weights.
        mode, level = features[s]
        active_budget = 64 - terminal_budget
        position = Fraction(s, active - 1)
        if family == "preference":
            targets = [s, (s ^ 1) % active, (s + 2) % active]
            complete = terminal_budget * (1 + 2 * mode) // 4
            weights = [
                active_budget - switching_mass - progress_mass,
                switching_mass,
                progress_mass,
                complete,
                terminal_budget - complete,
            ]
        elif family == "belief":
            reliability = Fraction(evidence_reliability_tenths, 10)
            positive_cue = reliability * position + (1 - reliability) * (1 - position)
            posterior_positive = reliability * position / positive_cue
            posterior_negative = (1 - reliability) * position / (1 - positive_cue)
            next_positive = min(active - 1, int(posterior_positive * (active - 1) + Fraction(1, 2)))
            next_negative = min(active - 1, int(posterior_negative * (active - 1) + Fraction(1, 2)))
            positive_mass = int(active_budget * positive_cue + Fraction(1, 2))
            complete = int(terminal_budget * position + Fraction(1, 2))
            targets = [next_negative, next_positive]
            weights = [
                active_budget - positive_mass,
                positive_mass,
                complete,
                terminal_budget - complete,
            ]
        elif family == "trust":
            stay = active_budget // 2
            movement = active_budget - stay
            upward = movement * trust_up_quarters // 4
            complete = 2 + int((terminal_budget - 4) * position)
            targets = [s, max(0, s - 1), min(active - 1, s + 1)]
            weights = [stay, movement - upward, upward, complete, terminal_budget - complete]
        elif family == "patience":
            stop = 2 + int((terminal_budget - 2) * position)
            targets = [s, min(active - 1, s + 1), 0]
            weights = [8, active_budget - 12, 4, terminal_budget - stop, stop]
        else:
            progress = Fraction(level, max(1, (active - 1) // 2))
            complete = int(terminal_budget * (Fraction(1, 5) + Fraction(mode, 2) + progress / 5))
            stay = active_budget // 3
            targets = [s, (s + 2) % active, (s ^ 1) % active]
            weights = [
                stay,
                active_budget - stay - switching_mass,
                switching_mass,
                complete,
                terminal_budget - complete,
            ]
        destinations = targets + [base_size - 2, base_size - 1]
        for destination, mass in zip(destinations, weights):
            base[s, destination] += mass / 64
    base[-2, -2] = base[-1, -1] = 1
    p = np.zeros((size, size))
    blocks = tuple(tuple(range(b * copies, (b + 1) * copies)) for b in range(base_size))
    for b in range(base_size):
        for state in blocks[b]:
            for destination in np.flatnonzero(base[b]):
                # Arbitrary internal redistribution preserves class probabilities.
                allocation = rng.multinomial(16, np.full(copies, 1 / copies)) / 16
                p[state, list(blocks[destination])] += base[b, destination] * allocation
            if perturbation and b < active:
                p[state] *= 1 - perturbation
                p[state, perturb_rng.integers(size)] += perturbation
    labels = tuple(
        "complete" if b == base_size - 2 else "disengage" if b == base_size - 1 else "active"
        for b in range(base_size)
        for _ in range(copies)
    )
    initial = np.zeros(size)
    initial[:copies] = 1 / copies
    return World(
        Chain(p, labels, initial), blocks, family, seed, perturbation, features, parameters
    )


def intervene(world: World, strength: float) -> Chain:
    """Expose hidden duplicate identity through opposing terminal responses.

    The environment is fixed for a trajectory. Strength zero recovers training.
    """
    if not 0 <= strength <= 1:
        raise ValueError("Intervention strength must lie in [0,1]")
    p = world.chain.transitions.copy()
    complete = next(i for i, label in enumerate(world.chain.labels) if label == "complete")
    disengage = next(i for i, label in enumerate(world.chain.labels) if label == "disengage")
    for s, label in enumerate(world.chain.labels):
        if label == "active":
            p[s] *= 1 - strength
            p[s, complete if s % 4 < 2 else disengage] += strength
    return Chain(p, world.chain.labels, world.chain.initial)
