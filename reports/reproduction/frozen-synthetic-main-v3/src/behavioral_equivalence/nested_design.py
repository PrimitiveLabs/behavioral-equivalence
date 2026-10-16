"""Fixed nested-abstraction design and capacity-matched parent-state controls."""

import numpy as np
from .core import Chain, Partition
from .clustering import feature_partition, random_partition
from .decision_model import decision_horizon
from .property_bounds import LabelEvent


def latent_partition(
    features: np.ndarray, count: int, seed: int, random: bool = False
) -> Partition:
    """Cluster parent indices, never refit the underlying human representation."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite nonempty parent feature matrix required")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(x):
        raise ValueError("Positive integer count within parent capacity required")
    carrier = Chain(np.eye(len(x)), ("latent",) * len(x), np.full(len(x), 1 / len(x)))
    return (
        random_partition(carrier, count, seed)
        if random
        else feature_partition(carrier, x, count, seed)
    )


def human_event_suite(horizons: tuple[int, ...]) -> tuple[LabelEvent, ...]:
    """A fixed eventual-feedback suite with pending-state prediction origins."""
    if not horizons or len(set(horizons)) != len(horizons) or any(h < 1 for h in horizons):
        raise ValueError("Distinct positive decision horizons required")
    return tuple(
        LabelEvent(mode, (target,), decision_horizon(h), threshold)
        for mode, target, threshold in [
            ("occupancy", "positive", 1),
            ("reachability", "positive", 1),
            ("reachability", "negative", 1),
            ("count_at_least", "positive", 2),
            ("count_at_least", "positive", 4),
        ]
        for h in horizons
    )


def nested_run_identity(source_digest: str, parent_root: str, parent_sha256: str) -> str:
    """Bind cache reuse to both immutable upstream artifact and implementation."""
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps([source_digest, parent_root, parent_sha256], separators=(",", ":")).encode()
    ).hexdigest()
