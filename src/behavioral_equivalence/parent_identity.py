"""Cache identities for one immutable fit and its selected quantization."""

import hashlib
import json


def parent_job_identity(
    training_root: str,
    protocol: str,
    architecture: str,
    seed: int,
    quantization_name: str,
    source_digest: str,
) -> str:
    payload = [training_root, protocol, architecture, seed, quantization_name, source_digest]
    return (
        "parent-estimation-"
        + hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    )
