# Cross-experiment evidence registry

`reports/evidence-registry.json` inventories retained aggregate evidence using the versioned schema in `spec/schemas/EvidenceRegistry.schema.json`. Every entry declares its stage, scientific domain, scope, byte count, SHA256, repository-relative artifact path, and any indexed parent identities. Stages are supplied explicitly. Shared payload fields such as `models` never determine whether an artifact represents quantization or calibrated transition estimation.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python tools/check_evidence.py
```

The checker rejects duplicate identifiers, absent or cyclic indexed parents, stage/domain conflicts, unsafe paths, symbolic links escaping the repository, and changed sizes or hashes. It reports `artifact_identity_only`. It does not execute a model checker, validate a theorem, establish statistical coverage, or declare any deliverable complete. Existing campaign-specific numerical verifiers remain necessary.

This schema wraps frozen reports without rewriting their historical format or source identity. The registry is a checkpoint: when a live training completion report changes, archive its earlier bytes or deliberately create a refreshed registry after inspection. A failed hash check must not silently accept new bytes. Parent links are included only when the referenced digest matches an indexed artifact; an absent link is not evidence that a complete scientific lineage has been verified.

Remaining integration work includes certificate-family adapters, all campaign inputs and outputs, durable end-to-end execution and billing reconciliation, and final benchmark aggregation. This inventory is one component of that work.
