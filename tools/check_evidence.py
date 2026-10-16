"""Check frozen evidence identities without launching experiments or claiming success."""

import argparse
import json
from pathlib import Path
from behavioral_equivalence.evidence_registry import EvidenceRegistry, verify_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, default=Path("reports/evidence-registry.json"))
    args = parser.parse_args()
    registry = EvidenceRegistry.model_validate_json((args.root / args.registry).read_text())
    print(json.dumps(verify_registry(registry, args.root), indent=2))


if __name__ == "__main__":
    main()
