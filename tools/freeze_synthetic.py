"""Seal a synthetic protocol against its source without opening benchmark results."""

from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seal(path: Path) -> dict:
    from cloud.synthetic_campaign import source_digest, validate_protocol

    protocol = json.loads(path.read_text())
    protocol["source_digest"] = source_digest(ROOT / "src/behavioral_equivalence", protocol)
    protocol["freeze_at"] = datetime.now(timezone.utc).isoformat()
    semantic = {k: v for k, v in protocol.items() if k not in ("protocol_digest", "campaign_id")}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    protocol["protocol_digest"] = digest
    protocol["campaign_id"] = "synthetic-main-v3-" + digest[:12]
    validate_protocol(protocol)
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        result = seal(Path(arg))
        print(result["campaign_id"])
