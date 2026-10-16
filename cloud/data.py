"""Cloud-only KuaiRand pilot ingestion with pinned archive checksum."""

from pathlib import Path
import modal

app = modal.App("behavioral-equivalence-data-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image, cpu=1, memory=2048, timeout=1200, max_containers=1, volumes={"/artifacts": volume}
)
def inspect_kuairand() -> dict:
    import csv
    import hashlib
    import io
    import json
    import tarfile
    import time
    import urllib.request

    destination = Path("/artifacts/datasets/kuairand-1k")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "source.tar.gz"
    expected = "6b0b9c8222d67fcd4c676218edca3f1f"
    url = "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
    start = time.monotonic()
    md5, sha = hashlib.md5(), hashlib.sha256()
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
        if md5.hexdigest() != expected:
            raise ValueError("Official archive checksum mismatch")
        temporary.rename(archive)
    else:
        with archive.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                md5.update(chunk)
                sha.update(chunk)
        if md5.hexdigest() != expected:
            raise ValueError("Stored archive checksum mismatch")
    entries = []
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isfile():
                entry = {"name": member.name, "size": member.size}
                if member.name.endswith(".csv"):
                    raw = source.extractfile(member)
                    with io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
                        reader = csv.reader(stream)
                        entry["columns"] = next(reader)
                        # Persist schema only, not user rows in local reports.
                entries.append(entry)
    result = {
        "dataset": "KuaiRand-1K",
        "source": url,
        "archive_md5": md5.hexdigest(),
        "archive_sha256": sha.hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "entries": entries,
        "wall_seconds": time.monotonic() - start,
        "license_source": "https://zenodo.org/api/records/10439422",
        "archive_license": "CC-BY-4.0",
        "purpose": "pilot only",
    }
    (destination / "inspection.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result
