"""Download and inspect the final candidate archive entirely in cloud storage."""

import modal

app = modal.App("behavioral-equivalence-kuairand27k-ingest-v2")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def ingest() -> dict:
    from pathlib import Path
    import csv
    import hashlib
    import json
    import tarfile
    import time
    import urllib.request

    root = Path("/artifacts/datasets/kuairand-27k")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.tar.gz"
    temporary = root / "source.partial"
    url = "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz"
    expected_md5 = "3e3c799a24e2d23a4d2c757fbf9adf59"
    expected_bytes = 9892191178
    start = time.monotonic()
    if not archive.exists():
        offset = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            if offset and response.status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Range response does not match partial archive")
                mode = "ab"
            else:
                offset = 0
                mode = "wb"
            with temporary.open(mode) as destination:
                last_gib = offset // (1024**3)
                while chunk := response.read(8 * 1024 * 1024):
                    destination.write(chunk)
                    offset += len(chunk)
                    if offset // (1024**3) > last_gib:
                        last_gib = offset // (1024**3)
                        print(f"27K archive download: {offset}/{expected_bytes} bytes")
        if temporary.stat().st_size != expected_bytes:
            raise ValueError("Incomplete archive length")
    candidate = archive if archive.exists() else temporary
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with candidate.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    if md5.hexdigest() != expected_md5 or candidate.stat().st_size != expected_bytes:
        raise ValueError("Official archive checksum or size mismatch")
    if not archive.exists():
        temporary.rename(archive)
    (root / "download-manifest.json").write_text(
        json.dumps(
            {
                "status": "verified_download",
                "source": url,
                "bytes": expected_bytes,
                "md5": md5.hexdigest(),
                "sha256": sha.hexdigest(),
            },
            indent=2,
        )
    )
    volume.commit()
    entries = []
    license_id = None
    license_sha = None
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if not member.isfile():
                continue
            entry = {"name": member.name, "bytes": member.size}
            if Path(member.name).name.lower() in {
                "license",
                "license.txt",
                "license.md",
                "licence",
                "licence.txt",
            }:
                text = source.extractfile(member).read()
                (root / "source-LICENSE.txt").write_bytes(text)
                license_sha = hashlib.sha256(text).hexdigest()
                if b"Attribution-ShareAlike 4.0 International" in text:
                    license_id = "CC-BY-SA-4.0"
                elif b"Attribution 4.0 International" in text:
                    license_id = "CC-BY-4.0"
                else:
                    raise ValueError("Embedded data license requires review")
            elif member.name.endswith("/README.md"):
                (root / "source-README.md").write_bytes(source.extractfile(member).read())
            elif member.name.endswith(".csv"):
                header = source.extractfile(member).readline().decode("utf-8-sig")
                entry["columns"] = next(csv.reader([header]))
            entries.append(entry)
    (root / "archive-entries.json").write_text(json.dumps(entries, indent=2))
    applicable_license = license_id
    license_basis = "embedded archive text"
    metadata_sha = None
    if license_id is None:
        metadata_url = "https://zenodo.org/api/records/10439422"
        metadata_bytes = urllib.request.urlopen(metadata_url, timeout=60).read()
        metadata = json.loads(metadata_bytes)
        deposit_file = next(f for f in metadata["files"] if f["key"] == "KuaiRand-27K.tar.gz")
        if (
            metadata["metadata"]["license"]["id"] != "cc-by-4.0"
            or deposit_file["checksum"] != "md5:" + expected_md5
            or deposit_file["size"] != expected_bytes
        ):
            raise ValueError("Deposit rights evidence does not match this exact archive")
        (root / "zenodo-rights-evidence.json").write_bytes(metadata_bytes)
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
        applicable_license = "CC-BY-4.0"
        license_basis = (
            metadata_url
            + " matched by exact file name, checksum and size; no embedded license found"
        )
        notice = "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. Gao et al. (2022), DOI 10.1145/3511808.3557624.\nSource: https://zenodo.org/records/10439422\nDeposit metadata licenses this exact archive under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .\nNo license text is embedded in this archive. This notice records external deposit evidence; it is not an extracted archive license.\nKeep attribution and indicate preprocessing changes. Private research only in this program; no raw redistribution.\n"
        (root / "source-LICENSE.txt").write_text(notice)
    report = {
        "status": "complete",
        "dataset": "KuaiRand-27K",
        "phase": "ingestion and rights/schema inspection",
        "source": url,
        "archive_bytes": expected_bytes,
        "archive_md5": md5.hexdigest(),
        "archive_sha256": sha.hexdigest(),
        "embedded_license": license_id,
        "applicable_license": applicable_license,
        "license_basis": license_basis,
        "deposit_metadata_sha256": metadata_sha,
        "embedded_license_sha256": license_sha,
        "entries": entries,
        "wall_seconds": time.monotonic() - start,
        "confirmatory_test_status": "not evaluated; no outcome performance inspected",
        "restrictions": [
            "private research use under attribution/share-alike obligations",
            "no product weight integration or raw redistribution",
            "pilot-user mapping and frozen evaluation still required",
        ],
    }
    (root / "inspection.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "entries"}
