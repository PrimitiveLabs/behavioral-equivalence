"""Cloud ingestion helpers and pilot fingerprint body on fabricated local files."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_cloud_summary_boundaries import body

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "controller,function", [("audit_27k.py", "file_digest"), ("item_metadata_audit.py", "sha256")]
)
@pytest.mark.parametrize("payload", [b"", b"abc", b"x" * (8 * 1024**2 + 1)])
def test_cloud_file_hashes_stream_exact_bytes(tmp_path, controller, function, payload):
    p = tmp_path / "artifact"
    p.write_bytes(payload)
    assert body(controller, function)(p) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileNotFoundError):
        body(controller, function)(tmp_path / "missing")


@pytest.mark.parametrize("bucket", [-1, 32])
def test_full_data_audit_rejects_outside_bucket_before_filesystem(bucket):
    with pytest.raises(ValueError, match="Invalid data bucket"):
        body("audit_27k.py", "audit_one")(bucket)


def pilot_body(tmp_path, calls):
    p = ROOT / "cloud/pilot_identity.py"
    function = next(
        n
        for n in ast.parse(p.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "fingerprint"
    )
    function.decorator_list = []

    class BindRoot(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "/artifacts/datasets/kuairand-1k/prepared-v4":
                return ast.copy_location(ast.Constant(str(tmp_path)), node)
            return node

    function = BindRoot().visit(function)
    scope = {"volume": SimpleNamespace(commit=lambda: calls.append("commit"))}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(p), "exec"
        ),
        scope,
    )
    return scope["fingerprint"]


def test_pilot_identity_hashes_sorted_raw_duplicates_without_outcomes(tmp_path):
    pl = pytest.importorskip("polars")
    import json

    calls = []
    records = [
        dict(user_id=[1, 2], time_ms=[20, 10], tab=[1, 2], is_click=[0, 1]),
        dict(user_id=[1, 2], time_ms=[10, 20], tab=[2, 1], is_click=[1, 0]),
        dict(user_id=[1, 2], time_ms=[10, 30], tab=[2, 1], is_click=[0, 0]),
    ]
    for i, r in enumerate(records):
        pl.DataFrame(r).write_parquet(tmp_path / f"log_{i}.parquet")
    run = pilot_body(tmp_path, calls)
    result = run()
    raw = (tmp_path / "pilot-user-fingerprints.json").read_bytes()
    artifact = json.loads(raw)
    expected = np.array([[10, 2], [10, 2], [20, 1]], dtype="<i8")
    assert artifact["records"][0]["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert artifact["records"][0]["rows"] == 3 and result["raw_rows"] == 6
    assert hashlib.sha256(raw).hexdigest() == result["artifact_sha256"]
    for i, r in enumerate(records):
        r["is_click"] = [1 - v for v in r["is_click"]]
        pl.DataFrame(r).write_parquet(tmp_path / f"log_{i}.parquet")
    run()
    assert (tmp_path / "pilot-user-fingerprints.json").read_bytes() == raw
    assert calls == ["commit", "commit"]


def test_pilot_identity_requires_all_three_unfiltered_logs(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="all three"):
        pilot_body(tmp_path, calls)()
    assert calls == []
