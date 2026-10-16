"""Real subprocess/filesystem tests; provider observations are simulated."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

TOOL = Path(__file__).parents[1] / "tools/guarded_modal.py"
spec = importlib.util.spec_from_file_location("guarded_modal", TOOL)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def workspace(tmp_path):
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    controller = cloud / "campaign.py"
    controller.write_text("# immutable controller fixture\n")
    state = tmp_path / ".campaign/provider.json"
    mode = tmp_path / "mode"
    mode.write_text("success")
    fake = tmp_path / "fake_modal.py"
    fake.write_text(f"""
from pathlib import Path
import json,sys,time
root=Path({str(tmp_path)!r}); state=root/'.campaign/provider.json'
with (root/'invocations').open('a') as f:f.write('invoked\\n')
(root/'argv.json').write_text(json.dumps(sys.argv[1:]))
mode=(root/'mode').read_text()
if mode=='missing':sys.exit(0)
if mode=='block':
    (root/'ready').touch();deadline=time.monotonic()+8
    while not (root/'release').exists() and time.monotonic()<deadline:time.sleep(.01)
state.parent.mkdir(exist_ok=True)
state.write_text('{{' if mode=='malformed' else json.dumps({{'function_call_id':'fc-original'}}))
if mode in ['failure','malformed']:sys.exit(3)
""")
    observations = []

    def observe(identity, wait):
        observations.append((identity, wait))
        return {"status": "complete", "private_result": "do-not-persist"}

    def run(**kwargs):
        args = dict(
            root=tmp_path,
            controller="cloud/campaign.py",
            state_file=".campaign/provider.json",
            modal_command=[sys.executable, str(fake)],
            observer=observe,
        )
        args.update(kwargs)
        return m.run_guarded(**args)

    return SimpleNamespace(
        root=tmp_path,
        state=state,
        mode=mode,
        fake=fake,
        controller=controller,
        run=run,
        observations=observations,
    )


def invocations(w):
    p = w.root / "invocations"
    return len(p.read_text().splitlines()) if p.exists() else 0


def test_submit_once_then_only_observe_original_even_after_native_state_is_removed(workspace):
    w = workspace
    before = w.controller.read_bytes()
    first = w.run()
    assert first["launched_cli"]
    w.state.unlink()
    second = w.run()
    assert invocations(w) == 1 and not second["launched_cli"]
    assert w.observations == [("fc-original", 0), ("fc-original", 0)]
    assert second["scientific_acceptance"] == "not_assessed"
    assert "private_result" not in json.dumps(second) and w.controller.read_bytes() == before


@pytest.mark.parametrize("mode", ["missing", "malformed", "failure"])
def test_failed_client_or_ambiguous_submission_never_starts_replacement(workspace, mode):
    w = workspace
    w.mode.write_text(mode)
    first = w.run()
    assert first["status"] == "needs_inspection"
    second = w.run()
    assert invocations(w) == 1
    if mode == "failure":
        assert (
            second["status"] == "provider_result_observed"
            and second["function_call_id"] == "fc-original"
        )
    else:
        assert second["status"] == "needs_inspection"


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("observation only"),
        RuntimeError("do-not-store-error-payload"),
        KeyboardInterrupt(),
    ],
)
def test_provider_observation_errors_keep_handle_and_release_locks(workspace, error):
    w = workspace

    def fail(*args):
        raise error

    first = w.run(observer=fail)
    assert first["function_call_id"] == "fc-original"
    assert first["status"] == (
        "observation_pending" if isinstance(error, TimeoutError) else "needs_inspection"
    )
    assert "do-not-store" not in json.dumps(first)
    w.state.unlink()
    second = w.run()
    assert second["function_call_id"] == "fc-original" and invocations(w) == 1


def test_changed_native_handle_is_rejected_without_losing_original(workspace):
    w = workspace
    w.run()
    w.state.write_text(json.dumps({"function_call_id": "fc-other"}))
    result = w.run()
    assert (
        result["error_type"] == "OriginalHandleChanged"
        and result["function_call_id"] == "fc-original"
    )
    assert invocations(w) == 1 and len(w.observations) == 1


@pytest.mark.parametrize(
    "content", ["{", "[]", "{}", '{"function_call_id":7}', '{"function_call_id":"bad"}']
)
def test_invalid_existing_native_state_never_executes_cli(workspace, content):
    w = workspace
    w.state.parent.mkdir()
    w.state.write_text(content)
    assert w.run()["status"] == "needs_inspection" and invocations(w) == 0


def test_missing_executable_keeps_ambiguous_marker_and_never_retries(workspace):
    w = workspace
    first = w.run(modal_command=[str(w.root / "nonexistent")])
    assert first["error_type"] == "FileNotFoundError"
    assert w.run()["error_type"] == "SubmissionOutcomeUnknown" and invocations(w) == 0


def test_process_interrupt_during_submission_is_retained(workspace, monkeypatch):
    w = workspace

    def fail(*args, **kwargs):
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(m.subprocess, "run", fail)
        assert w.run()["error_type"] == "KeyboardInterrupt"
    assert w.run()["error_type"] == "SubmissionOutcomeUnknown" and invocations(w) == 0


def test_argument_values_are_passed_literally_without_a_shell(workspace):
    w = workspace
    sentinel = w.root / "shell-would-create"
    value = "$(touch " + str(sentinel) + ")"
    w.run(arguments=["--label", value, "two words"])
    assert json.loads((w.root / "argv.json").read_text())[-3:] == ["--label", value, "two words"]
    assert not sentinel.exists()


@pytest.mark.parametrize("wait", [-1, 61, float("nan"), float("inf")])
def test_invalid_wait_never_launches(workspace, wait):
    w = workspace
    with pytest.raises(ValueError):
        w.run(wait_seconds=wait)
    assert invocations(w) == 0


def test_controller_state_and_storage_boundaries(workspace):
    w = workspace
    for kwargs in [
        dict(controller="missing.py"),
        dict(controller="fake_modal.py"),
        dict(state_file="outside.json"),
        dict(state_file=".campaign/../outside.json"),
        dict(state_file=None),
        dict(modal_command=[]),
    ]:
        with pytest.raises(ValueError):
            w.run(**kwargs)
    w.state.parent.mkdir()
    (w.state.parent / "guarded").symlink_to(w.root / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError):
        w.run()
    assert invocations(w) == 0


def test_corrupt_guard_receipt_is_not_a_license_to_submit_again(workspace):
    w = workspace
    w.run()
    p = next((w.root / ".campaign/guarded").glob("*.json"))
    r = json.loads(p.read_text())
    r["function_call_id"] = "not-a-call"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        w.run()
    assert invocations(w) == 1


@pytest.mark.parametrize(
    "name,flag,suffix",
    [
        ("synthetic_campaign.py", "--protocol-path", ""),
        ("uncertainty_campaign.py", "--config", ""),
        ("intervention_campaign.py", "--protocol-path", "-smoke"),
    ],
)
def test_native_protocol_identity_is_derived_and_wrong_explicit_state_rejected(
    workspace, name, flag, suffix
):
    w = workspace
    controller = w.root / "cloud" / name
    controller.write_text("# frozen fixture")
    config = w.root / "protocol.json"
    config.write_text(json.dumps({"campaign_id": "frozen-campaign"}))
    assert m.native_state(w.root, controller, [flag, "protocol.json"]) == w.root / ".campaign" / (
        "frozen-campaign" + suffix + ".json"
    )
    assert m.native_state(w.root, controller, [flag + "=protocol.json"]) == w.root / ".campaign" / (
        "frozen-campaign" + suffix + ".json"
    )
    with pytest.raises(ValueError, match="differs"):
        w.run(controller="cloud/" + name, arguments=[flag, "protocol.json"])
    assert invocations(w) == 0


def test_intervention_full_mode_and_invalid_protocol_arguments(workspace):
    w = workspace
    controller = w.root / "cloud/intervention_campaign.py"
    controller.write_text("# fixture")
    config = w.root / "protocol.json"
    config.write_text(json.dumps({"campaign_id": "frozen"}))
    assert (
        m.native_state(w.root, controller, ["--protocol-path", "protocol.json", "--no-smoke"]).name
        == "frozen-full.json"
    )
    for args in [
        [],
        ["--protocol-path"],
        ["--protocol-path", "protocol.json", "--protocol-path", "protocol.json"],
        ["--protocol-path", "protocol.json", "--smoke", "--no-smoke"],
    ]:
        with pytest.raises(ValueError):
            m.native_state(w.root, controller, args)
    config.write_text(json.dumps({"campaign_id": "../escape"}))
    with pytest.raises(ValueError):
        m.native_state(w.root, controller, ["--protocol-path", "protocol.json"])


def test_cross_process_launch_exclusion_and_parallel_independent_controllers(workspace):
    w = workspace
    w.mode.write_text("block")
    result = w.root / "first-result.json"
    code = f"""import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('guard', {str(TOOL)!r});m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.run_guarded(Path({str(w.root)!r}),'cloud/campaign.py','.campaign/provider.json',modal_command=[sys.executable,{str(w.fake)!r}],observer=lambda *args:{{'status':'complete'}})
Path({str(result)!r}).write_text(json.dumps(r))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 4
        while not (w.root / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (w.root / "ready").exists()
        with pytest.raises(RuntimeError):
            w.run()
        with pytest.raises(RuntimeError):
            w.run(state_file=".campaign/alias.json")
        other = w.root / "cloud/other.py"
        other.write_text("# another workstream")
        with pytest.raises(RuntimeError):
            w.run(controller="cloud/other.py")
        native = w.root / ".campaign/other.json"
        fake = w.root / "other-fake.py"
        fake.write_text(
            f'from pathlib import Path\nPath({str(native)!r}).write_text(\'{{"function_call_id":"fc-independent"}}\')\n'
        )
        parallel = w.run(
            controller="cloud/other.py",
            state_file=".campaign/other.json",
            modal_command=[sys.executable, str(fake)],
        )
        assert parallel["function_call_id"] == "fc-independent" and process.poll() is None
    finally:
        (w.root / "release").touch()
        stdout, stderr = process.communicate(timeout=6)
    assert process.returncode == 0, stderr
    assert (
        json.loads(result.read_text())["function_call_id"] == "fc-original" and invocations(w) == 1
    )


def test_existing_legacy_handle_attaches_without_any_cli_or_attempt_marker(workspace):
    w = workspace
    w.state.parent.mkdir()
    w.state.write_text(json.dumps({"function_call_id": "fc-legacy"}))
    result = w.run()
    assert result["function_call_id"] == "fc-legacy" and not result["launched_cli"]
    assert invocations(w) == 0 and not list((w.root / ".campaign/guarded").glob("*.attempt"))


def test_known_protocol_can_derive_existing_original_state_without_explicit_path(workspace):
    w = workspace
    controller = w.root / "cloud/synthetic_campaign.py"
    controller.write_text("# frozen source fixture")
    (w.root / "protocol.json").write_text(json.dumps({"campaign_id": "frozen"}))
    state = w.root / ".campaign/frozen.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"function_call_id": "fc-frozen"}))
    r = w.run(
        controller="cloud/synthetic_campaign.py",
        state_file=None,
        arguments=["--protocol-path", "protocol.json"],
    )
    assert r["function_call_id"] == "fc-frozen" and invocations(w) == 0


def test_nonmapping_provider_result_is_observed_without_persisting_its_payload(workspace):
    r = workspace.run(observer=lambda *args: ["private-payload"])
    assert r["provider_result_status"] == "available" and "private-payload" not in json.dumps(r)


def test_campaign_symlink_and_non_json_native_state_are_rejected(workspace):
    w = workspace
    with pytest.raises(ValueError):
        w.run(state_file=".campaign/state.txt")
    target = w.root / "other-metadata"
    target.mkdir()
    (w.root / ".campaign").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        w.run()
    assert invocations(w) == 0


def test_protocol_file_must_be_a_json_object(workspace):
    w = workspace
    controller = w.root / "cloud/synthetic_campaign.py"
    controller.write_text("# fixture")
    (w.root / "protocol.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        m.native_state(w.root, controller, ["--protocol-path", "protocol.json"])


def test_cli_forwards_native_arguments_and_never_claims_scientific_acceptance(monkeypatch, capsys):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "provider_result_observed", "scientific_acceptance": "not_assessed"}

    monkeypatch.setattr(m, "run_guarded", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_modal.py",
            "--state-file",
            ".campaign/a.json",
            "--wait-seconds",
            "2",
            "cloud/a.py",
            "--config",
            "study.json",
        ],
    )
    m.main()
    result = json.loads(capsys.readouterr().out)
    assert result["scientific_acceptance"] == "not_assessed"
    assert calls[0][0][1:] == (
        Path("cloud/a.py"),
        Path(".campaign/a.json"),
        ["--config", "study.json"],
    )
    assert calls[0][1]["wait_seconds"] == 2
    monkeypatch.setattr(m, "run_guarded", lambda *args, **kwargs: {"status": "needs_inspection"})
    with pytest.raises(SystemExit) as error:
        m.main()
    assert error.value.code == 1


def test_provider_adapter_reads_only_the_supplied_original_call(monkeypatch):
    calls = []

    class Call:
        def get(self, **kwargs):
            calls.append(kwargs)
            return "observed"

    def from_id(identity):
        calls.append(identity)
        return Call()

    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(FunctionCall=SimpleNamespace(from_id=from_id))
    )
    assert m.observe_call("fc-original", 2) == "observed" and calls == [
        "fc-original",
        {"timeout": 2},
    ]
