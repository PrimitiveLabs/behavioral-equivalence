"""Guard frozen single-call Modal controllers without changing their source."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_final_human_jobs import launcher_lock


def observe_call(call_id, wait_seconds):
    import modal

    return modal.FunctionCall.from_id(call_id).get(timeout=wait_seconds)


def _handle(path):
    try:
        value = json.loads(path.read_text())["function_call_id"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("Original provider state has no usable call handle") from None
    if not isinstance(value, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", value):
        raise ValueError("Original provider state has no usable call handle")
    return value


def native_state(root, controller, arguments):
    """Derive known frozen protocol-controller state paths before submission."""
    flags = {
        "synthetic_campaign.py": "--protocol-path",
        "uncertainty_campaign.py": "--config",
        "intervention_campaign.py": "--protocol-path",
    }
    flag = flags.get(controller.name)
    if flag is None:
        return None
    values = []
    for i, argument in enumerate(arguments):
        if argument == flag:
            if i + 1 >= len(arguments):
                raise ValueError("A single frozen protocol path is required")
            values.append(arguments[i + 1])
        elif argument.startswith(flag + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError("A single frozen protocol path is required")
    protocol = json.loads((root / values[0]).read_text())
    if not isinstance(protocol, dict):
        raise ValueError("Frozen protocol must be a JSON object")
    campaign = protocol.get("campaign_id")
    if not isinstance(campaign, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", campaign):
        raise ValueError("Invalid frozen campaign identity")
    if controller.name == "intervention_campaign.py":
        if "--smoke" in arguments and "--no-smoke" in arguments:
            raise ValueError("Choose a single smoke mode")
        campaign += "-full" if "--no-smoke" in arguments else "-smoke"
    return root / ".campaign" / (campaign + ".json")


def run_guarded(
    root,
    controller,
    state_file=None,
    arguments=(),
    *,
    modal_command=None,
    observer=observe_call,
    wait_seconds=0,
):
    """Invoke once or observe the saved original call; never replace it.

    Receipts describe client observations, not scientific acceptance. This is
    local filesystem coordination for invocations through this entrypoint,
    not distributed exactly-once execution or provider-enforced billing caps.
    """
    root = Path(root).resolve()
    controller = (root / controller).resolve()
    campaign = root / ".campaign"
    if (
        not controller.is_file()
        or controller.suffix != ".py"
        or root / "cloud" not in controller.parents
    ):
        raise ValueError("Controller must be a Python file inside this project's cloud directory")
    expected = native_state(root, controller, arguments)
    if state_file is None and expected is None:
        raise ValueError("Explicit native state is required for this controller")
    state = (root / state_file).resolve() if state_file is not None else expected.resolve()
    if expected is not None and state != expected.resolve():
        raise ValueError("Native state differs from the frozen protocol identity")
    if campaign.is_symlink() or campaign not in state.parents or state.suffix != ".json":
        raise ValueError("Native provider state must be inside this project's .campaign directory")
    if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 60:
        raise ValueError("Observation wait must be between zero and sixty seconds")
    modal_command = (
        list(modal_command) if modal_command is not None else [sys.executable, "-m", "modal"]
    )
    if not modal_command:
        raise ValueError("Modal command cannot be empty")
    command = modal_command + ["run", "--detach", str(controller)] + list(arguments)
    digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    identity = hashlib.sha256(str(state.relative_to(root)).encode()).hexdigest()
    directory = campaign / "guarded"
    if directory.is_symlink():
        raise ValueError("Guard storage must not be a symlink")
    receipt_path = directory / (identity + ".json")
    attempt = directory / (identity + ".attempt")
    controller_key = hashlib.sha256(str(controller.relative_to(root)).encode()).hexdigest()
    with (
        launcher_lock(directory / (controller_key + ".controller.lock")),
        launcher_lock(directory / (identity + ".state.lock")),
    ):
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        original = previous.get("function_call_id")
        if original is not None and (
            not isinstance(original, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", original)
        ):
            raise ValueError("Guard receipt has no usable original call handle")
        receipt = dict(
            native_state_file=str(state.relative_to(root)),
            controller=str(controller.relative_to(root)),
            command_digest=digest,
            controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
            function_call_id=original,
            launched_cli=False,
            scientific_acceptance="not_assessed",
        )

        def save(status, **values):
            receipt.update(
                status=status, observed_at=datetime.now(timezone.utc).isoformat(), **values
            )
            temporary = receipt_path.with_suffix(".partial")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(receipt_path)
            return dict(receipt)

        if state.exists():
            try:
                current = _handle(state)
            except ValueError:
                return save("needs_inspection", error_type="InvalidOriginalState")
            if original and current != original:
                return save("needs_inspection", error_type="OriginalHandleChanged")
            original = current
            receipt["function_call_id"] = original
        if not original:
            if attempt.exists() or previous:
                return save("needs_inspection", error_type="SubmissionOutcomeUnknown")
            with attempt.open("x") as marker:
                marker.write(digest + "\n")
            save("submission_attempted")
            try:
                result = subprocess.run(command, cwd=root, check=False)
            except (OSError, KeyboardInterrupt) as error:
                return save("needs_inspection", error_type=type(error).__name__)
            receipt.update(launched_cli=True, cli_returncode=result.returncode)
            if state.exists():
                try:
                    original = _handle(state)
                    receipt["function_call_id"] = original
                except ValueError:
                    return save("needs_inspection", error_type="InvalidOriginalState")
            if result.returncode or not original:
                return save(
                    "needs_inspection",
                    error_type="ClientExit" if result.returncode else "MissingOriginalHandle",
                )
        # Persist the original handle before any interruptible provider read.
        save("observation_requested")
        # The only repeat action is a read of the original provider call.
        try:
            result = observer(original, wait_seconds)
        except TimeoutError:
            return save("observation_pending")
        except KeyboardInterrupt:
            return save("needs_inspection", error_type="KeyboardInterrupt")
        except Exception as error:
            return save("needs_inspection", error_type=type(error).__name__)
        status = result.get("status") if isinstance(result, dict) else None
        return save(
            "provider_result_observed",
            provider_result_status=status if isinstance(status, str) else "available",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("controller", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_guarded(
        root, args.controller, args.state_file, args.arguments, wait_seconds=args.wait_seconds
    )
    print(json.dumps(result, indent=2))
    if result["status"] == "needs_inspection":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
