"""Inspect the full campaign without mistaking planned work for executed work."""

import argparse
import json
from pathlib import Path


def plan(config: dict) -> dict:
    tasks = {t["id"]: t for t in config["tasks"]}
    if len(tasks) != len(config["tasks"]):
        raise ValueError("Duplicate task IDs")
    visited: set[str] = set()

    def visit(task_id: str, path: tuple[str, ...]) -> None:
        if task_id not in tasks:
            raise ValueError(f"Missing dependency {task_id}")
        if task_id in path:
            raise ValueError("Dependency cycle: " + " -> ".join((*path, task_id)))
        if task_id in visited:
            return
        for dependency in tasks[task_id]["depends_on"]:
            visit(dependency, (*path, task_id))
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id, ())
    for deliverable, ids in config["deliverables"].items():
        if not ids or any(i not in tasks for i in ids):
            raise ValueError(f"Invalid deliverable mapping: {deliverable}")
    ready = [
        i
        for i, t in tasks.items()
        if t["status"] not in ("complete", "running")
        and all(tasks[d]["status"] == "complete" for d in t["depends_on"])
    ]
    return {
        "task_count": len(tasks),
        "deliverable_count": len(config["deliverables"]),
        "ready": ready,
        "complete": [i for i, t in tasks.items() if t["status"] == "complete"],
        "budget_usd": config["budget_usd"],
        "submission": "disabled in planner",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan"])
    parser.add_argument("--config", type=Path, default=Path("configs/program.json"))
    args = parser.parse_args()
    print(json.dumps(plan(json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
