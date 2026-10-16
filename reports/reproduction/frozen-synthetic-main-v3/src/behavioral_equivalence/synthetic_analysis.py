"""Descriptive synthetic benchmark tables with explicit verification scopes."""

import json


def world_method_rows(report: dict, audit: dict, horizons: list[int]) -> tuple[list, list]:
    if (
        report["status"] != "complete"
        or audit["status"] != "complete"
        or report["spec"] != audit["spec"]
        or report["world_digest"] != audit["world_digest"]
        or len(report["methods"]) != len(audit["methods"])
    ):
        raise ValueError("Matching completed world and audit required")
    methods, properties = [], []
    targets = {"complete", "disengage", "survival", "complete_given_active_next"}
    for index, (method, checked) in enumerate(zip(report["methods"], audit["methods"])):
        if (
            checked["index"] != index
            or checked["method"] != method["method"]
            or checked["states"] != method["states"]
            or checked["verified"] is not True
        ):
            raise ValueError("Method differs from audited input")
        observed = {(p["target"], p["horizon"]): p for p in method["properties"]}
        if len(observed) != len(method["properties"]) or set(observed) != {
            (t, h) for t in targets for h in horizons
        }:
            raise ValueError("Complete distinct property/horizon matrix required")
        base = dict(
            report["spec"],
            method_index=index,
            method=method["method"],
            parameters=json.dumps(method["parameters"], sort_keys=True, separators=(",", ":")),
        )
        methods.append(
            dict(
                base,
                states=method["states"],
                compression_ratio=method["compression_ratio"],
                epsilon_float=method["epsilon_measured"],
                epsilon_exact_normalized_for_display=checked["epsilon_approximation_for_display"],
                exact_budget_excess_for_display=checked["exact_budget_excess_for_display"],
                budget_strictly_satisfied=checked["budget_strictly_satisfied"],
                epsilon_is_exactly_zero=checked["epsilon_is_exactly_zero"],
                partition_seconds=method["partition_seconds"],
                property_seconds=method["property_seconds"],
                prism_seconds=method["independent_verification"]["seconds"],
                process_cumulative_peak_rss_kib=method["process_cumulative_peak_rss_kib"],
                transition_edges=method["transition_edges"],
            )
        )
        for h in horizons:
            unconditional = [observed[t, h] for t in ["complete", "disengage", "survival"]]
            if len({p["bound"] for p in unconditional}) != 1:
                raise ValueError("Unconditional label events must share the coupling bound")
            conditional = observed["complete_given_active_next", h]
            properties.append(
                dict(
                    base,
                    horizon=h,
                    maximum_unconditional_floating_error=max(p["max_error"] for p in unconditional),
                    floating_coupling_bound=unconditional[0]["bound"],
                    unconditional_violations=sum(p["violated"] for p in unconditional),
                    conditional_floating_error=conditional["max_error"],
                    conditional_bound=conditional["bound"],
                    conditional_violation=conditional["violated"],
                )
            )
    return methods, properties
