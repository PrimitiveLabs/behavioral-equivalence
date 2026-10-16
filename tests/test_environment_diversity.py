import numpy as np
from behavioral_equivalence.environment_diversity import diversity_catalogues, run_diversity_world


def test_catalogue_spans_hold_query_capacity_and_evaluation_points_fixed():
    catalogues, heldout = diversity_catalogues()
    assert [len(x) for x in catalogues.values()] == [8, 16, 64]
    assert len(heldout) == 16
    assert all(len(np.unique(x, axis=0)) == len(x) for x in catalogues.values())
    assert np.max(np.abs(catalogues["narrow"][:, 0])) < np.max(np.abs(catalogues["broad"][:, 0]))


def test_response_invariant_world_has_no_diversity_effect_and_replayed_certificates():
    report, arrays = run_diversity_world("trust", 32, 40000, "negative_control")
    assert len(report["records"]) == 240
    assert len(report["certificates"]) == 48
    assert all(s["query_count"] == 8 for s in report["searches"])
    assert max(r["total_prediction_max_error"] for r in report["records"]) < 1e-12
    assert len(arrays) == 96
