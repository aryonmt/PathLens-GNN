from __future__ import annotations

from pathlens.evaluation.catalog import (
    PATHLENS_FAMILY,
    PATHLENS_NESTED,
    comparison_models,
    family_models,
)


def test_comparison_drops_nested_pathlens_and_keeps_heads() -> None:
    models = {
        "degree": {"mrr": 0.13},
        "resource_allocation": {"mrr": 0.46},
        "three_hop": {"mrr": 0.45},
        "skipgnn": {"mrr": 0.14},
        "gcn": {"mrr": 0.14},
        "graphsage": {"mrr": 0.13},
        "one_hop": {"mrr": 0.10},
        "s1_s2": {"mrr": 0.14},
        "s1_s2_s3_fixed": {"mrr": 0.14},
        "pathlens_bce": {"mrr": 0.16},
        "pathlens_ranking": {"mrr": 0.38},
        "blend_pathlens_three_hop": {"mrr": 0.45},
        "rrf_pathlens_three_hop": {"mrr": 0.39},
        "residual_three_hop": {"mrr": 0.37},
        "ranking_diagnostics": {"mrr": 0.0},
    }
    compared = comparison_models(models)
    assert "one_hop" not in compared
    assert "s1_s2" not in compared
    assert "s1_s2_s3_fixed" not in compared
    assert "ranking_diagnostics" not in compared
    assert "pathlens_bce" in compared
    assert "pathlens_ranking" in compared
    assert "resource_allocation" in compared
    assert "residual_three_hop" in compared
    assert "rrf_pathlens_three_hop" in compared
    family = family_models(models)
    assert tuple(family) == PATHLENS_FAMILY
    assert set(family) > PATHLENS_NESTED
