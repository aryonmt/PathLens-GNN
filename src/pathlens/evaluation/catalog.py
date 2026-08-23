"""Which methods belong in which figure.

The five PathLens cards are one architecture, ablated. Nested hops
(`one_hop`, `s1_s2`, `s1_s2_s3_fixed`) stay in the family ladder. Head-to-head
scoreboard plots use every other filed method, including the two PathLens
heads (`pathlens_bce`, `pathlens_ranking`).
"""

from __future__ import annotations

from typing import Any

PATHLENS_FAMILY: tuple[str, ...] = (
    "one_hop",
    "s1_s2",
    "s1_s2_s3_fixed",
    "pathlens_bce",
    "pathlens_ranking",
)
PATHLENS_NESTED: frozenset[str] = frozenset({"one_hop", "s1_s2", "s1_s2_s3_fixed"})
SKIP_FROM_PLOTS: frozenset[str] = frozenset(
    {"ranking_diagnostics", "gat", "nbfnet", "figures", "imports"}
)
DELIVERY_METHODS: tuple[str, ...] = (
    "degree",
    "resource_allocation",
    "three_hop",
    "skipgnn",
    "gcn",
    "graphsage",
    "residual_three_hop",
    "one_hop",
    "s1_s2",
    "s1_s2_s3_fixed",
    "pathlens_bce",
    "pathlens_ranking",
    "blend_pathlens_three_hop",
)


def comparison_models(models: dict[str, Any]) -> dict[str, Any]:
    return {
        method_id: payload
        for method_id, payload in models.items()
        if method_id not in PATHLENS_NESTED and method_id not in SKIP_FROM_PLOTS
    }


def family_models(models: dict[str, Any]) -> dict[str, Any]:
    return {
        method_id: models[method_id]
        for method_id in PATHLENS_FAMILY
        if method_id in models
    }
