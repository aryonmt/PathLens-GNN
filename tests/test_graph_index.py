from __future__ import annotations

import numpy as np

from pathlens_gnn.graph.index import BipartiteIndex


def test_bridge_paths_and_projection_context_are_distinct() -> None:
    # d0-p0-d1-p1 is a three-hop bridge; d0 and d1 share p0 as two-hop context.
    edges = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)
    index = BipartiteIndex(3, 2, edges)
    paths = index.bridge_paths(0, 1)
    assert [(path.intermediate_protein, path.intermediate_drug) for path in paths] == [(0, 1)]
    assert index.top_similar_drugs(0)[0][0] == 1
    assert index.structural_features(np.asarray([[0, 1]], dtype=np.int64)).shape == (1, 2)


def test_pair_feature_tables_match_bridge_and_projection_mass() -> None:
    edges = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)
    index = BipartiteIndex(3, 2, edges)
    drug_mass, protein_mass, bridge = index.pair_feature_tables()
    pairs = np.asarray([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    looked_up = np.column_stack(
        (
            drug_mass[pairs[:, 0]] + protein_mass[pairs[:, 1]],
            bridge[pairs[:, 0], pairs[:, 1]],
        )
    )
    walked = np.asarray(
        [
            [
                index.projection_mass_drug(int(drug)) + index.projection_mass_protein(int(protein)),
                float(index.bridge_scores_for_drug(int(drug))[int(protein)]),
            ]
            for drug, protein in pairs
        ],
        dtype=np.float32,
    )
    assert looked_up.shape == (3, 2)
    assert np.allclose(looked_up, walked)
    assert np.allclose(index.structural_features(pairs), walked)
    for drug, protein in pairs:
        path_weight = sum(
            path.weight for path in index.bridge_paths(int(drug), int(protein), limit=None)
        )
        assert np.isclose(bridge[int(drug), int(protein)], path_weight)
