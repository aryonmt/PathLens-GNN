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
