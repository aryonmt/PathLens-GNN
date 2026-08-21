from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens_gnn.graph.index import BipartiteIndex  # noqa: E402
from pathlens_gnn.training.scoring import DevicePairFeatures  # noqa: E402


def test_device_pair_feature_lookup_matches_numpy_tables() -> None:
    edges = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)
    index = BipartiteIndex(3, 2, edges)
    pairs = np.asarray([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
    features = DevicePairFeatures.from_index(index, torch.device("cpu"))
    looked_up = features.lookup(
        torch.as_tensor(pairs[:, 0], dtype=torch.long),
        torch.as_tensor(pairs[:, 1], dtype=torch.long),
    )
    assert torch.allclose(
        looked_up,
        torch.as_tensor(index.structural_features(pairs)),
    )
