from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens.graph.skip import build_symmetric_adjacency  # noqa: E402
from pathlens.model.decode import score_from_embeddings  # noqa: E402
from pathlens.model.graphsage import GraphSAGE  # noqa: E402
from pathlens.model.skipgnn import scipy_to_torch_sparse  # noqa: E402


def test_sage_layer_is_self_plus_mean_neighbors() -> None:
    from pathlens.model.graphsage import SAGEConv

    adj = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    features = torch.eye(3, dtype=torch.float32)
    layer = SAGEConv(3, 2)
    with torch.no_grad():
        layer.weight_self.copy_(torch.ones(3, 2))
        layer.weight_neigh.copy_(torch.full((3, 2), 2.0))
        layer.bias.copy_(torch.zeros(2))
        output = layer(features, adj)
    expected = features @ layer.weight_self
    mean_neighbors = (adj @ features) / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    expected = expected + mean_neighbors @ layer.weight_neigh
    assert torch.allclose(output, expected)


def test_graphsage_score_matrix_matches_pair_forward() -> None:
    bipartite = np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    adj = scipy_to_torch_sparse(build_symmetric_adjacency(2, 2, bipartite), "cpu")
    features = torch.eye(4, dtype=torch.float32)
    model = GraphSAGE(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    embeddings = model.encode(features, adj)
    matrix = score_from_embeddings(model, embeddings, num_drugs=2, num_proteins=2)
    drugs = torch.tensor([0, 0, 1, 1])
    proteins = torch.tensor([2, 3, 2, 3])
    logits, _ = model(features, adj, (drugs, proteins))
    assert torch.allclose(matrix.reshape(-1), logits.reshape(-1))
