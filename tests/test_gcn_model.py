from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens.graph.skip import build_skipgnn_adjacencies  # noqa: E402
from pathlens.model.decode import score_from_embeddings  # noqa: E402
from pathlens.model.gcn import GCN  # noqa: E402
from pathlens.model.skipgnn import scipy_to_torch_sparse  # noqa: E402


def _tiny_original(device: str = "cpu"):
    bipartite = np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    original, skip = build_skipgnn_adjacencies(2, 2, bipartite)
    features = torch.eye(4, dtype=torch.float32, device=device)
    return (
        features,
        scipy_to_torch_sparse(original, device),
        scipy_to_torch_sparse(skip, device),
    )


def test_gcn_forward_is_two_layer_original_graph_only() -> None:
    features, adj, skip = _tiny_original()
    model = GCN(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    idx = (torch.tensor([0, 1]), torch.tensor([2, 3]))
    logits, embeddings = model(features, adj, idx)

    hidden = torch.relu(model.gc1(features, adj))
    hidden = model.gc2(hidden, adj)
    feat = torch.cat((hidden[idx[0]], hidden[idx[1]]), dim=1)
    expected = model.decoder2(model.decoder1(feat))
    assert torch.allclose(logits, expected)
    assert torch.allclose(embeddings, hidden)
    skip_hidden = torch.relu(model.gc1(features, skip))
    skip_hidden = model.gc2(skip_hidden, skip)
    assert not torch.allclose(embeddings, skip_hidden)


def test_gcn_score_matrix_matches_pair_forward() -> None:
    features, adj, _skip = _tiny_original()
    model = GCN(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    embeddings = model.encode(features, adj)
    matrix = score_from_embeddings(model, embeddings, num_drugs=2, num_proteins=2)
    drugs = torch.tensor([0, 0, 1, 1])
    proteins = torch.tensor([2, 3, 2, 3])
    logits, _ = model(features, adj, (drugs, proteins))
    assert torch.allclose(matrix.reshape(-1), logits.reshape(-1))
