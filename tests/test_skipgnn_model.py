from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens.graph.skip import build_skipgnn_adjacencies  # noqa: E402
from pathlens.model.skipgnn import (  # noqa: E402
    SkipGNN,
    scipy_to_torch_sparse,
    score_all_pairs,
)


def _tiny_graphs(device: str = "cpu"):
    bipartite = np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    original, skip = build_skipgnn_adjacencies(2, 2, bipartite)
    features = torch.eye(4, dtype=torch.float32, device=device)
    return (
        features,
        scipy_to_torch_sparse(original, device),
        scipy_to_torch_sparse(skip, device),
    )


def test_graph_convolution_on_identity_is_normalized_adj_times_weight() -> None:
    from pathlens.model.layers import GraphConvolution

    adj = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    layer = GraphConvolution(2, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        layer.bias.copy_(torch.zeros(3))
        output = layer(torch.eye(2), adj)
    expected = adj @ layer.weight
    assert torch.allclose(output, expected)


def test_forward_matches_legacy1_layer_algebra() -> None:
    features, o_adj, s_adj = _tiny_graphs()
    model = SkipGNN(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    idx = (torch.tensor([0, 1]), torch.tensor([2, 3]))
    logits, embeddings = model(features, o_adj, s_adj, idx)

    o_x = torch.relu(model.o_gc1(features, o_adj) + model.s_gc1_o(features, s_adj))
    s_x = torch.relu(model.s_gc1(features, s_adj) + model.o_gc1_s(o_x, o_adj))
    hidden = model.o_gc2(o_x, o_adj) + model.s_gc2_o(s_x, s_adj)
    feat = torch.cat((hidden[idx[0]], hidden[idx[1]]), dim=1)
    expected = model.decoder2(model.decoder1(feat))
    assert torch.allclose(logits, expected)
    assert torch.allclose(embeddings, hidden)
    assert logits.shape == (2, 1)


def test_decoder_concat_order_is_left_then_right() -> None:
    features, o_adj, s_adj = _tiny_graphs()
    model = SkipGNN(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    left_right = model(features, o_adj, s_adj, (torch.tensor([0]), torch.tensor([3])))[0]
    right_left = model(features, o_adj, s_adj, (torch.tensor([3]), torch.tensor([0])))[0]
    assert not torch.allclose(left_right, right_left)


def test_score_matrix_matches_batched_pair_forward() -> None:
    features, o_adj, s_adj = _tiny_graphs()
    model = SkipGNN(nfeat=4, nhid1=3, nhid2=2, nhid_decode1=4, dropout=0.0)
    model.eval()
    matrix = score_all_pairs(model, features, o_adj, s_adj, num_drugs=2, num_proteins=2)
    drugs = torch.tensor([0, 0, 1, 1])
    proteins = torch.tensor([2, 3, 2, 3])
    logits, _ = model(features, o_adj, s_adj, (drugs, proteins))
    assert torch.allclose(matrix.reshape(-1), logits.reshape(-1))
