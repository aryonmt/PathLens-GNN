# Validation scoreboard (`biosnap-dti-v2`)

Nested PathLens hops (`one_hop`, `s1_s2`, `s1_s2_s3_fixed`) are in the
family table only. Comparison includes PathLens BCE and ranking heads.

## Comparison

| Method | Group | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Stage |
|---|---|---:|---:|---:|---:|---:|---|
| `resource_allocation` | heuristic | 0.845 | 0.462 | 0.672 | 0.504 | 0.547 | eval |
| `blend_pathlens_three_hop` | mix | 0.887 | 0.455 | 0.673 | 0.499 | 0.544 | eval |
| `three_hop` | heuristic | 0.847 | 0.455 | 0.673 | 0.499 | 0.544 | eval |
| `rrf_pathlens_three_hop` | mix | 0.907 | 0.393 | 0.567 | 0.428 | 0.458 | eval |
| `pathlens_ranking` | pathlens_family | 0.868 | 0.375 | 0.492 | 0.397 | 0.424 | imported |
| `residual_three_hop` | residual | 0.889 | 0.367 | 0.545 | 0.404 | 0.430 | eval |
| `pathlens_bce` | pathlens_family | 0.852 | 0.160 | 0.317 | 0.187 | 0.235 | imported |
| `skipgnn` | gnn_baseline | 0.824 | 0.144 | 0.232 | 0.157 | 0.197 | eval |
| `gcn` | gnn_baseline | 0.822 | 0.136 | 0.221 | 0.147 | 0.187 | eval |
| `degree` | heuristic | 0.774 | 0.134 | 0.211 | 0.144 | 0.182 | eval |
| `graphsage` | gnn_baseline | 0.790 | 0.131 | 0.185 | 0.135 | 0.176 | eval |

## PathLens family (one architecture, ablated)

| Method | Group | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Stage |
|---|---|---:|---:|---:|---:|---:|---|
| `pathlens_ranking` | pathlens_family | 0.868 | 0.375 | 0.492 | 0.397 | 0.424 | imported |
| `pathlens_bce` | pathlens_family | 0.852 | 0.160 | 0.317 | 0.187 | 0.235 | imported |
| `s1_s2` | pathlens_family | 0.825 | 0.143 | 0.227 | 0.153 | 0.205 | imported |
| `s1_s2_s3_fixed` | pathlens_family | 0.826 | 0.136 | 0.224 | 0.147 | 0.199 | imported |
| `one_hop` | pathlens_family | 0.788 | 0.103 | 0.192 | 0.114 | 0.161 | imported |
