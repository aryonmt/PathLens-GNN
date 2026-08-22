# Method status — `biosnap-dti-v2`

Split seed 41. Test sealed. Update this table in the same change that finishes a method.

| Method | Kind | Loss | Train here | Status | Last run |
|---|---|---|---|---|---|
| `degree` | heuristic | none | no | not_started | — |
| `resource_allocation` | heuristic | none | no | not_started | — |
| `three_hop` | heuristic | none | no | not_started | — |
| `skipgnn` | model | BCE 1:1 | yes | not_started | — |
| `one_hop` | model | BCE 1:1 | import | import_pending | v2 registered |
| `s1_s2` | model | BCE 1:1 | import | import_pending | v2 registered |
| `s1_s2_s3_fixed` | model | BCE 1:1 | import | import_pending | v2 registered |
| `pathlens_bce` | model | BCE 1:1 | import | import_pending | v2 registered |
| `pathlens_ranking` | model | sampled softmax | import | import_pending | v2 freeze `646700d4` |
| `gcn` | model | BCE 1:1 | yes | not_started | — |
| `graphsage` | model | BCE 1:1 | yes | not_started | — |
| `gat` | model | BCE 1:1 | if hours remain | deferred | — |
| `nbfnet` | model | TBD | if hours remain | deferred | — |

Status values: `not_started`, `import_pending`, `running`, `done`, `deferred`.
