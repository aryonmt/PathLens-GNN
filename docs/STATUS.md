# Method status — `biosnap-dti-v2`

Split seed 41. Test sealed. Update this table in the same change that finishes a method.

| Method | Kind | Loss | Train here | Status | Last run |
|---|---|---|---|---|---|
| `degree` | heuristic | none | no | done | eval T4 `5daa27c` |
| `resource_allocation` | heuristic | none | no | done | eval T4 `d2b90ca` |
| `three_hop` | heuristic | none | no | done | eval T4 `46fc64f` |
| `skipgnn` | model | BCE 1:1 | yes | not_started | — |
| `one_hop` | model | BCE 1:1 | import | done | imported v2 `bc7a4d50` |
| `s1_s2` | model | BCE 1:1 | import | done | imported v2 `6d2eb356` |
| `s1_s2_s3_fixed` | model | BCE 1:1 | import | done | imported v2 `74544ab8` |
| `pathlens_bce` | model | BCE 1:1 | import | done | imported v2 `aef8a3ea` |
| `pathlens_ranking` | model | sampled softmax | import | done | freeze `646700d4` |
| `gcn` | model | BCE 1:1 | yes | not_started | — |
| `graphsage` | model | BCE 1:1 | yes | not_started | — |
| `gat` | model | BCE 1:1 | if hours remain | deferred | — |
| `nbfnet` | model | TBD | if hours remain | deferred | — |

Status values: `not_started`, `import_pending`, `running`, `done`, `deferred`.

Imported cards are validation-only from `pathlens-stage-output-v2-report.zip`. Weights stay local and gitignored. `binary_skipgnn` from that ZIP is **not** the official `skipgnn` run. `resource_allocation` is the current validation MRR leader. PathLens siblings stay on the board as an ablation ladder, not as extra systems to beat.
