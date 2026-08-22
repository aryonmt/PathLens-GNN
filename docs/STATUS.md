# Method status — `biosnap-dti-v2`

Split seed 41. Test sealed. Update this table in the same change that finishes a method.

| Method | Kind | Loss | Train here | Status | Last run |
|---|---|---|---|---|---|
| `degree` | heuristic | none | no | done | eval T4 `5daa27c` |
| `resource_allocation` | heuristic | none | no | done | eval T4 `d2b90ca` |
| `three_hop` | heuristic | none | no | done | eval T4 `46fc64f` |
| `skipgnn` | model | BCE 1:1 | yes | done | eval T4 `56970fe` |
| `one_hop` | model | BCE 1:1 | import | done | imported v2 `bc7a4d50` |
| `s1_s2` | model | BCE 1:1 | import | done | imported v2 `6d2eb356` |
| `s1_s2_s3_fixed` | model | BCE 1:1 | import | done | imported v2 `74544ab8` |
| `pathlens_bce` | model | BCE 1:1 | import | done | imported v2 `aef8a3ea` |
| `pathlens_ranking` | model | sampled softmax | import | done | freeze `646700d4` |
| `gcn` | model | BCE 1:1 | yes | done | eval T4 `9fa9913` |
| `graphsage` | model | BCE 1:1 | yes | done | eval T4 `4138751` |
| `gat` | model | BCE 1:1 | if hours remain | deferred | — |
| `nbfnet` | model | TBD | if hours remain | deferred | — |

Status values: `not_started`, `import_pending`, `running`, `done`, `deferred`.

Imported cards are validation-only from `pathlens-stage-output-v2-report.zip`. Weights stay local and gitignored. Official `skipgnn` (`56970fe`), `gcn` (`9fa9913`), and `graphsage` (`4138751`) are in-repo GPU cards, not the v2 `binary_skipgnn` overlay. GraphSAGE is Hamilton mean-SAGE (Huang et al. did not report it). `resource_allocation` remains the validation MRR leader.
