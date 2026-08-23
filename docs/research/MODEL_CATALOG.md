# Model catalog — `biosnap-dti-v2`

Exact definitions. All methods share campaign `biosnap-dti-v2`, BioSNAP DTI,
split seed **41**, Tesla T4 unless marked imported. Validation queries: 1,514.
Test stays unused until `STAGE=final`.

Graph convention: \(B \in \{0,1\}^{n_d \times n_p}\) is the **context**
bipartite adjacency unless a trainer paragraph says `visible = context ∪ train_positive`.

## Shared evaluation

- **Filtered ranking:** for each validation (or test) drug–protein positive,
  rank all proteins; mask other known positives from `all_positive` (includes
  test edges in the mask, not in the score). Rank =
  \(1 + \#\{\text{strictly higher scores}\}\) (`strict_gt`). Ties do not push
  the target down.
- **Hard classification:** 1:1 degree-matched unknown non-edges.
- **Uniform classification:** 1:1 typed random unknown non-edges.
- Primary freeze metric: validation filtered MRR. Tie-break: hard AUPRC.

Filed numbers and sources: [`REPORT.md`](REPORT.md), [`STATUS.md`](../STATUS.md).

---

## Heuristics (context only, no training)

### `degree`

\[\mathrm{score}(d,p) = \log\bigl(1 + \deg(d)\,\deg(p)\bigr)\]

Kind: popularity. Expected to be weak on ranking; it is the floor.

### `resource_allocation` (RA / Adamic–Adar family)

Drug–drug projection with \(1/\deg(p)\) on the shared protein, zero diagonal,
then one hop back:

\[
W = \mathrm{zerodiag}\bigl(B\,\mathrm{diag}(1/\deg p)\,B^\top\bigr),\quad
\mathrm{score} = W B
\]

Code: `score_resource_allocation` in `src/pathlens/graph/scoring.py`.

### `three_hop` (normalized L3 / Kovács-style)

Same as RA with an extra \(1/\deg(d')\) on the intermediate drug:

\[
\mathrm{score} = \bigl(W \odot (1/\deg d')^\top\bigr) B
\]

This is the first **opposite-type** return walk on a bipartite graph. It is
not a leak of the test set.

---

## SkipGNN family (trained here, BCE 1:1)

Visible graph: `context ∪ train_positive`. Features: identity. Decoder: two-layer
linear on concatenated embeddings (Huang clone). **Selection: best validation
1:1 AUROC**, not MRR. Seed 13. Max 15 epochs. That is why a later epoch with
a different ranking profile is not the filed checkpoint.

### `skipgnn`

Original GCN channel + skip channel \(\mathrm{sign}(A^2)\). Best epoch **6**,
val AUROC 0.932. Filed MRR 0.144, hard AUPRC 0.824. GPU ~10.3 s.
Eval T4 `56970fe`.

### `gcn`

Same loop, original graph only (skip channel dropped). Best epoch **8**,
val AUROC 0.929. Filed MRR 0.136, hard AUPRC 0.822. Eval T4 `9fa9913`.

### `graphsage`

Hamilton **mean-SAGE** on unnormalized symmetrized \(A\), not a Huang et al.
paper number. Best epoch **9**, val AUROC 0.917. Filed MRR 0.131, hard AUPRC
0.790. Eval T4 `4138751`.

---

## PathLens family (imported, one architecture)

Checkpoints live at `runs/biosnap-dti-v2/<id>/imported/checkpoint.pt`.
Do not retrain. Message-passing graph: **context only** (legacy2 runner).
S1 / S2 / S3 are hop-separated operators in
`legacy2/src/pathlens_gnn/model/operators.py`:

| Channel | What it is | Opposite-type path? |
|---|---|---|
| S1 (`one_hop`) | Normalized 1-hop over \(B\) | yes |
| S2 (`two_hop`) | Same-type RA projection | no |
| S3 (`three_hop`) | S1 after S2 | yes |

Each hop has a branch MLP and a typed expert. An adaptive gate mixes experts
when enabled. Pair features include projection mass and a frozen 3-hop bridge.

| Card | Isolated change | Loss | Val MRR | Val hard AUPRC | Checkpoint SHA-256 prefix |
|---|---|---|---|---|---|
| `one_hop` | S1 only | BCE 1:1 | 0.103 | 0.788 | `bc7a4d50` |
| `s1_s2` | + S2 | BCE 1:1 | 0.143 | 0.825 | `6d2eb356` |
| `s1_s2_s3_fixed` | + S3, fixed mix | BCE 1:1 | 0.136 | 0.826 | `74544ab8` |
| `pathlens_bce` | full adaptive gate | BCE 1:1 | 0.160 | 0.852 | `aef8a3ea` |
| `pathlens_ranking` | same as BCE, sampled softmax | InfoNCE, 64 neg, 25% hard | 0.375 | 0.868 | `646700d4` |

`pathlens_ranking` confirmation seeds 13/29/71: MRR **0.373 ± 0.003**,
hard AUPRC **0.883 ± 0.013**. Freeze checkpoint SHA-256:

`646700d456ba4cb0de795e7067e1830e8f18598b97fe5cc89d195d631d356586`

**Reporting rule:** nested hops are not rival systems. Head-to-head tables use
only `pathlens_bce` and `pathlens_ranking` from this family.

Ranking loss, not extra hops, is what moved PathLens (BCE cards stay at
MRR 0.10–0.16). Campaign v2 PathLens training was up to 300 epochs (hours).
That is a different compute class from the ~10 s SkipGNN loop in this repo.

---

## Mixes and residual

### `blend_pathlens_three_hop`

Per-drug z-score, then \(\alpha\,z(3\text{-hop}) + (1-\alpha)\,z(\text{PathLens})\).
\(\alpha\) selected on **validation filtered MRR**. Selected \(\alpha = 1.0\)
(pure z-scored 3-hop). Sweep (MRR): α=0 → 0.375, best mix α=0.4 → 0.402,
α=0.9 → 0.379, **α=1.0 → 0.455**. Mixing never beat 3-hop on MRR.
Filed hard AUPRC 0.887 (z-score lifts classification, not ranks).
Eval T4 `73d5a2a`.

### `rrf_pathlens_three_hop`

Reciprocal rank fusion, \(k=60\). Best hard AUPRC in the scoreboard
(**0.907**), MRR 0.393. **Not selected**, because the freeze metric is MRR.

This is the clean example of "better hard AUC, different winner": RRF would
win an AUPRC contest; it loses the preregistered ranking contest.

### `residual_three_hop`

\(\mathrm{score} = 3\text{-hop} + \mathrm{MLP}(d,p)\). Sampled-softmax, select
on **validation filtered MRR**. Last MLP layer was **not** zero-init. Epoch 1
MRR already 0.366 (below frozen 3-hop 0.455). Loss 4.096 → 4.064. Stopped
epoch 18, best epoch **10**, MRR 0.367, hard AUPRC 0.889. GPU 10.4 s.
Negative result, not a truncated PathLens job. Eval T4 `a5506bb`.

---

## Deferred

`gat`, `nbfnet`: not run.

## Diagnostic (not a method)

`ranking_diagnostics`: tie modes (`strict_gt` / `average` / `random`),
`all_positive` vs `visible` mask, degree-tertile MRR. Eval T4 `90202e6`.
Does not replace filed cards.

Under **average-tie**, PathLens 0.375 slightly beats 3-hop 0.362 and RA 0.352.
The freeze rule was **not** changed after seeing that.
