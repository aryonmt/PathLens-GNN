# Technical narrative — what we tried and what it changed

This is the lab log, not a hero story. The honest result: **on filtered
ranking, parameter-free L3 / resource-allocation beat every learned model we
filed.** GNNs look strong on 1:1 AUC. PathLens ranking loss closed a lot of
the GNN gap and still lost to 3-hop on the preregistered metric.

## Starting point

The project is a leakage-safe comparison on BioSNAP DTI, not a product.
Charter: same split, same metric suite, SkipGNN trained here, PathLens
imported, paper PR-AUC 0.928 is not a baseline.

We locked a coverage-preserving ~60/20/10/10 context / train / val / test
split (seed 41) so every scored node has context degree ≥ 1. Test stayed
sealed until the negative freeze was recorded.

## Why heuristics first

If a GNN cannot beat a cheap path count, the GNN is not doing ranking work.
Degree is the popularity floor. Resource allocation and 3-hop implement the
bipartite fact that **odd hops** are the drug–protein paths.

**Seen:** RA MRR 0.462, 3-hop 0.455, degree 0.134. Hard AUPRC for RA/3-hop
~0.85, degree 0.77. The ranking problem is mostly "does a normalized 3-walk
exist in context?"

## Why SkipGNN, then GCN, then GraphSAGE

SkipGNN is the paper we were handed. We trained the official encoder
(original + \(\mathrm{sign}(A^2)\)) with Huang's BCE 1:1 loop, **selecting
on validation AUROC** so the comparison is fair to the paper's protocol.

**Seen:** SkipGNN MRR 0.144, hard AUPRC 0.824. Best epoch 6 of 15. Val AUROC
0.932. That AUROC does not buy MRR.

GCN asks: is the skip channel doing anything? **Seen:** GCN MRR 0.136. Skip
adds **+0.008 MRR**. Real and small.

GraphSAGE (Hamilton mean, not a Huang number) asks whether a different
aggregator would move ranking. **Seen:** MRR 0.131, worse than GCN. Mean-SAGE
on this bipartite graph is not the missing piece.

GAT / NBFNet stayed deferred: hours were better spent on ranking diagnostics
than on another 1:1-AUC GNN.

## Why the PathLens ladder is nested, not five rivals

PathLens was built to separate hops that SkipGNN mixes. The five cards are
**one architecture, ablated**:

S1 only → +S2 → +fixed S3 → full adaptive gate (BCE) → **same model,
ranking loss**.

**Seen:** extra hops barely move BCE MRR (0.10 → 0.16). Sampled-softmax
jumps to **0.375**. Ranking loss, not S3, is the PathLens result. Confirmation
seeds 13/29/71: 0.373 ± 0.003.

The family champion still loses to 3-hop (0.455) and RA (0.462) on the freeze
metric. That is a negative freeze, not a missing epoch. Campaign v2 PathLens
already used up to 300 epochs.

## Why we mixed PathLens with 3-hop

Hypothesis: PathLens has a complementary ranking signal; late fusion should
beat both.

**Seen:** α-sweep on validation MRR. Any mix with PathLens is ≤ 0.402. Pure
z-scored 3-hop (α=1) is 0.455. Cliff at α=0.9 → 0.379. PathLens is not a
useful additive ranker on this split.

RRF (\(k=60\)) was the **hard-AUPRC winner (0.907)** and a ranking loser
(0.393). We did **not** switch the freeze metric to AUPRC after seeing that.
That would be metric shopping.

## Why residual 3-hop

Hypothesis: keep the heuristic ranks and learn a small correction.

**Seen:** without zero-init on the last layer, epoch 1 already scrambled
3-hop (MRR 0.366 vs 0.455). Training barely moved the loss. Best MRR 0.367
at epoch 10. Same ~10 s GPU budget as SkipGNN. Negative result.

## Why ranking-tie diagnostics

RA / 3-hop looked almost too good. Optimistic ties (`strict_gt`) do not
penalize exact ties. If many targets sit in a huge zero-score pile, MRR is
inflated.

**Seen:** 28.6% of validation positives have 3-hop/RA score **exactly 0**.
Median tied-others = 0; mean is huge because of that pile. PathLens has
**zero ties**. Average-tie MRR: PathLens 0.375, 3-hop 0.362, RA 0.352.
We **left the freeze rule on `strict_gt`**. Changing it after the fact would
make PathLens "win" by redefinition.

The `visible` mask (hide only context∪train∪val, not test) lowers everyone.
That is standard filter-set sensitivity, not a training leak.

Degree tertiles killed another story: PathLens does **not** win the cold-start
tail. Heuristics lead on **low** degree (RA 0.496 vs PathLens 0.250); PathLens
leads on **high** degree (0.466 vs RA 0.410).

## Selection metric vs headline metric

Keep these distinct in the report:

| Model | What selected the checkpoint / mix | What we use to freeze |
|---|---|---|
| SkipGNN, GCN, GraphSAGE | val 1:1 AUROC | val filtered MRR |
| residual 3-hop, blend α | val filtered MRR | val filtered MRR |
| RRF | none (fixed k) | val filtered MRR (it loses) |
| PathLens ranking | campaign v2 ranking protocol | val filtered MRR |

A SkipGNN epoch after 6 might have a different MRR; we did not keep it,
because the paper-faithful loop selects AUROC. RRF has a better hard AUPRC
than 3-hop; we did not keep it, because freeze is MRR.

## Opening test

Freeze rule: freeze a learned candidate only if it beats **`three_hop` and
`skipgnn` on MRR**. PathLens ranking beats SkipGNN and loses to 3-hop.
Record: **negative freeze**. Then open test **once** to confirm order, not
to pick a new winner.

`STAGE=final` requires `FINAL_TEST_TOKEN=OPEN_SEALED_TEST_ONCE`. The Kaggle
delivery notebook scores every method in one Save Version.
