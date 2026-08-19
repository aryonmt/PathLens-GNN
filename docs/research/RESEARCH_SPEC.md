# Research and Model Specification

## Task

Transductive missing-link prioritization over a bipartite drug-protein graph. Every scored endpoint must have at least one edge in the context graph. Non-edges are unknown candidates, not verified biological negatives.

## Sparse operators

Let `B` be the binary drug-by-protein incidence matrix built only from context edges.

- `S1`: symmetrically normalized bipartite one-hop propagation.
- `S2`: resource-allocation projection `B Dp^-1 B^T` for drugs and its protein analogue, normalized and with diagonal return walks removed.
- `S3`: one-hop/projection composition evaluated as sequential sparse multiplications. A dense `A^2` or `A^3` is forbidden.

Each branch applies one sparse propagation followed by a learned linear projection, GELU, dropout, and LayerNorm. Node inputs are learned embeddings.

## Decoder and fusion

Each channel has a typed nonlinear expert over `[z_drug, z_protein, z_drug * z_protein]` and emits `s1`, `s2`, or `s3`. A gate consumes all channel embeddings and log-scaled projection/bridge statistics and emits `alpha = softmax(...)`. The final logit is exactly `sum(alpha_h * s_h)`.

No post-fusion MLP is allowed because it would make channel-logit contributions unfaithful.

## Interpretation contract

- Channel contribution: `alpha_h * s_h`; model behavior, not causal evidence.
- Two-hop context: most similar drugs sharing targets and proteins sharing drugs; it is not a drug-protein path.
- Three-hop bridge: up to five ranked paths `drug -> protein_1 -> drug_2 -> protein` with degree-normalized weight.

## Baselines and ablations

Degree/type shortcut, normalized three-hop heuristic, one-hop encoder, binary SkipGNN, S1, S1+weighted-S2 fixed fusion, S1+weighted-S2+S3 fixed fusion, and full adaptive fusion.

## Split discipline

Research evaluation uses sealed splits. After freeze, the test set is opened at most once. That report must never be used to retune or reselect a model on the same split. A later architecture change requires a new preregistered coverage-preserving split.
