# Legacy Audit and Approved Model Redesign

## 1. Legacy Pipeline

The original code in `legacy/SkipGNN/` follows this pipeline:

1. Positive training edges define the original graph adjacency matrix `A`.
2. The skip adjacency is constructed as `sign(A @ A)`.
3. Input node attributes are either a dense one-hot identity matrix or node2vec vectors.
4. Graph convolution modules propagate over both original and skip graphs.
5. Direct and skip representations are fused by fixed summation.
6. Final node embeddings for a candidate pair are concatenated.
7. A two-linear-layer decoder outputs a binary logit.

The central biological intuition is useful: in molecular interaction networks, directly interacting entities may be dissimilar, while entities sharing interaction partners can be biologically similar.

For a bipartite DTI graph:

```text
Drug A -> Protein X -> Drug B
```

`A^2` connects Drug A and Drug B because they share Protein X.

## 2. Problems That Must Be Corrected

These corrections are scientific validity requirements, not the main model novelty.

### Data schema and negatives

- DTI negatives must always be `(drug, protein)` pairs in the correct typed relation.
- GDI negatives must always be `(gene, disease)` pairs.
- Negative pairs must not overlap any known positive in any split used to define the benchmark.
- Pair leakage and reversed duplicates must be prevented.
- Validation and test negatives must be fixed, deterministic, and stored.
- Add uniform type-aware negatives and hard negatives matched by degree or structural similarity.

### Evaluation

- No shuffling for validation/test.
- No dropped validation/test samples.
- Aggregate loss across every evaluated sample.
- Use logits consistently with `BCEWithLogitsLoss`.
- Select any classification threshold on validation only.
- Report AUROC, AUPRC, F1, MRR, Hits@K, calibration, runtime, and memory.
- Run paired seeds and report mean, standard deviation, and confidence intervals.
- Stratify performance by node degree to detect popularity bias.

### Efficiency and reproducibility

- Replace explicit identity matrices with embedding parameters.
- Keep graph matrices sparse.
- Avoid full-graph recomputation for every small edge mini-batch.
- Use deterministic seeds and stored split manifests.
- Pin dependencies and save configuration with each checkpoint.
- Save state dictionaries and metadata rather than pickled full models.

## 3. Main Model Contribution

Working name: **Path-Aware Adaptive SkipGNN**.

### 3.1 Direct channel

The direct interaction channel operates on the normalized original adjacency:

```text
Z1 = Encoder1(A, X)
```

It captures immediate interaction structure.

### 3.2 Weighted two-hop channel

The original model discards path strength by applying `sign` to `A^2`. The new model retains weighted second-order evidence:

```text
W2 = normalize(weight(A^2))
Z2 = Encoder2(W2, X)
```

Candidate weight functions to ablate:

- binary sign baseline;
- raw path count;
- `log(1 + path_count)`;
- degree-normalized Resource Allocation weighting.

Resource Allocation form:

```text
W2[i,j] = sum over k of A[i,k] * A[k,j] / degree(k)
```

This reduces the influence of generic high-degree intermediates.

### 3.3 Explicit three-hop channel

In bipartite graphs, even-length paths connect same-type nodes and odd-length paths connect opposite-type nodes. DTI and GDI predict opposite-type relations, so explicit three-hop evidence is relevant:

```text
Drug A -> Protein X -> Drug B -> Protein Y
```

The model adds:

```text
W3 = normalize(weight(A^3))
Z3 = Encoder3(W3, X)
```

Do not materialize dense `A^3`. Use sparse products, sampled paths, or candidate-local structural features.

### 3.4 Pair-conditioned adaptive fusion

Legacy SkipGNN uses the same fixed sum for every node and candidate. The new model generates per-pair hop weights:

```text
pair_features = combine(Z1_i, Z1_j, Z2_i, Z2_j, Z3_i, Z3_j, structural_features)
alpha = softmax(gating_mlp(pair_features))
score = alpha1 * direct_score + alpha2 * two_hop_score + alpha3 * three_hop_score
```

The gate provides both additional modeling capacity and an interpretable per-pair summary.

### 3.5 Relation-appropriate decoder

For homogeneous undirected DDI/PPI relations, ensure symmetry with features such as:

- `zi * zj` (Hadamard product);
- `abs(zi - zj)`;
- symmetric aggregation or an explicitly symmetric scoring function.

For typed bipartite DTI/GDI relations, use a typed bilinear decoder or directional nonlinear MLP.

The decoder must include a real non-linearity, unlike the legacy pair of consecutive linear layers.

### 3.6 Path-based explanation

Every prediction should expose:

- direct/two-hop/three-hop gate weights;
- weighted path counts;
- top intermediate nodes;
- top supporting two- or three-hop paths;
- prediction uncertainty if an ensemble is available.

Explanation data must come from model inputs or learned weights, not be fabricated solely for UI presentation.

## 4. Experiment Matrix

### Primary dataset

DTI is the primary task because:

- it exposes the legacy typed-negative failure;
- it has a natural target-prioritization product use case;
- its bipartite structure motivates the explicit three-hop channel.

### Confirmation dataset

PPI is preferred as the second task if time allows because it tests whether the method also helps a homogeneous network.

### Baselines

- type-only shortcut baseline;
- common-neighbor / Resource Allocation / L3 heuristic where applicable;
- GCN;
- GraphSAGE if time allows;
- faithfully reimplemented original SkipGNN;
- weighted `A^2` only;
- `A + weighted A^2`;
- `A + weighted A^2 + weighted A^3`;
- full model with pair-conditioned gate.

### Required ablations

- binary versus weighted `A^2`;
- with and without `A^3`;
- fixed sum versus adaptive gate;
- legacy concatenation versus relation-appropriate decoder;
- uniform type-aware versus hard negatives.

## 5. Success Criteria

The project succeeds if it:

- eliminates schema shortcuts;
- improves or remains competitive with original SkipGNN on valid difficult negatives;
- demonstrates a measurable contribution from at least one new path channel or adaptive gate;
- produces stable results across seeds;
- materially reduces training memory;
- produces faithful path evidence usable by the product.

Raw legacy AUROC is not the sole success criterion because the legacy DTI test set is scientifically compromised.

