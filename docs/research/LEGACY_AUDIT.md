# Legacy SkipGNN Audit

## What the legacy system did

The original pipeline constructed an adjacency `A`, created a binary skip graph `sign(A @ A)`, propagated dense identity or node2vec features over direct and skip graphs, summed the channels, concatenated candidate node embeddings, and scored them with two consecutive linear layers.

## Critical data defects

- The DTI notebook read the source header as data. `#Drug` and `Gene` became entities, and `(#Drug, Gene)` became a positive interaction.
- The canonical source has 15,138 unique pairs, 5,017 drugs, 2,324 proteins, and 7,341 entities. The legacy entity list has two additional pseudo-entities.
- Approximately 79% of legacy DTI negatives have an invalid type/order. A type-only shortcut achieves roughly 0.895 AUROC on the stored test fold.
- The GDI notebook uses a stale loop variable and produces gene-gene negatives for a gene-disease relation.
- The stored DTI fold contains endpoints unseen in training positives, while the model has only ID-based features; this silently mixes transductive and unsupported cold-start evaluation.

## Evaluation defects

- Validation/test used shuffle and `drop_last=True`.
- Only the final batch loss was reported.
- F1 thresholded logits at 0.5 instead of logits at 0 or probabilities at 0.5.
- Evaluation omitted inference mode and model selection differed from the paper description.
- Training edges were also message-passing edges, so a candidate's own positive edge could be directly visible to the encoder.

## Engineering defects

- Dense identity matrices waste quadratic memory.
- Stored loss tensors retained computation graphs.
- Dependencies were incomplete/unpinned, imports were fragile, and the Windows multiprocessing guard was absent.
- The decoder was effectively linear and order-sensitive for undirected relations.

These repairs are scientific prerequisites, not the claimed model contribution.
