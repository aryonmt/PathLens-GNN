from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pathlens_gnn.evaluation.metrics import classification_report
from pathlens_gnn.graph.index import BipartiteIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate registered non-learned DTI baselines.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = np.load(args.processed / "splits.npz")
    entities = json.loads((args.processed / "entities.json").read_text(encoding="utf-8"))
    graph = BipartiteIndex(len(entities["drugs"]), len(entities["proteins"]), arrays["context"])
    results: dict[str, object] = {}
    for negative_name in ("validation_uniform", "validation_hard"):
        positives = arrays["validation_positive"]
        negatives = arrays[negative_name]
        pairs = np.concatenate((positives, negatives)).astype(np.int64)
        labels = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives)))).astype(
            np.int64
        )
        degree_scores = np.asarray(
            [
                np.log1p(
                    len(graph.drug_neighbors[int(drug)])
                    * len(graph.protein_neighbors[int(protein)])
                )
                for drug, protein in pairs
            ]
        )
        bridge_scores = graph.structural_features(pairs)[:, 1]
        results[negative_name] = {
            "degree_type_shortcut": classification_report(labels, degree_scores).to_dict(),
            "normalized_three_hop": classification_report(labels, bridge_scores).to_dict(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
