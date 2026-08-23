from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.ranking import (
    filtered_per_drug_ranking_from_scores,
    known_proteins_by_drug,
)
from pathlens.evaluation.report import evaluate_for_stage
from pathlens.graph.combine import (
    BLEND_ALPHAS,
    blend_scores,
    mean_rank_scores,
    reciprocal_rank_fusion,
    select_blend_alpha,
)
from pathlens.graph.scoring import build_adjacency, score_three_hop
from pathlens.runtime.device import as_numpy
from pathlens.runtime.pathlens_infer import score_frozen_pathlens

BLEND_METHOD = "blend_pathlens_three_hop"
RRF_METHOD = "rrf_pathlens_three_hop"
FIXED_BLEND_ALPHA = 0.5


def combine_pathlens_three_hop(
    split: ProcessedSplit,
    *,
    device: str,
    stage: str,
    repo_root: str | Path | None = None,
    hop_scores: NDArray[np.floating] | None = None,
    pathlens_scores: NDArray[np.floating] | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    hop = (
        np.asarray(hop_scores, dtype=np.float64)
        if hop_scores is not None
        else as_numpy(
            score_three_hop(
                build_adjacency(split.num_drugs, split.num_proteins, split.context, device)
            )
        ).astype(np.float64, copy=False)
    )
    pathlens = (
        np.asarray(pathlens_scores, dtype=np.float64)
        if pathlens_scores is not None
        else score_frozen_pathlens(
            split,
            device=device,
            checkpoint=checkpoint,
            repo_root=repo_root,
        )
    )
    known = known_proteins_by_drug(split.all_positive)
    alpha, sweep = select_blend_alpha(
        hop,
        pathlens,
        split.validation_positive,
        known_by_drug=known,
        alphas=BLEND_ALPHAS,
    )
    blended = blend_scores(hop, pathlens, alpha)
    fixed = blend_scores(hop, pathlens, FIXED_BLEND_ALPHA)
    rrf = reciprocal_rank_fusion(hop, pathlens)
    mean_rank = mean_rank_scores(hop, pathlens)

    blend_payload = evaluate_for_stage(blended, split, stage)
    blend_payload["combine"] = {
        "kind": "late_fusion",
        "left": "three_hop",
        "right": "pathlens_ranking",
        "alpha": alpha,
        "alpha_meaning": "alpha * zscore(three_hop) + (1-alpha) * zscore(pathlens_ranking)",
        "alpha_selected_on": "validation_filtered_mrr",
        "sweep": sweep,
        "fixed_alpha": FIXED_BLEND_ALPHA,
        "fixed_alpha_mrr": filtered_per_drug_ranking_from_scores(
            split.validation_positive,
            scores=fixed,
            known_by_drug=known,
        ).mrr,
    }

    rrf_payload = evaluate_for_stage(rrf, split, stage)
    rrf_payload["combine"] = {
        "kind": "reciprocal_rank_fusion",
        "left": "three_hop",
        "right": "pathlens_ranking",
        "k": 60,
        "mean_rank_mrr": filtered_per_drug_ranking_from_scores(
            split.validation_positive,
            scores=mean_rank,
            known_by_drug=known,
        ).mrr,
    }
    return {"blend": blend_payload, "rrf": rrf_payload}
