from __future__ import annotations

from pathlib import Path

import numpy as np

from pathlens_gnn.artifact.runtime import ArtifactRuntime

FIXTURE = Path("artifacts/fixtures/demo-v1")


def test_fixture_runtime_scores_with_faithful_contributions() -> None:
    runtime = ArtifactRuntime(FIXTURE)
    score = runtime.score_pair(0, 1, (1.0, 0.5))
    assert 0.0 <= score.pathlens_score <= 100.0
    assert np.isclose(sum(score.gate_weights), 1.0)
    assert np.isclose(sum(score.contributions), score.logit)
    runtime.close()
