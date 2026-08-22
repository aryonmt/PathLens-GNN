from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pathlens.constants import IMPORT_METHODS
from pathlens.evaluation.figures import load_validation_report
from pathlens.evaluation.ranking import ranking_report_from_ranks
from pathlens.importing.campaign_v2 import import_v2_report

TOY_RANKS = np.asarray([1, 2], dtype=np.int64)
TOY_MRR = float(np.mean(1.0 / TOY_RANKS))


def _toy_model(mrr: float, auprc: float) -> dict[str, object]:
    return {
        "filtered_ranking": {
            "mrr": mrr,
            "hits_at_1": 0.1,
            "hits_at_3": 0.2,
            "hits_at_5": 0.3,
            "hits_at_10": 0.4,
            "hits_at_20": 0.5,
            "hits_at_50": 0.6,
            "queries": 10,
        },
        "classification": {
            "hard": {
                "auprc": auprc,
                "auroc": 0.8,
                "auprc_bootstrap_95_ci": [0.7, 0.9],
                "curves": {"recall": [0.0, 1.0], "precision": [1.0, 0.5]},
            },
            "uniform": {"auprc": auprc + 0.01, "curves": {"recall": [0.0, 1.0]}},
        },
    }


def _write_report_zip(path: Path, *, include_test: bool = False) -> str:
    freeze_bytes = b"freeze-seed-13-checkpoint"
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    models = {
        "full_adaptive_ranking": _toy_model(TOY_MRR, 0.868),
        "degree_type_shortcut": _toy_model(TOY_MRR, 0.77),
        "normalized_three_hop": _toy_model(TOY_MRR, 0.85),
        "one_hop": _toy_model(TOY_MRR, 0.79),
        "binary_skipgnn": _toy_model(TOY_MRR, 0.83),
        "s1_s2_weighted": _toy_model(TOY_MRR, 0.83),
        "s1_s2_s3_fixed": _toy_model(TOY_MRR, 0.83),
        "full_adaptive_fusion": _toy_model(TOY_MRR, 0.85),
    }
    report = {
        "split": "test" if include_test else "validation",
        "checkpoint_sha256": freeze_sha,
        "models": models,
    }
    if include_test:
        report["test_positive"] = [[0, 1]]
    freeze = {
        "checkpoint_sha256": freeze_sha,
        "selection_metric": "validation_filtered_mrr",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("validation-report/validation-report.json", json.dumps(report))
        archive.writestr("model-freeze.json", json.dumps(freeze))
        archive.writestr("confirmation/candidate-0/seed-13/checkpoint.pt", freeze_bytes)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            **{
                f"{name}__ranks": TOY_RANKS
                for name in (
                    "full_adaptive_ranking",
                    "one_hop",
                    "s1_s2_weighted",
                    "s1_s2_s3_fixed",
                    "full_adaptive_fusion",
                )
            },
        )
        archive.writestr("validation-report/validation-report.npz", buffer.getvalue())
        for member in (
            "registered/one_hop/checkpoint.pt",
            "registered/s1_s2_weighted/checkpoint.pt",
            "registered/s1_s2_s3_fixed/checkpoint.pt",
            "registered/full_adaptive_fusion/checkpoint.pt",
        ):
            archive.writestr(member, member.encode("utf-8"))
    return freeze_sha


def test_import_v2_writes_provenance_and_skips_skipgnn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_sha = hashlib.sha256(b"freeze-seed-13-checkpoint").hexdigest()
    monkeypatch.setattr("pathlens.importing.campaign_v2.V2_FREEZE_CHECKPOINT_SHA256", freeze_sha)
    archive = tmp_path / "report.zip"
    assert _write_report_zip(archive) == freeze_sha
    catalog = import_v2_report(archive, tmp_path / "runs", copy_weights=True)
    assert set(catalog["methods"]) == set(IMPORT_METHODS)
    assert "skipgnn" not in catalog["methods"]
    one_hop = tmp_path / "runs" / "one_hop" / "imported"
    metrics = json.loads((one_hop / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((one_hop / "provenance.json").read_text(encoding="utf-8"))
    assert metrics["method"] == "one_hop"
    assert metrics["classification"]["hard"]["auprc_ci"] == [0.7, 0.9]
    assert metrics["filtered_ranking"]["ndcg_at_10"] == pytest.approx(
        ranking_report_from_ranks(TOY_RANKS).ndcg_at_10
    )
    assert "ndcg_at_50" in metrics["filtered_ranking"]
    assert provenance["retrain"] is False
    assert (one_hop / "checkpoint.pt").is_file()
    ranking = json.loads(
        (tmp_path / "runs" / "pathlens_ranking" / "imported" / "provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert ranking["checkpoint_sha256"] == freeze_sha
    assert catalog["test_sealed"] is True
    assert catalog["methods"]["one_hop"]["output_dir"].replace("\\", "/").endswith(
        "one_hop/imported"
    )
    figure_report = load_validation_report(tmp_path / "runs")
    assert set(figure_report["models"]) == set(IMPORT_METHODS)


def test_import_v2_refuses_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pathlens.importing.campaign_v2.V2_FREEZE_CHECKPOINT_SHA256",
        hashlib.sha256(b"freeze-seed-13-checkpoint").hexdigest(),
    )
    archive = tmp_path / "report.zip"
    _write_report_zip(archive, include_test=True)
    with pytest.raises(ValueError, match="validation"):
        import_v2_report(archive, tmp_path / "runs", copy_weights=False)
