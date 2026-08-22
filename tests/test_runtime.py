from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pathlens.runtime.runner import run_stage


def _prepared(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "entities.json").write_text(
        json.dumps(
            {
                "drugs": ["DB00001", "DB00002", "DB00003", "DB00004"],
                "proteins": ["P00001", "P00002", "P00003", "P00004"],
            }
        ),
        encoding="utf-8",
    )
    (processed / "manifest.json").write_text(json.dumps({"seed": 13}), encoding="utf-8")
    np.savez(
        processed / "splits.npz",
        all_positive=np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3], [3, 0], [3, 2]],
            dtype=np.int64,
        ),
        context=np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
            dtype=np.int64,
        ),
        train_positive=np.asarray([[3, 0]], dtype=np.int64),
        train_uniform=np.asarray([[0, 3]], dtype=np.int64),
        validation_positive=np.asarray([[1, 1]], dtype=np.int64),
        validation_uniform=np.asarray([[2, 0]], dtype=np.int64),
        validation_hard=np.asarray([[2, 0]], dtype=np.int64),
        test_positive=np.asarray([[0, 2]], dtype=np.int64),
        test_uniform=np.asarray([[3, 1]], dtype=np.int64),
        test_hard=np.asarray([[3, 1]], dtype=np.int64),
    )
    return processed


def test_heuristic_smoke_writes_metrics_and_skips_test(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    result = run_stage(
        "three_hop",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=True,
        archive_path=tmp_path / "out.zip",
    )
    assert Path(result["output_dir"], "metrics.json").is_file()
    assert Path(result["archive"]).is_file()
    assert "mrr" in result["filtered_ranking"]
    assert "curves" not in result["classification"]["hard"]
    assert "test" not in result


def test_final_stage_stays_sealed(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    with pytest.raises(PermissionError, match="sealed"):
        run_stage(
            "three_hop",
            "final",
            device="cpu",
            processed_dir=processed,
            output_root=tmp_path / "runs",
            download=False,
            strict_identity=False,
            write_archive=False,
            final_test_token="OPEN_SEALED_TEST_ONCE",
        )


def test_heuristic_eval_includes_curves(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    result = run_stage(
        "three_hop",
        "eval",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert "curves" in result["classification"]["hard"]
    assert "degree_slices" in result["classification"]["hard"]
    assert "ndcg_at_10" in result["filtered_ranking"]
    assert "test" not in result


def test_skipgnn_without_torch_explains_the_train_extra(tmp_path: Path) -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed")
    processed = _prepared(tmp_path)
    with pytest.raises(RuntimeError, match="PyTorch"):
        run_stage(
            "skipgnn",
            "smoke",
            device="cpu",
            processed_dir=processed,
            output_root=tmp_path / "runs",
            download=False,
            strict_identity=False,
            write_archive=False,
        )


def test_skipgnn_smoke_writes_metrics_and_skips_test(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    processed = _prepared(tmp_path)
    result = run_stage(
        "skipgnn",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert Path(result["output_dir"], "metrics.json").is_file()
    assert "mrr" in result["filtered_ranking"]
    assert "curves" not in result["classification"]["hard"]
    assert "test" not in result
    assert result["training"]["epochs"] == 2
    assert result["training"]["selection"] == "validation_auroc_1to1"


def test_skipgnn_eval_includes_curves_and_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    processed = _prepared(tmp_path)
    result = run_stage(
        "skipgnn",
        "eval",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert "curves" in result["classification"]["hard"]
    assert "degree_slices" in result["classification"]["hard"]
    assert "test" not in result
    assert Path(result["checkpoint"]).is_file()


def test_gcn_smoke_writes_metrics_and_skips_test(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    processed = _prepared(tmp_path)
    result = run_stage(
        "gcn",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert Path(result["output_dir"], "metrics.json").is_file()
    assert "mrr" in result["filtered_ranking"]
    assert "curves" not in result["classification"]["hard"]
    assert "test" not in result
    assert result["training"]["epochs"] == 2


def test_graphsage_smoke_writes_metrics_and_skips_test(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    processed = _prepared(tmp_path)
    result = run_stage(
        "graphsage",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert Path(result["output_dir"], "metrics.json").is_file()
    assert "mrr" in result["filtered_ranking"]
    assert "test" not in result
    assert result["training"]["epochs"] == 2


def test_residual_smoke_writes_metrics_and_skips_test(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    processed = _prepared(tmp_path)
    result = run_stage(
        "residual_three_hop",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert Path(result["output_dir"], "metrics.json").is_file()
    assert "mrr" in result["filtered_ranking"]
    assert "test" not in result
    assert result["training"]["epochs"] == 2
    assert result["training"]["selection"] == "validation_filtered_mrr"


def test_blend_writes_rrf_sibling_from_injected_scores(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    hop = np.zeros((4, 4), dtype=np.float64)
    hop[1, 1] = 5.0
    pathlens = np.zeros((4, 4), dtype=np.float64)
    pathlens[1, 3] = 5.0
    result = run_stage(
        "blend_pathlens_three_hop",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
        hop_scores=hop,
        pathlens_scores=pathlens,
    )
    assert result["method"] == "blend_pathlens_three_hop"
    assert "test" not in result
    assert result["combine"]["alpha"] >= 0.5
    by_alpha = {row["alpha"]: row["mrr"] for row in result["combine"]["sweep"]}
    assert by_alpha[1.0] > by_alpha[0.0]
    rrf = tmp_path / "runs" / "rrf_pathlens_three_hop" / "smoke" / "metrics.json"
    assert rrf.is_file()
    assert json.loads(rrf.read_text(encoding="utf-8"))["method"] == "rrf_pathlens_three_hop"


def test_combine_train_is_refused(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    with pytest.raises(ValueError, match="frozen-score mix"):
        run_stage(
            "blend_pathlens_three_hop",
            "train",
            device="cpu",
            processed_dir=processed,
            output_root=tmp_path / "runs",
            download=False,
            strict_identity=False,
            write_archive=False,
            hop_scores=np.zeros((4, 4)),
            pathlens_scores=np.zeros((4, 4)),
        )


def test_gat_trainer_is_not_implemented_yet(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    with pytest.raises(NotImplementedError, match="no trainer"):
        run_stage(
            "gat",
            "smoke",
            device="cpu",
            processed_dir=processed,
            output_root=tmp_path / "runs",
            download=False,
            strict_identity=False,
            write_archive=False,
        )
