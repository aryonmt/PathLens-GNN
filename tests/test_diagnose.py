from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathlens.runtime.runner import run_stage


def _prepared(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "entities.json").write_text(
        '{"drugs": ["DB00001", "DB00002", "DB00003", "DB00004"], '
        '"proteins": ["P00001", "P00002", "P00003", "P00004"]}',
        encoding="utf-8",
    )
    (processed / "manifest.json").write_text('{"seed": 41}', encoding="utf-8")
    np.savez(
        processed / "splits.npz",
        all_positive=np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3], [3, 0], [3, 2], [0, 2]],
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


def test_diagnostic_smoke_writes_heuristic_columns_and_skips_test(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    result = run_stage(
        "ranking_diagnostics",
        "smoke",
        device="cpu",
        processed_dir=processed,
        output_root=tmp_path / "runs",
        download=False,
        strict_identity=False,
        write_archive=False,
    )
    assert result["diagnostic"] is True
    assert "test" not in result
    assert set(result["methods"]) == {"degree", "resource_allocation", "three_hop"}
    three = result["methods"]["three_hop"]
    assert set(three) == {"all_positive", "visible"}
    assert set(three["all_positive"]["ranking"]) == {"strict_gt", "average", "random"}
    assert "fraction_zero_target" in three["all_positive"]["ties"]
    assert "mrr_by_drug_degree" in three["all_positive"]
    assert "pathlens_ranking" not in result["methods"]
    assert Path(result["output_dir"], "metrics.json").is_file()


def test_diagnostic_train_is_refused(tmp_path: Path) -> None:
    processed = _prepared(tmp_path)
    with pytest.raises(ValueError, match="diagnostic"):
        run_stage(
            "ranking_diagnostics",
            "train",
            device="cpu",
            processed_dir=processed,
            output_root=tmp_path / "runs",
            download=False,
            strict_identity=False,
            write_archive=False,
        )


def test_visible_filter_is_context_train_validation_only(tmp_path: Path) -> None:
    from pathlens.data.processed import load_processed
    from pathlens.runtime.diagnose import visible_positive_pairs

    processed = _prepared(tmp_path)
    split = load_processed(processed, allow_test=False)
    visible = {tuple(row) for row in visible_positive_pairs(split)}
    all_positive = {tuple(row) for row in split.all_positive}
    assert (0, 2) in all_positive
    assert (0, 2) not in visible
    assert (1, 1) in visible
    assert (3, 0) in visible
