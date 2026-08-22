from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pathlens.evaluation.figures import (
    FIGURE_NAMES,
    copy_figure_files,
    load_validation_report,
    synthetic_report,
)
from pathlens.runtime.runner import archive_run_dir


def test_figure_catalog_is_locked() -> None:
    assert FIGURE_NAMES == (
        "pr_hard.png",
        "pr_uniform.png",
        "roc_hard.png",
        "hits_at_k.png",
        "mrr_and_hard_auprc.png",
        "degree_slices_hard.png",
    )
    report = synthetic_report()
    assert "three_hop" in report["models"]
    assert "curves" in report["models"]["three_hop"]["classification"]["hard"]


def _write_metrics(path: Path, method_id: str, mrr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method_id,
                "filtered_ranking": {"mrr": mrr, "hits_at_10": mrr},
                "classification": {"hard": {"auprc": mrr}, "uniform": {"auprc": mrr}},
            }
        ),
        encoding="utf-8",
    )


def test_load_runs_directory_prefers_eval_over_imported(tmp_path: Path) -> None:
    root = tmp_path / "biosnap-dti-v2"
    _write_metrics(root / "one_hop" / "imported" / "metrics.json", "one_hop", 0.10)
    _write_metrics(root / "three_hop" / "imported" / "metrics.json", "three_hop", 0.11)
    _write_metrics(root / "three_hop" / "eval" / "metrics.json", "three_hop", 0.45)
    (root / "imports").mkdir()
    (root / "imports" / "manifest.json").write_text("{}", encoding="utf-8")

    report = load_validation_report(root)

    assert set(report["models"]) == {"one_hop", "three_hop"}
    assert report["models"]["three_hop"]["filtered_ranking"]["mrr"] == 0.45
    assert report["models"]["one_hop"]["filtered_ranking"]["mrr"] == 0.10


def test_stage_archive_includes_copied_figures(tmp_path: Path) -> None:
    png = tmp_path / "pr_hard.png"
    png.write_bytes(b"png")
    run_dir = tmp_path / "eval"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    copied = copy_figure_files([png], run_dir / "figures")
    archive = archive_run_dir(run_dir, tmp_path / "pathlens-stage-output.zip")
    with zipfile.ZipFile(archive) as bundle:
        names = {name.replace("\\", "/") for name in bundle.namelist()}
    assert copied[0] == run_dir / "figures" / "pr_hard.png"
    assert "metrics.json" in names
    assert "figures/pr_hard.png" in names
