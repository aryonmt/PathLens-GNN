from __future__ import annotations

from pathlib import Path

from pathlens import METHODS_DIR, REPO_ROOT, method_ids

REQUIRED = {
    "degree",
    "resource_allocation",
    "three_hop",
    "skipgnn",
    "one_hop",
    "s1_s2",
    "s1_s2_s3_fixed",
    "pathlens_bce",
    "pathlens_ranking",
    "gcn",
    "graphsage",
    "gat",
    "nbfnet",
}


def test_every_phase1_method_has_a_card() -> None:
    assert set(method_ids()) == REQUIRED
    for method_id in REQUIRED:
        folder = METHODS_DIR / method_id
        assert (folder / "METHOD.md").is_file()
        assert (folder / "config.yaml").is_file()


def test_status_board_lists_every_method() -> None:
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    for method_id in REQUIRED:
        assert f"`{method_id}`" in status


def test_runs_campaign_directory_exists() -> None:
    assert (Path(REPO_ROOT) / "runs" / "biosnap-dti-v2").is_dir()


def test_notebook_defaults_to_smoke() -> None:
    import json

    notebook = json.loads(
        (REPO_ROOT / "kaggle" / "pathlens_training.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][1]["source"])
    assert 'STAGE = "smoke"' in source
    assert "METHOD" in source
    assert "OPEN_SEALED_TEST_ONCE" not in source
