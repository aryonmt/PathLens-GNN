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


def test_imported_methods_are_marked_done() -> None:
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    imported = ("one_hop", "s1_s2", "s1_s2_s3_fixed", "pathlens_bce", "pathlens_ranking")
    for method_id in imported:
        assert f"`{method_id}`" in status
        provenance = (
            REPO_ROOT / "runs" / "biosnap-dti-v2" / method_id / "imported" / "provenance.json"
        )
        assert provenance.is_file()
    assert "freeze `646700d4`" in status
    assert (REPO_ROOT / "runs" / "biosnap-dti-v2" / "imports" / "manifest.json").is_file()


def test_three_hop_eval_card_is_filed() -> None:
    import json

    from pathlens.evaluation.figures import FIGURE_NAMES

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    metrics_path = campaign / "three_hop" / "eval" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["method"] == "three_hop"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `three_hop` |" in status
    assert "done" in status.split("`three_hop`")[1].split("\n")[0]
    for name in FIGURE_NAMES:
        assert (campaign / "figures" / name).is_file()


def test_runs_campaign_directory_exists() -> None:
    assert (Path(REPO_ROOT) / "runs" / "biosnap-dti-v2").is_dir()


def test_kaggle_drop_layout_exists() -> None:
    from pathlens.evaluation.figures import FIGURE_NAMES

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    assert (REPO_ROOT / "runs" / "README.md").is_file()
    assert (REPO_ROOT / "outputs" / "README.md").is_file()
    assert (campaign / "figures" / "README.md").is_file()
    assert (campaign / "figures" / "test").is_dir()
    for method_id in ("degree", "resource_allocation", "three_hop"):
        assert (campaign / method_id / "eval").is_dir()
    for method_id in ("skipgnn", "gcn", "graphsage"):
        assert (campaign / method_id / "train").is_dir()
        assert (campaign / method_id / "eval").is_dir()
    figures_readme = (campaign / "figures" / "README.md").read_text(encoding="utf-8")
    for name in FIGURE_NAMES:
        assert f"`{name}`" in figures_readme


def test_notebook_defaults_to_degree_eval() -> None:
    import json

    notebook = json.loads(
        (REPO_ROOT / "kaggle" / "pathlens_training.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][1]["source"])
    assert 'METHOD = "degree"' in source
    assert 'STAGE = "eval"' in source
    assert 'FINAL_TEST_TOKEN = ""' in source
    assert "OPEN_SEALED_TEST_ONCE" not in source
    assert 'STAGE = "final"' not in source
    figures_source = "".join(notebook["cells"][4]["source"])
    assert "write_validation_figures" in figures_source
    checkout = "".join(notebook["cells"][2]["source"])
    assert 'repository_source = str(REPO / "src")' in checkout
    assert "sys.path.insert(0, repository_source)" in checkout
