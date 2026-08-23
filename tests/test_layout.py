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
    "blend_pathlens_three_hop",
    "rrf_pathlens_three_hop",
    "residual_three_hop",
    "ranking_diagnostics",
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


def test_imported_pathlens_checkpoints_are_in_the_repo() -> None:
    import hashlib

    from pathlens.constants import IMPORT_METHODS, V2_FREEZE_CHECKPOINT_SHA256

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    for method_id in IMPORT_METHODS:
        checkpoint = campaign / method_id / "imported" / "checkpoint.pt"
        assert checkpoint.is_file(), method_id
    freeze = campaign / "pathlens_ranking" / "imported" / "checkpoint.pt"
    digest = hashlib.sha256(freeze.read_bytes()).hexdigest()
    assert digest == V2_FREEZE_CHECKPOINT_SHA256


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


def test_degree_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "degree" / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "degree"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `degree` |" in status
    assert "done" in status.split("`degree`")[1].split("\n")[0]


def test_graphsage_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "graphsage" / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "graphsage"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    assert (campaign / "graphsage" / "eval" / "provenance.json").is_file()
    assert not (campaign / "graphsage" / "eval" / "model.pt").exists()
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `graphsage` |" in status
    assert "done" in status.split("`graphsage`")[1].split("\n")[0]


def test_gcn_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "gcn" / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "gcn"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    assert (campaign / "gcn" / "eval" / "provenance.json").is_file()
    assert not (campaign / "gcn" / "eval" / "model.pt").exists()
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `gcn` |" in status
    assert "done" in status.split("`gcn`")[1].split("\n")[0]


def test_skipgnn_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "skipgnn" / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "skipgnn"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    assert (campaign / "skipgnn" / "eval" / "provenance.json").is_file()
    assert not (campaign / "skipgnn" / "eval" / "model.pt").exists()
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `skipgnn` |" in status
    assert "done" in status.split("`skipgnn`")[1].split("\n")[0]


def test_resource_allocation_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "resource_allocation" / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["method"] == "resource_allocation"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `resource_allocation` |" in status
    assert "done" in status.split("`resource_allocation`")[1].split("\n")[0]


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
    for method_id in ("skipgnn", "gcn", "graphsage", "residual_three_hop"):
        assert (campaign / method_id / "train").is_dir()
        assert (campaign / method_id / "eval").is_dir()
    for method_id in ("blend_pathlens_three_hop", "rrf_pathlens_three_hop", "ranking_diagnostics"):
        assert (campaign / method_id / "eval").is_dir()
    figures_readme = (campaign / "figures" / "README.md").read_text(encoding="utf-8")
    for name in FIGURE_NAMES:
        assert f"`{name}`" in figures_readme


def test_blend_and_rrf_eval_cards_are_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    for method_id in ("blend_pathlens_three_hop", "rrf_pathlens_three_hop"):
        payload = json.loads(
            (campaign / method_id / "eval" / "metrics.json").read_text(encoding="utf-8")
        )
        assert payload["method"] == method_id
        assert payload["stage"] == "eval"
        assert "test" not in payload
        assert (campaign / method_id / "eval" / "provenance.json").is_file()
        assert not (campaign / method_id / "eval" / "model.pt").exists()
        assert f"| `{method_id}` |" in status
        assert "done" in status.split(f"`{method_id}`")[1].split("\n")[0]
    blend = json.loads(
        (campaign / "blend_pathlens_three_hop" / "eval" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    rrf = json.loads(
        (campaign / "rrf_pathlens_three_hop" / "eval" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert blend["combine"]["alpha"] == 1.0
    assert rrf["combine"]["k"] == 60


def test_residual_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "residual_three_hop" / "eval" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["method"] == "residual_three_hop"
    assert payload["stage"] == "eval"
    assert "test" not in payload
    assert payload["training"]["selection"] == "validation_filtered_mrr"
    assert payload["training"]["best_epoch"] == 10
    assert len(payload["training"]["history"]) == 18
    assert (campaign / "residual_three_hop" / "eval" / "provenance.json").is_file()
    assert not (campaign / "residual_three_hop" / "eval" / "model.pt").exists()
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `residual_three_hop` |" in status
    assert "done" in status.split("`residual_three_hop`")[1].split("\n")[0]


def test_ranking_diagnostics_eval_card_is_filed() -> None:
    import json

    campaign = REPO_ROOT / "runs" / "biosnap-dti-v2"
    payload = json.loads(
        (campaign / "ranking_diagnostics" / "eval" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["method"] == "ranking_diagnostics"
    assert payload["stage"] == "eval"
    assert payload["diagnostic"] is True
    assert "test" not in payload
    assert set(payload["methods"]) >= {
        "degree",
        "resource_allocation",
        "three_hop",
        "pathlens_ranking",
    }
    pathlens = payload["methods"]["pathlens_ranking"]["all_positive"]
    three = payload["methods"]["three_hop"]["all_positive"]
    assert pathlens["ties"]["fraction_in_tie"] == 0.0
    assert three["ranking"]["average"]["mrr"] < three["ranking"]["strict_gt"]["mrr"]
    assert (campaign / "ranking_diagnostics" / "eval" / "provenance.json").is_file()
    assert (campaign / "ranking_diagnostics" / "eval" / "figures" / "mrr_tie_break.png").is_file()
    status = (REPO_ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "| `ranking_diagnostics` |" in status
    assert "done" in status.split("`ranking_diagnostics`")[1].split("\n")[0]


def test_delivery_source_pack_exists() -> None:
    docs = REPO_ROOT / "docs"
    for relative in (
        "delivery/README.md",
        "research/PAPER_CRITIQUE.md",
        "research/TECHNICAL_NARRATIVE.md",
        "research/MODEL_CATALOG.md",
        "research/EDA.md",
        "research/FREEZE.md",
    ):
        assert (docs / relative).is_file(), relative
    assert (REPO_ROOT / "runs" / "biosnap-dti-v2" / "figures" / "scoreboard.md").is_file()


def test_notebook_defaults_to_delivery_final() -> None:
    import json

    notebook = json.loads(
        (REPO_ROOT / "kaggle" / "pathlens_training.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][1]["source"])
    assert 'SUITE = "delivery"' in source
    assert 'STAGE = "final"' in source
    assert 'FINAL_TEST_TOKEN = "OPEN_SEALED_TEST_ONCE"' in source
    assert 'DEVICE = "cuda:0"' in source
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "write_delivery_figures" in joined
    assert "DELIVERY_METHODS" in joined
    assert "write_eda_figures" in joined
    assert "archive_run_dir" in joined
    checkout = "".join(notebook["cells"][2]["source"])
    assert 'repository_source = str(REPO / "src")' in checkout
    assert "sys.path.insert(0, repository_source)" in checkout
    assert "legacy2" in checkout
