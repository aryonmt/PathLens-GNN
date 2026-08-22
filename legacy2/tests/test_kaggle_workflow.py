from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pathlens_gnn.training.kaggle import (
    authorize_final_evaluation,
    restore_output_archive,
)
from pathlens_gnn.training.tuning_state import load_state, validate_state, write_state


def test_committed_notebook_is_safe_to_run_all() -> None:
    notebook = json.loads(Path("kaggle/pathlens_training.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")

    parameters = next(cell for cell in code_cells if cell["id"] == "parameters")
    parameters_source = "".join(parameters["source"])
    assert 'STAGE = "smoke"' in parameters_source
    assert "export" not in parameters_source
    assert "report" in parameters_source
    assert "REGISTERED_ARCHIVE" in parameters_source
    final_stage = next(cell for cell in code_cells if cell["id"] == "final-stage")
    final_source = "".join(final_stage["source"])
    assert 'if STAGE == "final"' in final_source
    assert "authorize_final_evaluation" in final_source
    assert "--confirm-sealed-test" in final_source
    assert "sys.executable" in final_source
    data_source = "".join(next(cell for cell in code_cells if cell["id"] == "data")["source"])
    assert "prepare-data" in data_source
    assert 'sys.executable' in data_source
    assert '"-m"' in data_source
    assert "pathlens_gnn" in data_source
    assert '"41"' in data_source
    assert "biosnap-dti-canonical-v2" in data_source
    confirmation = "".join(
        next(cell for cell in code_cells if cell["id"] == "confirmation-stage")["source"]
    )
    assert "pathlens_ranking.yaml" in confirmation
    assert "leaderboard.json" not in confirmation
    assert "sys.executable" in confirmation
    report = "".join(next(cell for cell in code_cells if cell["id"] == "report-stage")["source"])
    assert "write_validation_report" in report
    assert "--confirm-sealed-test" not in report
    assert "validation-report" in report


def test_notebook_bootstraps_current_kernel_imports() -> None:
    notebook = json.loads(Path("kaggle/pathlens_training.ipynb").read_text(encoding="utf-8"))
    checkout_cell = next(cell for cell in notebook["cells"] if cell["id"] == "checkout")
    checkout_source = "".join(checkout_cell["source"])

    assert 'subprocess.run([sys.executable, "-m", "pip"' in checkout_source
    assert 'repository_source = str(REPO / "src")' in checkout_source
    assert "sys.path.insert(0, repository_source)" in checkout_source


def test_notebook_streams_training_logs_from_current_kernel_python() -> None:
    notebook = json.loads(Path("kaggle/pathlens_training.ipynb").read_text(encoding="utf-8"))
    training_cell = next(cell for cell in notebook["cells"] if cell["id"] == "train-stage")
    training_source = "".join(training_cell["source"])

    assert training_source.count("sys.executable") == 3
    assert training_source.count('"-u"') == 3


def test_restore_output_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        restore_output_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_restore_output_archive_extracts_valid_files(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("tuning/run_state.json", "{}")

    output = tmp_path / "output"
    restore_output_archive(archive, output)
    assert (output / "tuning/run_state.json").read_text(encoding="utf-8") == "{}"


def test_restore_output_archive_accepts_dataset_directory_containing_zip(tmp_path: Path) -> None:
    dataset = tmp_path / "pathlens-stage-output-9"
    dataset.mkdir()
    archive = dataset / "pathlens-stage-output.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("confirmation/confirmation.json", '{"selected": {}}')

    output = tmp_path / "output"
    restore_output_archive(dataset, output)
    restored = (output / "confirmation/confirmation.json").read_text(encoding="utf-8")
    assert restored == '{"selected": {}}'


def test_restore_output_archive_copies_extracted_stage_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "extracted"
    (dataset / "confirmation").mkdir(parents=True)
    (dataset / "confirmation" / "confirmation.json").write_text("{}", encoding="utf-8")
    (dataset / "stage-confirmation.json").write_text("{}", encoding="utf-8")

    output = tmp_path / "output"
    restore_output_archive(dataset, output)
    assert (output / "confirmation/confirmation.json").read_text(encoding="utf-8") == "{}"
    assert (output / "stage-confirmation.json").read_text(encoding="utf-8") == "{}"


def test_restore_output_archive_rejects_empty_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "empty"
    dataset.mkdir()
    with pytest.raises(FileNotFoundError, match="directory"):
        restore_output_archive(dataset, tmp_path / "output")


def test_final_evaluation_requires_token_and_is_one_time(tmp_path: Path) -> None:
    output = tmp_path / "final-evaluation.json"
    with pytest.raises(PermissionError, match="explicit one-time token"):
        authorize_final_evaluation("", output)

    authorize_final_evaluation("OPEN_SEALED_TEST_ONCE", output)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to rerun"):
        authorize_final_evaluation("OPEN_SEALED_TEST_ONCE", output)


def test_tuning_state_round_trips_and_binds_registered_budget(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    write_state(
        state_path,
        search_space_sha256="registered-hash",
        max_trials=24,
        max_wall_seconds=345600,
        active_elapsed_seconds=16.5,
        completed_trials=3,
        status="paused",
    )
    state = load_state(state_path)
    assert state["active_elapsed_seconds"] == 16.5
    assert state["completed_trials"] == 3
    validate_state(state, "registered-hash", 24, 345600)

    with pytest.raises(ValueError, match="changed registered budget or search space"):
        validate_state(state, "different-hash", 24, 345600)
