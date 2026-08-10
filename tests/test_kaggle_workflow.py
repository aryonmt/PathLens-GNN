from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pathlens_gnn.training.kaggle import (
    authorize_final_evaluation,
    restore_output_archive,
)
from scripts.run_tuning import _load_state, _validate_state, _write_state


def test_committed_notebook_is_safe_to_run_all() -> None:
    notebook = json.loads(Path("kaggle/pathlens_training.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")

    parameters = next(cell for cell in code_cells if cell["id"] == "parameters")
    assert 'STAGE = "smoke"' in "".join(parameters["source"])
    final_stage = next(cell for cell in code_cells if cell["id"] == "final-stage")
    final_source = "".join(final_stage["source"])
    assert 'if STAGE == "final"' in final_source
    assert "authorize_final_evaluation" in final_source
    assert "--confirm-sealed-test" in final_source


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
    _write_state(
        state_path,
        search_space_sha256="registered-hash",
        max_trials=24,
        max_wall_seconds=345600,
        active_elapsed_seconds=16.5,
        completed_trials=3,
        status="paused",
    )
    state = _load_state(state_path)
    assert state["active_elapsed_seconds"] == 16.5
    assert state["completed_trials"] == 3
    _validate_state(state, "registered-hash", 24, 345600)

    with pytest.raises(SystemExit, match="changed registered budget or search space"):
        _validate_state(state, "different-hash", 24, 345600)
