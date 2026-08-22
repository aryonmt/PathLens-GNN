from __future__ import annotations

from typer.testing import CliRunner

from pathlens_gnn.cli import app

runner = CliRunner()


def test_prepare_data_is_a_subcommand() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "prepare-data" in help_result.stdout
    assert "version" in help_result.stdout

    command_help = runner.invoke(app, ["prepare-data", "--help"])
    assert command_help.exit_code == 0
    assert "--source" in command_help.stdout
    assert "Got unexpected extra argument" not in command_help.stdout
