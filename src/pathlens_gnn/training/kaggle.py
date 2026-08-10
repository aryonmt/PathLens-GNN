from __future__ import annotations

import zipfile
from pathlib import Path

FINAL_TEST_TOKEN = "OPEN_SEALED_TEST_ONCE"


def restore_output_archive(archive_path: str | Path, destination: str | Path) -> None:
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def authorize_final_evaluation(token: str, output_path: str | Path) -> None:
    if token != FINAL_TEST_TOKEN:
        raise PermissionError("Final evaluation requires the explicit one-time token")
    if Path(output_path).exists():
        raise FileExistsError("Final evaluation already exists; refusing to rerun the sealed test")
