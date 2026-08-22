from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

FINAL_TEST_TOKEN = "OPEN_SEALED_TEST_ONCE"


def restore_output_archive(archive_path: str | Path, destination: str | Path) -> None:
    source = _resolve_resume_source(Path(archive_path))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] Restoring outputs from {source}", flush=True)
    if source.is_dir():
        _copy_tree(source, destination)
        return
    destination_root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def _resolve_resume_source(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"RESUME_ARCHIVE does not exist: {path}")
    if _looks_like_extracted_output(path):
        return path
    archives = sorted(item for item in path.rglob("*.zip") if item.is_file())
    preferred = [
        item
        for item in archives
        if item.name.startswith("pathlens-stage-output") and item.suffix == ".zip"
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(archives) == 1:
        return archives[0]
    found = "\n".join(f"  {item}" for item in archives) or "  (no zip files)"
    raise FileNotFoundError(
        f"{path} is a directory. Set RESUME_ARCHIVE to the ZIP file inside the "
        f"Kaggle input, for example {path / 'pathlens-stage-output.zip'}. Found:\n{found}"
    )


def _looks_like_extracted_output(path: Path) -> bool:
    if (path / "confirmation" / "confirmation.json").exists():
        return True
    return any(path.glob("stage-*.json"))


def _copy_tree(source: Path, destination: Path) -> None:
    source_root = source.resolve()
    destination_root = destination.resolve()
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        resolved = target.resolve()
        if resolved != destination_root and destination_root not in resolved.parents:
            raise ValueError(f"Unsafe archive member: {relative}")
        if source_root in resolved.parents or resolved == source_root:
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def authorize_final_evaluation(token: str, output_path: str | Path) -> None:
    if token != FINAL_TEST_TOKEN:
        raise PermissionError("Final evaluation requires the explicit one-time token")
    if Path(output_path).exists():
        raise FileExistsError("Final evaluation already exists; refusing to rerun the sealed test")
