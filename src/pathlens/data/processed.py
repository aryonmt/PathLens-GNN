from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

VALIDATION_KEYS = (
    "all_positive",
    "context",
    "train_positive",
    "train_uniform",
    "validation_positive",
    "validation_uniform",
    "validation_hard",
)
TEST_KEYS = ("test_positive", "test_uniform", "test_hard")


class SealedTestError(PermissionError):
    """Raised when code tries to read the sealed test split."""


@dataclass(frozen=True, slots=True)
class ProcessedSplit:
    num_drugs: int
    num_proteins: int
    seed: int
    manifest: dict[str, Any]
    all_positive: NDArray[np.int64]
    context: NDArray[np.int64]
    train_positive: NDArray[np.int64]
    train_uniform: NDArray[np.int64]
    validation_positive: NDArray[np.int64]
    validation_uniform: NDArray[np.int64]
    validation_hard: NDArray[np.int64]
    _test: dict[str, NDArray[np.int64]] | None

    def test_array(self, name: str) -> NDArray[np.int64]:
        if self._test is None:
            raise SealedTestError("The v2 test set is sealed until freeze + STAGE=final")
        if name not in self._test:
            raise KeyError(name)
        return self._test[name]


def load_processed(processed_dir: str | Path, *, allow_test: bool = False) -> ProcessedSplit:
    processed = Path(processed_dir)
    entities = json.loads((processed / "entities.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    wanted = VALIDATION_KEYS + (TEST_KEYS if allow_test else ())
    with np.load(processed / "splits.npz") as arrays:
        loaded = {name: np.asarray(arrays[name], dtype=np.int64) for name in wanted}
    test_arrays = (
        {name: loaded[name] for name in TEST_KEYS} if allow_test else None
    )
    return ProcessedSplit(
        num_drugs=len(entities["drugs"]),
        num_proteins=len(entities["proteins"]),
        seed=int(manifest["seed"]),
        manifest=manifest,
        all_positive=loaded["all_positive"],
        context=loaded["context"],
        train_positive=loaded["train_positive"],
        train_uniform=loaded["train_uniform"],
        validation_positive=loaded["validation_positive"],
        validation_uniform=loaded["validation_uniform"],
        validation_hard=loaded["validation_hard"],
        _test=test_arrays,
    )
