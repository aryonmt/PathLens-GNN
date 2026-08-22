from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pathlens import REPO_ROOT
from pathlens.constants import (
    CAMPAIGN_ID,
    DEFAULT_SPLIT_SEED,
    IMPORT_METHODS,
    V2_FREEZE_CHECKPOINT_SHA256,
)
from pathlens.evaluation.ranking import ranking_report_from_ranks

DEFAULT_V2_REPORT_ZIP = (
    REPO_ROOT
    / "legacy2"
    / "outputs-kaggle"
    / "kaggle-stages"
    / "pathlens-stage-output-v2-report.zip"
)
SKIP_REPORT_MODELS = frozenset(
    {
        "degree_type_shortcut",
        "normalized_three_hop",
        "binary_skipgnn",
    }
)


@dataclass(frozen=True, slots=True)
class ImportSpec:
    method_id: str
    report_name: str
    checkpoint_member: str


IMPORT_SPECS: tuple[ImportSpec, ...] = (
    ImportSpec("one_hop", "one_hop", "registered/one_hop/checkpoint.pt"),
    ImportSpec("s1_s2", "s1_s2_weighted", "registered/s1_s2_weighted/checkpoint.pt"),
    ImportSpec(
        "s1_s2_s3_fixed",
        "s1_s2_s3_fixed",
        "registered/s1_s2_s3_fixed/checkpoint.pt",
    ),
    ImportSpec(
        "pathlens_bce",
        "full_adaptive_fusion",
        "registered/full_adaptive_fusion/checkpoint.pt",
    ),
    ImportSpec(
        "pathlens_ranking",
        "full_adaptive_ranking",
        "confirmation/candidate-0/seed-13/checkpoint.pt",
    ),
)


def import_v2_report(
    archive_path: str | Path,
    output_root: str | Path,
    *,
    copy_weights: bool = True,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Campaign v2 report ZIP not found: {archive_path}. "
            "Place pathlens-stage-output-v2-report.zip under legacy2/outputs-kaggle/kaggle-stages/."
        )
    output_root = Path(output_root)
    archive_sha = _sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        _assert_no_test_payload(archive)
        report = json.loads(_read_member(archive, "validation-report/validation-report.json"))
        freeze = json.loads(_read_member(archive, "model-freeze.json"))
        if report.get("split") != "validation":
            raise ValueError("Refusing to import a non-validation report")
        _assert_validation_payload(report)
        if {spec.method_id for spec in IMPORT_SPECS} != IMPORT_METHODS:
            raise RuntimeError("IMPORT_SPECS is out of sync with IMPORT_METHODS")
        models = report["models"]
        extra = set(models) - {spec.report_name for spec in IMPORT_SPECS} - SKIP_REPORT_MODELS
        if extra:
            raise ValueError(f"Unexpected report models: {sorted(extra)}")
        missing = {spec.report_name for spec in IMPORT_SPECS} - set(models)
        if missing:
            raise ValueError(f"Report is missing required models: {sorted(missing)}")
        if freeze.get("checkpoint_sha256") != V2_FREEZE_CHECKPOINT_SHA256:
            raise ValueError("Freeze record SHA-256 does not match the locked seed-13 checkpoint")
        if report.get("checkpoint_sha256") != V2_FREEZE_CHECKPOINT_SHA256:
            raise ValueError("Validation report checkpoint SHA-256 does not match freeze")
        ranks_by_legacy = _load_rank_arrays(archive)

        catalog: dict[str, Any] = {
            "campaign": CAMPAIGN_ID,
            "split_seed": DEFAULT_SPLIT_SEED,
            "source_archive": archive_path.name,
            "source_archive_sha256": archive_sha,
            "split": "validation",
            "test_sealed": True,
            "methods": {},
        }
        for spec in IMPORT_SPECS:
            catalog["methods"][spec.method_id] = _write_method(
                archive,
                spec,
                models[spec.report_name],
                freeze=freeze,
                output_root=output_root,
                archive_name=archive_path.name,
                archive_sha=archive_sha,
                copy_weights=copy_weights,
                ranks=ranks_by_legacy[spec.report_name],
            )
    catalog_path = output_root / "imports" / "manifest.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def _write_method(
    archive: zipfile.ZipFile,
    spec: ImportSpec,
    model: dict[str, Any],
    *,
    freeze: dict[str, Any],
    output_root: Path,
    archive_name: str,
    archive_sha: str,
    copy_weights: bool,
    ranks: np.ndarray,
) -> dict[str, Any]:
    weights = _read_member(archive, spec.checkpoint_member)
    checkpoint_sha = hashlib.sha256(weights).hexdigest()
    if spec.method_id == "pathlens_ranking" and checkpoint_sha != V2_FREEZE_CHECKPOINT_SHA256:
        raise ValueError(
            f"{spec.method_id} checkpoint SHA-256 is {checkpoint_sha}, "
            f"expected {V2_FREEZE_CHECKPOINT_SHA256}"
        )
    run_dir = output_root / spec.method_id / "imported"
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking = _ranking_with_ndcg(model["filtered_ranking"], ranks)
    classification = _normalize_classification(model["classification"])
    payload: dict[str, Any] = {
        "method": spec.method_id,
        "stage": "imported",
        "campaign": CAMPAIGN_ID,
        "split_seed": DEFAULT_SPLIT_SEED,
        "legacy_name": spec.report_name,
        "filtered_ranking": ranking,
        "classification": classification,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    provenance = {
        "method": spec.method_id,
        "campaign": CAMPAIGN_ID,
        "split_seed": DEFAULT_SPLIT_SEED,
        "retrain": False,
        "source_archive": archive_name,
        "source_archive_sha256": archive_sha,
        "legacy_report_name": spec.report_name,
        "checkpoint_member": spec.checkpoint_member,
        "checkpoint_sha256": checkpoint_sha,
        "freeze_record": freeze if spec.method_id == "pathlens_ranking" else None,
        "confirmation": (
            _confirmation_summary(archive) if spec.method_id == "pathlens_ranking" else None
        ),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if copy_weights:
        (run_dir / "checkpoint.pt").write_bytes(weights)
    try:
        relative_dir = run_dir.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        relative_dir = run_dir.as_posix()
    hard = classification["hard"]
    if not isinstance(hard, dict):
        raise TypeError("hard classification block must be a mapping")
    return {
        "legacy_name": spec.report_name,
        "checkpoint_sha256": checkpoint_sha,
        "hard_auprc": hard["auprc"],
        "mrr": ranking["mrr"],
        "hits_at_10": ranking["hits_at_10"],
        "ndcg_at_10": ranking["ndcg_at_10"],
        "output_dir": relative_dir,
    }


def _ranking_with_ndcg(stored: dict[str, Any], ranks: np.ndarray) -> dict[str, Any]:
    completed = ranking_report_from_ranks(np.asarray(ranks, dtype=np.int64)).to_dict()
    if "mrr" in stored and not np.isclose(float(stored["mrr"]), float(completed["mrr"])):
        raise ValueError(
            f"Stored MRR {stored['mrr']} does not match ranks-derived MRR {completed['mrr']}"
        )
    merged = dict(stored)
    merged.update(completed)
    return merged


def _load_rank_arrays(archive: zipfile.ZipFile) -> dict[str, np.ndarray]:
    payload = _read_member(archive, "validation-report/validation-report.npz")
    with np.load(io.BytesIO(payload)) as arrays:
        missing = [
            spec.report_name
            for spec in IMPORT_SPECS
            if f"{spec.report_name}__ranks" not in arrays.files
        ]
        if missing:
            raise ValueError(f"Report npz is missing ranks for: {missing}")
        return {
            spec.report_name: np.asarray(arrays[f"{spec.report_name}__ranks"], dtype=np.int64)
            for spec in IMPORT_SPECS
        }


def _normalize_classification(classification: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for bank, payload in classification.items():
        if not isinstance(payload, dict):
            normalized[bank] = payload
            continue
        item = dict(payload)
        if "auprc_bootstrap_95_ci" in item and "auprc_ci" not in item:
            item["auprc_ci"] = item["auprc_bootstrap_95_ci"]
        normalized[bank] = item
    return normalized


def _confirmation_summary(archive: zipfile.ZipFile) -> dict[str, Any] | None:
    try:
        confirmation = json.loads(_read_member(archive, "confirmation/confirmation.json"))
    except KeyError:
        return None
    selected = confirmation.get("selected", {})
    return {
        "seeds": confirmation.get("seeds"),
        "mean_validation_filtered_mrr": selected.get("mean_validation_filtered_mrr"),
        "sample_sd_validation_filtered_mrr": selected.get("sample_sd_validation_filtered_mrr"),
        "mean_validation_hard_auprc": selected.get("mean_validation_hard_auprc"),
        "sample_sd_validation_hard_auprc": selected.get("sample_sd_validation_hard_auprc"),
    }


def _assert_no_test_payload(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    leaked = [
        name
        for name in names
        if "final-evaluation" in name or name.endswith("final-artifacts.npz")
    ]
    if leaked:
        raise ValueError(f"Archive looks like a test dump: {leaked}")


def _assert_validation_payload(report: dict[str, Any]) -> None:
    blob = json.dumps(report)
    for marker in ("test_positive", "test_uniform", "test_hard"):
        if marker in blob:
            raise ValueError(f"Validation report must not include {marker}")


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return archive.read(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
