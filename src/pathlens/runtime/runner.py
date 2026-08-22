from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from pathlens import REPO_ROOT
from pathlens.constants import (
    CAMPAIGN_ID,
    COMBINE_METHODS,
    FINAL_TEST_TOKEN,
    HEURISTIC_METHODS,
    STAGES,
    TRAINED_METHODS,
)
from pathlens.data.download import ensure_biosnap_tsv
from pathlens.data.prepare import prepare_biosnap_dataset
from pathlens.data.processed import ProcessedSplit, load_processed
from pathlens.evaluation.report import evaluate_score_matrix
from pathlens.graph.scoring import build_adjacency, score_method
from pathlens.runtime.config import load_method_config
from pathlens.runtime.device import describe_device, resolve_device


def run_stage(
    method_id: str,
    stage: str,
    *,
    device: str = "auto",
    repo_root: str | Path | None = None,
    raw_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    download: bool = True,
    strict_identity: bool = True,
    final_test_token: str = "",
    write_archive: bool = True,
    archive_path: str | Path | None = None,
    hop_scores: Any = None,
    pathlens_scores: Any = None,
    pathlens_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of {STAGES}")
    if stage == "final":
        raise PermissionError("final is disabled until freeze. The v2 test stays sealed.")
    if final_test_token and final_test_token != FINAL_TEST_TOKEN:
        raise PermissionError("Invalid FINAL_TEST_TOKEN")

    config = load_method_config(method_id)
    requested = device if device != "auto" else str(config.get("device", "auto"))
    resolved_device = resolve_device(requested)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    processed = ensure_processed(
        processed_dir or root / "data" / "processed" / CAMPAIGN_ID,
        raw_path or root / "data" / "raw" / "biosnap.tsv",
        download=download,
        strict_identity=strict_identity,
    )
    split = load_processed(processed, allow_test=False)
    run_dir = Path(output_root or root / "runs" / CAMPAIGN_ID) / method_id / stage
    run_dir.mkdir(parents=True, exist_ok=True)

    if method_id in COMBINE_METHODS:
        if stage == "train":
            raise ValueError(f"{method_id} is a frozen-score mix; use STAGE=smoke or STAGE=eval")
        payload = _run_combine(
            method_id,
            split,
            device=resolved_device,
            stage=stage,
            campaign_root=run_dir.parent.parent,
            repo_root=root,
            hop_scores=hop_scores,
            pathlens_scores=pathlens_scores,
            checkpoint=pathlens_checkpoint,
        )
    elif method_id in HEURISTIC_METHODS:
        if stage == "train":
            raise ValueError(f"{method_id} is a heuristic; use STAGE=smoke or STAGE=eval")
        payload = _run_heuristic(
            method_id,
            split,
            device=resolved_device,
            full=stage == "eval",
        )
    elif method_id in TRAINED_METHODS:
        payload = _run_trained(
            method_id,
            split,
            config,
            device=resolved_device,
            stage=stage,
            run_dir=run_dir,
        )
    else:
        raise NotImplementedError(
            f"{method_id} has no trainer in this slice. Implemented trainers: "
            f"{', '.join(sorted(TRAINED_METHODS))}."
        )
    result = {
        "method": method_id,
        "stage": stage,
        "campaign": CAMPAIGN_ID,
        "split_seed": split.seed,
        "device": resolved_device,
        "environment": describe_device(resolved_device),
        "processed": str(processed),
        **payload,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    archive = None
    if write_archive:
        archive = Path(archive_path) if archive_path is not None else _default_archive_path(root)
        archive_run_dir(run_dir, archive)
    result["output_dir"] = str(run_dir)
    result["archive"] = None if archive is None else str(archive)
    print(
        f"[pathlens] method={method_id} stage={stage} device={resolved_device} "
        f"mrr={payload['filtered_ranking']['mrr']:.4f} "
        f"hard_auprc={payload['classification']['hard']['auprc']:.4f}",
        flush=True,
    )
    return result


def ensure_processed(
    processed_dir: str | Path,
    raw_path: str | Path,
    *,
    download: bool,
    strict_identity: bool,
) -> Path:
    processed = Path(processed_dir)
    if (processed / "splits.npz").is_file() and (processed / "manifest.json").is_file():
        return processed
    source = Path(raw_path)
    if not source.is_file():
        if not download:
            raise FileNotFoundError(f"Raw BioSNAP TSV not found: {source}")
        print(f"[pathlens] Downloading BioSNAP to {source}", flush=True)
        ensure_biosnap_tsv(source)
    print(f"[pathlens] Preparing processed split at {processed}", flush=True)
    prepare_biosnap_dataset(source, processed, strict_identity=strict_identity)
    return processed


def _run_trained(
    method_id: str,
    split: ProcessedSplit,
    config: dict[str, Any],
    *,
    device: str,
    stage: str,
    run_dir: Path,
) -> dict[str, Any]:
    if method_id == "skipgnn":
        from pathlens.training.skipgnn import run_skipgnn

        return run_skipgnn(
            split,
            config,
            device=device,
            stage=stage,
            run_dir=run_dir,
        )
    if method_id == "gcn":
        from pathlens.training.gcn import run_gcn

        return run_gcn(
            split,
            config,
            device=device,
            stage=stage,
            run_dir=run_dir,
        )
    if method_id == "graphsage":
        from pathlens.training.graphsage import run_graphsage

        return run_graphsage(
            split,
            config,
            device=device,
            stage=stage,
            run_dir=run_dir,
        )
    if method_id == "residual_three_hop":
        from pathlens.training.residual import run_residual_three_hop

        return run_residual_three_hop(
            split,
            config,
            device=device,
            stage=stage,
            run_dir=run_dir,
        )
    raise NotImplementedError(f"{method_id} has no trainer in this slice")


def _run_combine(
    method_id: str,
    split: ProcessedSplit,
    *,
    device: str,
    stage: str,
    campaign_root: Path,
    repo_root: Path,
    hop_scores: Any,
    pathlens_scores: Any,
    checkpoint: str | Path | None,
) -> dict[str, Any]:
    from pathlens.runtime.fusion import BLEND_METHOD, RRF_METHOD, combine_pathlens_three_hop

    combined = combine_pathlens_three_hop(
        split,
        device=device,
        stage=stage,
        repo_root=repo_root,
        hop_scores=hop_scores,
        pathlens_scores=pathlens_scores,
        checkpoint=checkpoint,
    )
    environment = describe_device(device)
    written: dict[str, Path] = {}
    for key, sibling_id in (("blend", BLEND_METHOD), ("rrf", RRF_METHOD)):
        sibling_dir = campaign_root / sibling_id / stage
        sibling_dir.mkdir(parents=True, exist_ok=True)
        card = {
            "method": sibling_id,
            "stage": stage,
            "campaign": CAMPAIGN_ID,
            "split_seed": split.seed,
            "device": device,
            "environment": environment,
            **combined[key],
        }
        path = sibling_dir / "metrics.json"
        path.write_text(json.dumps(card, indent=2, default=_json_default), encoding="utf-8")
        written[key] = path
    sibling_copy = campaign_root / BLEND_METHOD / stage / f"{RRF_METHOD}.metrics.json"
    shutil.copy2(written["rrf"], sibling_copy)
    payload = dict(combined["blend" if method_id == BLEND_METHOD else "rrf"])
    payload["sibling"] = RRF_METHOD if method_id == BLEND_METHOD else BLEND_METHOD
    return payload


def _run_heuristic(
    method_id: str,
    split: ProcessedSplit,
    *,
    device: str,
    full: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    adjacency = build_adjacency(split.num_drugs, split.num_proteins, split.context, device)
    scores = score_method(method_id, adjacency)
    payload = evaluate_score_matrix(
        scores,
        split,
        include_curves=full,
        include_bootstrap=full,
        include_slices=full,
    )
    payload["score_seconds"] = time.perf_counter() - started
    return payload


def _default_archive_path(repo_root: Path) -> Path:
    working = Path("/kaggle/working")
    if working.is_dir():
        return working / "pathlens-stage-output.zip"
    return repo_root / "pathlens-stage-output.zip"


def archive_run_dir(run_dir: str | Path, destination: str | Path) -> Path:
    run_dir = Path(run_dir)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.make_archive(str(destination.with_suffix("")), "zip", root_dir=run_dir)
    return destination


def _json_default(value: object) -> object:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")
