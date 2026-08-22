from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pathlens import REPO_ROOT
from pathlens.constants import V2_FREEZE_CHECKPOINT_SHA256
from pathlens.data.processed import ProcessedSplit
from pathlens.graph.scoring import build_adjacency, score_three_hop
from pathlens.importing.campaign_v2 import DEFAULT_V2_REPORT_ZIP, import_v2_report
from pathlens.runtime.device import as_numpy


def resolve_pathlens_checkpoint(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    imported = root / "runs" / "biosnap-dti-v2" / "pathlens_ranking" / "imported" / "checkpoint.pt"
    if imported.is_file():
        _assert_freeze_sha(imported)
        return imported
    archive = find_v2_report(root)
    if archive is None:
        raise FileNotFoundError(
            "Frozen PathLens checkpoint not found at "
            "runs/biosnap-dti-v2/pathlens_ranking/imported/checkpoint.pt. "
            "Clone research/ranking-loss or run import-v2 from the v2 report ZIP."
        )
    import_v2_report(archive, root / "runs" / "biosnap-dti-v2", copy_weights=True)
    if not imported.is_file():
        raise FileNotFoundError(f"import-v2 did not write {imported}")
    _assert_freeze_sha(imported)
    return imported


def find_v2_report(repo_root: str | Path | None = None) -> Path | None:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    default = Path(DEFAULT_V2_REPORT_ZIP)
    if not default.is_absolute():
        default = root / default
    if default.is_file():
        return default
    stages = root / "legacy2" / "outputs-kaggle" / "kaggle-stages"
    local = stages / "pathlens-stage-output-v2-report.zip"
    if local.is_file():
        return local
    incoming = Path("/kaggle/input")
    if incoming.is_dir():
        hits = sorted(incoming.rglob("pathlens-stage-output-v2-report.zip"))
        if hits:
            return hits[0]
    return None


def score_frozen_pathlens(
    split: ProcessedSplit,
    *,
    device: str,
    checkpoint: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> NDArray[np.float64]:
    torch = _require_torch()
    path = Path(checkpoint) if checkpoint is not None else resolve_pathlens_checkpoint(repo_root)
    _ensure_legacy2_path(repo_root)
    from pathlens_gnn.graph.index import BipartiteIndex
    from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN
    from pathlens_gnn.training.scoring import DevicePairFeatures, score_all_proteins

    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw_model = dict(payload["training_config"]["model"])
    if "enabled_channels" in raw_model:
        raw_model["enabled_channels"] = tuple(raw_model["enabled_channels"])
    config = PathLensConfig(**raw_model)
    index = BipartiteIndex(split.num_drugs, split.num_proteins, split.context)
    drug_mass = np.asarray(
        [index.projection_mass_drug(drug) for drug in range(split.num_drugs)],
        dtype=np.float32,
    )
    protein_mass = np.asarray(
        [index.projection_mass_protein(protein) for protein in range(split.num_proteins)],
        dtype=np.float32,
    )
    adjacency = build_adjacency(split.num_drugs, split.num_proteins, split.context, device)
    bridge = as_numpy(score_three_hop(adjacency)).astype(np.float32, copy=False)
    resolved = torch.device(device)
    features = DevicePairFeatures(
        drug_mass=torch.as_tensor(drug_mass, device=resolved),
        protein_mass=torch.as_tensor(protein_mass, device=resolved),
        bridge=torch.as_tensor(bridge, device=resolved),
    )
    edge_index = torch.as_tensor(split.context.T, dtype=torch.long, device=resolved)
    model = PathLensGNN(split.num_drugs, split.num_proteins, edge_index, config).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    drugs = torch.arange(split.num_drugs, device=resolved)
    with torch.inference_mode():
        embeddings = model.encode()
        scores = score_all_proteins(model, embeddings, features, drugs)
    return as_numpy(scores).astype(np.float64, copy=False)


def _assert_freeze_sha(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != V2_FREEZE_CHECKPOINT_SHA256:
        raise ValueError(
            f"PathLens checkpoint SHA-256 is {digest}, expected {V2_FREEZE_CHECKPOINT_SHA256}"
        )


def _ensure_legacy2_path(repo_root: str | Path | None) -> None:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    source = str(root / "legacy2" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Scoring the frozen PathLens checkpoint requires PyTorch. "
            "Run this on Kaggle GPU, not on the laptop."
        ) from error
    return torch
