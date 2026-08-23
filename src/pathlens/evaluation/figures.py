from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pathlens.evaluation.catalog import comparison_models, family_models

FIGURE_NAMES = (
    "pr_hard.png",
    "pr_uniform.png",
    "roc_hard.png",
    "hits_at_k.png",
    "mrr_and_hard_auprc.png",
    "degree_slices_hard.png",
)
DELIVERY_FIGURE_NAMES = (
    *FIGURE_NAMES,
    "pathlens_family.png",
    "bipartite_hops.png",
    "blend_alpha_sweep.png",
    "epoch_selection.png",
    "mrr_by_degree_tertile.png",
    "val_vs_test_mrr.png",
)

DISPLAY_NAMES = {
    "full_adaptive_ranking": "PathLens ranking",
    "pathlens_ranking": "PathLens ranking",
    "full_adaptive_fusion": "PathLens BCE",
    "pathlens_bce": "PathLens BCE",
    "normalized_three_hop": "3-hop",
    "three_hop": "3-hop",
    "degree_type_shortcut": "Degree",
    "degree": "Degree",
    "resource_allocation": "RA / AA",
    "binary_skipgnn": "SkipGNN",
    "skipgnn": "SkipGNN",
    "one_hop": "One-hop",
    "s1_s2_weighted": "S1+S2",
    "s1_s2": "S1+S2",
    "s1_s2_s3_fixed": "S1+S2+S3",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "blend_pathlens_three_hop": "Blend 3-hop+PathLens",
    "rrf_pathlens_three_hop": "RRF 3-hop+PathLens",
    "residual_three_hop": "Residual 3-hop",
    "gat": "GAT",
    "nbfnet": "NBFNet",
}

HITS_KEYS = ("hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "hits_at_20", "hits_at_50")
METRICS_STAGE_PRIORITY = ("eval", "imported", "smoke")


def load_validation_report(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        models = _load_runs_directory(path)
        if not models:
            raise FileNotFoundError(f"No metrics.json under {path} (eval, imported, or smoke)")
        return {"models": models, "split": "validation"}
    if path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            payload = archive.read("validation-report/validation-report.json")
        return cast(dict[str, Any], json.loads(payload))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_validation_figures(report: dict[str, Any], output: Path) -> list[Path]:
    plt = _pyplot()
    output.mkdir(parents=True, exist_ok=True)
    models = comparison_models(report["models"])
    if not models:
        raise FileNotFoundError("No comparison methods with metrics")
    written = [
        _pr_curve(plt, models, output / "pr_hard.png", bank="hard"),
        _pr_curve(plt, models, output / "pr_uniform.png", bank="uniform"),
        _roc_curve(plt, models, output / "roc_hard.png", bank="hard"),
        _hits_curve(plt, models, output / "hits_at_k.png"),
        _metric_bars(plt, models, output / "mrr_and_hard_auprc.png"),
        _degree_slices(plt, models, output / "degree_slices_hard.png"),
    ]
    family = family_models(report["models"])
    if family:
        written.append(_metric_bars(plt, family, output / "pathlens_family.png", family=True))
    return written


def copy_figure_files(paths: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _load_runs_directory(root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        payload = _load_preferred_metrics(method_dir)
        if payload is None or payload.get("diagnostic"):
            continue
        if "filtered_ranking" not in payload or "classification" not in payload:
            continue
        method_id = str(payload.get("method", method_dir.name))
        models[method_id] = {
            "filtered_ranking": payload["filtered_ranking"],
            "classification": payload["classification"],
        }
    return models


def _load_preferred_metrics(method_dir: Path) -> dict[str, Any] | None:
    for stage in METRICS_STAGE_PRIORITY:
        metrics_path = method_dir / stage / "metrics.json"
        if metrics_path.is_file():
            return cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    return None


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _label(method_id: str) -> str:
    return DISPLAY_NAMES.get(method_id, method_id)


def _pr_curve(plt: Any, models: dict[str, Any], path: Path, *, bank: str) -> Path:
    figure, axes = plt.subplots(figsize=(6.2, 4.4))
    for name, model in models.items():
        curves = model["classification"][bank].get("curves")
        if not curves:
            continue
        axes.plot(
            curves["recall"],
            curves["precision"],
            label=_label(name),
            linewidth=1.6,
        )
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(f"Validation PR ({bank} negatives)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.grid(True, alpha=0.3)
    _legend(axes)
    return _save(plt, figure, path)


def _roc_curve(plt: Any, models: dict[str, Any], path: Path, *, bank: str) -> Path:
    figure, axes = plt.subplots(figsize=(6.2, 4.4))
    for name, model in models.items():
        curves = model["classification"][bank].get("curves")
        if not curves or "fpr" not in curves:
            continue
        axes.plot(curves["fpr"], curves["tpr"], label=_label(name), linewidth=1.6)
    axes.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1, label="Chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(f"Validation ROC ({bank} negatives)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.grid(True, alpha=0.3)
    _legend(axes)
    return _save(plt, figure, path)


def _hits_curve(plt: Any, models: dict[str, Any], path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(6.2, 4.4))
    ks = np.asarray([1, 3, 5, 10, 20, 50], dtype=np.int64)
    for name, model in models.items():
        ranking = model["filtered_ranking"]
        values = [float(ranking.get(key, np.nan)) for key in HITS_KEYS]
        axes.plot(ks, values, marker="o", label=_label(name), linewidth=1.6)
    axes.set_xscale("log")
    axes.set_xticks(list(ks))
    axes.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes.set_xlabel("K")
    axes.set_ylabel("Hits@K")
    axes.set_title("Validation filtered Hits@K")
    axes.set_ylim(0, 1)
    axes.grid(True, alpha=0.3, which="both")
    _legend(axes)
    return _save(plt, figure, path)


def _metric_bars(
    plt: Any,
    models: dict[str, Any],
    path: Path,
    *,
    family: bool = False,
) -> Path:
    names = list(models)
    labels = [_label(name) for name in names]
    mrr = [models[name]["filtered_ranking"]["mrr"] for name in names]
    auprc = [models[name]["classification"]["hard"]["auprc"] for name in names]
    x = np.arange(len(names))
    width = 0.38
    width_inches = 8.6 if len(names) > 8 else 7.4
    figure, axes = plt.subplots(figsize=(width_inches, 4.6))
    axes.bar(x - width / 2, mrr, width, label="Filtered MRR")
    axes.bar(x + width / 2, auprc, width, label="Hard AUPRC")
    axes.set_xticks(x)
    axes.set_xticklabels(labels, rotation=35, ha="right")
    axes.set_ylim(0, 1)
    axes.set_ylabel("Score")
    if family:
        axes.set_title("PathLens ablation ladder (validation)")
    else:
        axes.set_title("Validation ranking vs hard classification")
    axes.grid(True, axis="y", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return _save(plt, figure, path)


def _degree_slices(plt: Any, models: dict[str, Any], path: Path) -> Path:
    names = [
        name
        for name, model in models.items()
        if "degree_slices" in model["classification"]["hard"]
    ]
    slices = ("low", "mid", "high")
    x = np.arange(len(slices))
    width = 0.8 / max(1, len(names))
    figure, axes = plt.subplots(figsize=(6.8, 4.4))
    for index, name in enumerate(names):
        values = [
            models[name]["classification"]["hard"]["degree_slices"]
            .get(slice_name, {})
            .get("auprc", np.nan)
            for slice_name in slices
        ]
        axes.bar(x + index * width, values, width, label=_label(name))
    axes.set_xticks(x + width * (len(names) - 1) / 2)
    axes.set_xticklabels(["Low degree", "Mid degree", "High degree"])
    axes.set_ylim(0, 1)
    axes.set_ylabel("Hard AUPRC")
    axes.set_title("Validation hard AUPRC by pair degree")
    axes.grid(True, axis="y", alpha=0.3)
    _legend(axes)
    return _save(plt, figure, path)


def write_delivery_figures(campaign: Path, output: Path) -> list[Path]:
    campaign = Path(campaign)
    output = Path(output)
    report = load_validation_report(campaign)
    written = write_validation_figures(report, output)
    plt = _pyplot()
    written.append(write_bipartite_hops(plt, output / "bipartite_hops.png"))
    blend_path = campaign / "blend_pathlens_three_hop" / "eval" / "metrics.json"
    if blend_path.is_file():
        payload = json.loads(blend_path.read_text(encoding="utf-8"))
        written.append(write_blend_sweep(plt, payload, output / "blend_alpha_sweep.png"))
    written.extend(write_epoch_selection(plt, campaign, output / "epoch_selection.png"))
    diagnostics = campaign / "ranking_diagnostics" / "eval" / "metrics.json"
    if diagnostics.is_file():
        payload = json.loads(diagnostics.read_text(encoding="utf-8"))
        written.append(
            write_mrr_by_degree(plt, payload, output / "mrr_by_degree_tertile.png")
        )
    test_models = _load_final_pairs(campaign)
    if test_models:
        written.append(write_val_vs_test(plt, test_models, output / "val_vs_test_mrr.png"))
    return written


def write_bipartite_hops(plt: Any, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7.2, 3.6))
    axes.set_xlim(-0.4, 6.4)
    axes.set_ylim(-1.35, 1.35)
    axes.axis("off")
    drugs = [(0.0, 0.7, "D"), (2.0, 0.7, "D′"), (4.0, 0.7, "D″")]
    proteins = [(1.0, -0.7, "P"), (3.0, -0.7, "P′")]
    for x, y, label in drugs:
        axes.scatter([x], [y], s=420, c="#1f4e79", zorder=3)
        axes.text(x, y, label, ha="center", va="center", color="white", fontsize=10)
    for x, y, label in proteins:
        axes.scatter([x], [y], s=420, c="#b35c00", zorder=3)
        axes.text(x, y, label, ha="center", va="center", color="white", fontsize=10)
    axes.plot([0.0, 1.0], [0.7, -0.7], color="0.2", lw=1.8)
    axes.plot([1.0, 2.0], [-0.7, 0.7], color="0.2", lw=1.8)
    axes.plot([2.0, 3.0], [0.7, -0.7], color="0.2", lw=1.8)
    axes.text(0.5, 0.12, "1-hop\nopposite type", ha="center", va="center", fontsize=8)
    axes.text(1.5, 0.12, "2-hop\nsame type", ha="center", va="center", fontsize=8)
    axes.text(2.5, 0.12, "3-hop\nopposite type", ha="center", va="center", fontsize=8)
    axes.set_title("On a bipartite DTI graph, even hops stay on the same part")
    return _save(plt, figure, path)


def write_blend_sweep(plt: Any, payload: dict[str, Any], path: Path) -> Path:
    sweep = payload.get("combine", {}).get("sweep", [])
    alphas = [row["alpha"] for row in sweep]
    mrrs = [row["mrr"] for row in sweep]
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.plot(alphas, mrrs, marker="o", linewidth=1.8)
    selected = payload.get("combine", {}).get("alpha")
    if selected is not None:
        axes.axvline(float(selected), color="0.4", linestyle="--", label=f"selected α={selected}")
        axes.legend(frameon=False)
    axes.set_xlabel("α (weight on z-scored 3-hop)")
    axes.set_ylabel("Validation filtered MRR")
    axes.set_title("Late-fusion sweep: mixing PathLens never beat α=1")
    axes.set_ylim(0, 0.55)
    axes.grid(True, alpha=0.3)
    return _save(plt, figure, path)


def write_epoch_selection(plt: Any, campaign: Path, path: Path) -> list[Path]:
    rows = [
        ("skipgnn", "val_auroc", "1:1 AUROC"),
        ("gcn", "val_auroc", "1:1 AUROC"),
        ("graphsage", "val_auroc", "1:1 AUROC"),
        ("residual_three_hop", "val_mrr", "filtered MRR"),
    ]
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    drawn = False
    for method_id, key, _ylabel in rows:
        metrics_path = campaign / method_id / "eval" / "metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = payload.get("training", {}).get("history", [])
        if not history or key not in history[0]:
            continue
        epochs = [row["epoch"] for row in history]
        values = [row[key] for row in history]
        axes.plot(epochs, values, marker="o", label=_label(method_id), linewidth=1.6)
        best = payload.get("training", {}).get("best_epoch")
        if best is not None:
            axes.axvline(int(best), linestyle=":", alpha=0.4)
        drawn = True
    if not drawn:
        plt.close(figure)
        return []
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Selection metric")
    axes.set_title("Checkpoint selection (best epoch marked)")
    axes.grid(True, alpha=0.3)
    _legend(axes)
    return [_save(plt, figure, path)]


def write_mrr_by_degree(plt: Any, payload: dict[str, Any], path: Path) -> Path:
    methods = payload.get("methods", {})
    wanted = ("resource_allocation", "three_hop", "pathlens_ranking", "degree")
    slices = ("low", "mid", "high")
    names = [name for name in wanted if name in methods]
    x = np.arange(len(slices))
    width = 0.8 / max(1, len(names))
    figure, axes = plt.subplots(figsize=(6.8, 4.4))
    for index, name in enumerate(names):
        tertiles = methods[name]["all_positive"]["mrr_by_drug_degree"]
        values = [tertiles[slice_name]["mrr"] for slice_name in slices]
        axes.bar(x + index * width, values, width, label=_label(name))
    axes.set_xticks(x + width * (len(names) - 1) / 2)
    axes.set_xticklabels(["Low degree", "Mid degree", "High degree"])
    axes.set_ylim(0, 1)
    axes.set_ylabel("Filtered MRR (strict / all_positive)")
    axes.set_title("Heuristics lead on the sparse tail; PathLens leads on hubs")
    axes.grid(True, axis="y", alpha=0.3)
    _legend(axes)
    return _save(plt, figure, path)


def write_val_vs_test(plt: Any, models: dict[str, dict[str, float]], path: Path) -> Path:
    models = comparison_models(models)
    names = list(models)
    labels = [_label(name) for name in names]
    val = [models[name]["validation"] for name in names]
    test = [models[name]["test"] for name in names]
    x = np.arange(len(names))
    width = 0.38
    figure, axes = plt.subplots(figsize=(8.6, 4.6))
    axes.bar(x - width / 2, val, width, label="Validation MRR")
    axes.bar(x + width / 2, test, width, label="Test MRR")
    axes.set_xticks(x)
    axes.set_xticklabels(labels, rotation=35, ha="right")
    axes.set_ylim(0, 1)
    axes.set_ylabel("Filtered MRR")
    axes.set_title("Paired validation vs one-shot test (same STAGE=final run)")
    axes.grid(True, axis="y", alpha=0.3)
    axes.legend()
    return _save(plt, figure, path)


def _load_final_pairs(campaign: Path) -> dict[str, dict[str, float]]:
    pairs: dict[str, dict[str, float]] = {}
    for method_dir in sorted(path for path in campaign.iterdir() if path.is_dir()):
        metrics_path = method_dir / "final" / "metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = payload.get("test", {})
        if "filtered_ranking" not in payload or "filtered_ranking" not in test:
            continue
        method_id = str(payload.get("method", method_dir.name))
        pairs[method_id] = {
            "validation": float(payload["filtered_ranking"]["mrr"]),
            "test": float(test["filtered_ranking"]["mrr"]),
            "filtered_ranking": payload["filtered_ranking"],
            "classification": payload.get("classification", {}),
        }
    return pairs


def _legend(axes: Any) -> None:
    axes.legend(frameon=False, fontsize=8)


def _save(plt: Any, figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def synthetic_report() -> dict[str, Any]:
    recall = np.linspace(0, 1, 5)
    return {
        "models": {
            "three_hop": _toy_model(recall, lift=0.2, mrr=0.45),
            "pathlens_ranking": _toy_model(recall, lift=0.1, mrr=0.37),
        }
    }


def _toy_model(recall: NDArray[np.float64], *, lift: float, mrr: float) -> dict[str, Any]:
    precision = np.clip(1.0 - 0.5 * recall + lift, 0.05, 1.0)
    fpr = recall
    tpr = np.clip(recall + lift, 0, 1)
    return {
        "filtered_ranking": {
            "mrr": mrr,
            "hits_at_1": mrr * 0.8,
            "hits_at_3": mrr,
            "hits_at_5": min(1.0, mrr + 0.1),
            "hits_at_10": min(1.0, mrr + 0.2),
            "hits_at_20": min(1.0, mrr + 0.3),
            "hits_at_50": min(1.0, mrr + 0.4),
        },
        "classification": {
            "hard": {
                "auprc": float(precision.mean()),
                "curves": {
                    "recall": recall.tolist(),
                    "precision": precision.tolist(),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                },
                "degree_slices": {
                    "low": {"auprc": 0.6},
                    "mid": {"auprc": 0.7},
                    "high": {"auprc": 0.9},
                },
            },
            "uniform": {
                "auprc": float(precision.mean() + 0.05),
                "curves": {
                    "recall": recall.tolist(),
                    "precision": np.clip(precision + 0.05, 0, 1).tolist(),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                },
            },
        },
    }
