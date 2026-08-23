from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathlens.evaluation.catalog import PATHLENS_FAMILY, PATHLENS_NESTED
from pathlens.evaluation.figures import _load_preferred_metrics


def collect_scoreboard(campaign: str | Path) -> list[dict[str, Any]]:
    campaign = Path(campaign)
    rows: list[dict[str, Any]] = []
    for method_dir in sorted(path for path in campaign.iterdir() if path.is_dir()):
        payload = _load_preferred_metrics(method_dir)
        if payload is None or payload.get("diagnostic"):
            continue
        ranking = payload.get("filtered_ranking")
        classification = payload.get("classification", {})
        if not ranking or "hard" not in classification:
            continue
        method_id = str(payload.get("method", method_dir.name))
        rows.append(
            {
                "method": method_id,
                "group": _group(method_id),
                "hard_auprc": float(classification["hard"]["auprc"]),
                "mrr": float(ranking["mrr"]),
                "hits_at_10": float(ranking.get("hits_at_10", float("nan"))),
                "ndcg_at_10": float(ranking.get("ndcg_at_10", float("nan"))),
                "ndcg_at_50": float(ranking.get("ndcg_at_50", float("nan"))),
                "stage": str(payload.get("stage", "")),
            }
        )
    rows.sort(key=lambda row: (-row["mrr"], -row["hard_auprc"], row["method"]))
    return rows


def write_scoreboard_tables(campaign: str | Path, output: str | Path) -> Path:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_scoreboard(campaign)
    payload = {
        "split": "validation",
        "comparison": [row for row in rows if row["method"] not in PATHLENS_NESTED],
        "pathlens_family": [row for row in rows if row["method"] in PATHLENS_FAMILY],
        "all": rows,
    }
    json_path = output_dir / "scoreboard.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _markdown(payload)
    markdown_path = output_dir / "scoreboard.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path


def _group(method_id: str) -> str:
    if method_id in PATHLENS_FAMILY:
        return "pathlens_family"
    if method_id in {"degree", "resource_allocation", "three_hop"}:
        return "heuristic"
    if method_id in {"skipgnn", "gcn", "graphsage"}:
        return "gnn_baseline"
    if method_id in {"blend_pathlens_three_hop", "rrf_pathlens_three_hop"}:
        return "mix"
    if method_id == "residual_three_hop":
        return "residual"
    return "other"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Validation scoreboard (`biosnap-dti-v2`)",
        "",
        "Nested PathLens hops (`one_hop`, `s1_s2`, `s1_s2_s3_fixed`) are in the",
        "family table only. Comparison includes PathLens BCE and ranking heads.",
        "",
        "## Comparison",
        "",
        _table(payload["comparison"]),
        "",
        "## PathLens family (one architecture, ablated)",
        "",
        _table(payload["pathlens_family"]),
        "",
    ]
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]]) -> str:
    header = "| Method | Group | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Stage |"
    sep = "|---|---|---:|---:|---:|---:|---:|---|"
    body = [
        "| `{method}` | {group} | {hard_auprc:.3f} | {mrr:.3f} | {hits_at_10:.3f} | "
        "{ndcg_at_10:.3f} | {ndcg_at_50:.3f} | {stage} |".format(**row)
        for row in rows
    ]
    return "\n".join([header, sep, *body])
