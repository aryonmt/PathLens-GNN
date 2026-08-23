from __future__ import annotations

import json
from pathlib import Path

from pathlens.evaluation.scoreboard import collect_scoreboard, write_scoreboard_tables


def _write(path: Path, method_id: str, mrr: float, auprc: float, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method_id,
                "stage": stage,
                "filtered_ranking": {
                    "mrr": mrr,
                    "hits_at_10": mrr,
                    "ndcg_at_10": mrr,
                    "ndcg_at_50": mrr,
                },
                "classification": {"hard": {"auprc": auprc}},
            }
        ),
        encoding="utf-8",
    )


def test_scoreboard_separates_nested_pathlens(tmp_path: Path) -> None:
    root = tmp_path / "biosnap-dti-v2"
    _write(root / "one_hop" / "imported" / "metrics.json", "one_hop", 0.10, 0.79, "imported")
    _write(
        root / "pathlens_ranking" / "imported" / "metrics.json",
        "pathlens_ranking",
        0.38,
        0.87,
        "imported",
    )
    _write(root / "three_hop" / "eval" / "metrics.json", "three_hop", 0.45, 0.85, "eval")
    markdown = write_scoreboard_tables(root, tmp_path / "out").read_text(encoding="utf-8")
    rows = collect_scoreboard(root)
    assert [row["method"] for row in rows] == ["three_hop", "pathlens_ranking", "one_hop"]
    comparison, family = markdown.split("## PathLens family", 1)
    assert "| `three_hop` |" in comparison
    assert "| `one_hop` |" not in comparison
    assert "| `one_hop` |" in family
    assert (tmp_path / "out" / "scoreboard.json").is_file()
