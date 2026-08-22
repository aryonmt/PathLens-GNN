from __future__ import annotations

from typing import Any


class ResidualThreeHop:
    """Learned pair residual added to a frozen three-hop score."""

    def __init__(
        self,
        num_drugs: int,
        num_proteins: int,
        dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.drug = nn.Embedding(num_drugs, dim)
        self.protein = nn.Embedding(num_proteins, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.drug.weight, std=0.02)
        nn.init.normal_(self.protein.weight, std=0.02)

    def to(self, device: str) -> ResidualThreeHop:
        self.drug = self.drug.to(device)
        self.protein = self.protein.to(device)
        self.mlp = self.mlp.to(device)
        return self

    def train(self, mode: bool = True) -> ResidualThreeHop:
        self.drug.train(mode)
        self.protein.train(mode)
        self.mlp.train(mode)
        return self

    def eval(self) -> ResidualThreeHop:
        return self.train(False)

    def parameters(self) -> list[Any]:
        return [*self.drug.parameters(), *self.protein.parameters(), *self.mlp.parameters()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "drug": self.drug.state_dict(),
            "protein": self.protein.state_dict(),
            "mlp": self.mlp.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.drug.load_state_dict(state["drug"])
        self.protein.load_state_dict(state["protein"])
        self.mlp.load_state_dict(state["mlp"])

    def residual(self, drugs: Any, proteins: Any) -> Any:
        drug = self.drug(drugs)
        protein = self.protein(proteins)
        return self.mlp(torch_cat(drug, protein)).squeeze(-1)

    def forward(self, drugs: Any, proteins: Any, hop_scores: Any) -> Any:
        return hop_scores + self.residual(drugs, proteins)

    def score_matrix(self, hop_scores: Any, *, chunk_size: int = 256) -> Any:
        torch = _require_torch()
        hop = hop_scores
        if not hasattr(hop, "device"):
            hop = torch.as_tensor(hop, dtype=torch.float32, device=self.drug.weight.device)
        rows = []
        proteins = torch.arange(self.num_proteins, device=hop.device)
        for start in range(0, self.num_drugs, chunk_size):
            stop = min(start + chunk_size, self.num_drugs)
            drugs = torch.arange(start, stop, device=hop.device)
            residual = self._residual_block(drugs, proteins)
            rows.append(hop[start:stop] + residual)
        return torch.cat(rows, dim=0)

    def _residual_block(self, drugs: Any, proteins: Any) -> Any:
        torch = _require_torch()
        drug = self.drug(drugs)
        protein = self.protein(proteins)
        pair = torch.cat(
            (
                drug[:, None, :].expand(-1, self.num_proteins, -1),
                protein[None, :, :].expand(int(drugs.shape[0]), -1, -1),
                drug[:, None, :] * protein[None, :, :],
            ),
            dim=-1,
        )
        return self.mlp(pair).squeeze(-1)


def torch_cat(drug: Any, protein: Any) -> Any:
    torch = _require_torch()
    return torch.cat((drug, protein, drug * protein), dim=-1)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "residual_three_hop requires PyTorch. Kaggle images already have it."
        ) from error
    return torch
