from __future__ import annotations

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray


def resolve_device(requested: str) -> str:
    name = requested.strip() or "auto"
    if name == "cpu":
        return "cpu"
    torch = _optional_torch()
    if name in {"auto", "cuda"}:
        if torch is not None and bool(torch.cuda.is_available()):
            return "cuda:0"
        if name == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        return "cpu"
    if name.startswith("cuda"):
        if torch is None:
            raise RuntimeError(
                "CUDA was requested but PyTorch is not installed. "
                "On the laptop use --device cpu. On Kaggle, torch is preinstalled."
            )
        if not bool(torch.cuda.is_available()):
            raise RuntimeError(f"{name} was requested but CUDA is not available")
        return name
    raise ValueError(f"Unsupported device: {requested!r}")


def is_torch(value: object) -> bool:
    return type(value).__module__.startswith("torch")


def as_numpy(value: Any) -> NDArray[np.floating]:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return np.asarray(value.detach().cpu().numpy())


def describe_device(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": device,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": None,
        "cuda_name": None,
    }
    torch = _optional_torch()
    if torch is None:
        return payload
    payload["torch"] = str(torch.__version__)
    if device.startswith("cuda") and bool(torch.cuda.is_available()):
        index = int(device.split(":")[1]) if ":" in device else 0
        payload["cuda_name"] = str(torch.cuda.get_device_name(index))
    return payload


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch
