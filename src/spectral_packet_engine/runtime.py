from __future__ import annotations

from dataclasses import dataclass
import platform

import torch


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    requested_device: str
    system: str
    machine: str
    torch_version: str
    device: torch.device
    accelerator: str
    backend: str
    available_backends: tuple[str, ...]
    selected_automatically: bool
    selection_reason: str
    preferred_real_dtype: torch.dtype
    supports_float64: bool
    notes: tuple[str, ...]

    @property
    def is_gpu(self) -> bool:
        return self.device.type in {"cuda", "mps"}


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _available_backends() -> tuple[str, ...]:
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if _mps_is_available():
        backends.append("mps")
    return tuple(backends)


def resolve_torch_device(preferred: str | torch.device | None = "auto") -> torch.device:
    if isinstance(preferred, torch.device):
        device = preferred
    else:
        token = "auto" if preferred is None else str(preferred).lower()
        if token == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if _mps_is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            device = torch.device(token)
        except (TypeError, RuntimeError, ValueError) as exc:
            available = ", ".join(_available_backends())
            raise ValueError(
                f"unsupported torch device '{preferred}'. Use one of: auto, {available}"
            ) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested but no Apple Metal device is available")
    return device


def inspect_torch_runtime(preferred: str | torch.device | None = "auto") -> TorchRuntime:
    requested_device = "auto" if preferred is None else str(preferred)
    device = resolve_torch_device(preferred)
    available_backends = _available_backends()
    selected_automatically = not isinstance(preferred, torch.device) and str(preferred or "auto").lower() == "auto"
    notes: list[str] = []
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
        backend = "cuda"
        preferred_real_dtype = torch.float64
        supports_float64 = True
        selection_reason = "Auto-selected CUDA because a CUDA device is available." if selected_automatically else "Using the requested CUDA device."
    elif device.type == "mps":
        accelerator = "Apple Metal Performance Shaders"
        backend = "mps"
        preferred_real_dtype = torch.float32
        supports_float64 = False
        selection_reason = (
            "Auto-selected Apple Metal because CUDA is unavailable and MPS is available."
            if selected_automatically
            else "Using the requested Apple Metal device."
        )
        notes.append("MPS falls back to float32 for the core runtime because float64 is not supported.")
    else:
        accelerator = platform.processor() or "CPU"
        backend = "cpu"
        preferred_real_dtype = torch.float64
        supports_float64 = True
        selection_reason = "Auto-selected CPU because no accelerator backend is available." if selected_automatically else "Using the requested CPU device."
        notes.append("CPU is the baseline reference path for the product.")

    return TorchRuntime(
        requested_device=requested_device,
        system=platform.system(),
        machine=platform.machine().lower(),
        torch_version=torch.__version__,
        device=device,
        accelerator=accelerator,
        backend=backend,
        available_backends=available_backends,
        selected_automatically=selected_automatically,
        selection_reason=selection_reason,
        preferred_real_dtype=preferred_real_dtype,
        supports_float64=supports_float64,
        notes=tuple(notes),
    )


__all__ = [
    "TorchRuntime",
    "inspect_torch_runtime",
    "resolve_torch_device",
]
