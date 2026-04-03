from __future__ import annotations

from dataclasses import dataclass
import platform

import torch


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    system: str
    machine: str
    torch_version: str
    device: torch.device
    accelerator: str
    backend: str
    preferred_real_dtype: torch.dtype
    supports_float64: bool

    @property
    def is_gpu(self) -> bool:
        return self.device.type in {"cuda", "mps"}


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


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
        device = torch.device(token)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested but no Apple Metal device is available")
    return device


def inspect_torch_runtime(preferred: str | torch.device | None = "auto") -> TorchRuntime:
    device = resolve_torch_device(preferred)
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
        backend = "cuda"
        preferred_real_dtype = torch.float64
        supports_float64 = True
    elif device.type == "mps":
        accelerator = "Apple Metal Performance Shaders"
        backend = "mps"
        preferred_real_dtype = torch.float32
        supports_float64 = False
    else:
        accelerator = platform.processor() or "CPU"
        backend = "cpu"
        preferred_real_dtype = torch.float64
        supports_float64 = True

    return TorchRuntime(
        system=platform.system(),
        machine=platform.machine().lower(),
        torch_version=torch.__version__,
        device=device,
        accelerator=accelerator,
        backend=backend,
        preferred_real_dtype=preferred_real_dtype,
        supports_float64=supports_float64,
    )


__all__ = [
    "TorchRuntime",
    "inspect_torch_runtime",
    "resolve_torch_device",
]
