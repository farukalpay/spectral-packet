from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import platform
import subprocess

import torch

_log = logging.getLogger(__name__)


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
    cpu_features: tuple[str, ...]
    num_threads: int

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


def _detect_cpu_features() -> tuple[str, ...]:
    """Detect CPU SIMD features (AVX, AVX2, AVX-512, etc.)."""
    features: list[str] = []
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags = line.split(":", 1)[1].lower()
                        for feat in ("avx512f", "avx512bw", "avx512vl", "avx2", "avx", "sse4_2", "sse4_1", "fma"):
                            if feat in flags:
                                features.append(feat)
                        break
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                flags = result.stdout.lower()
                for feat in ("avx512f", "avx2", "avx", "sse4.2", "sse4.1", "fma"):
                    if feat.replace(".", "_") in flags.replace(".", "_"):
                        features.append(feat)
    except Exception:
        _log.debug("CPU feature detection failed, continuing without feature info")
    return tuple(features)


def _optimal_thread_count() -> int:
    """Determine optimal thread count for CPU compute."""
    physical = os.cpu_count() or 4
    # Use physical cores (not hyperthreads) for compute-bound work
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                cores = content.count("processor\t:")
                threads_per_core = 1
                for line in content.split("\n"):
                    if "siblings" in line:
                        siblings = int(line.split(":")[1].strip())
                        for line2 in content.split("\n"):
                            if "cpu cores" in line2:
                                cpu_cores = int(line2.split(":")[1].strip())
                                threads_per_core = max(1, siblings // cpu_cores)
                                break
                        break
                physical = max(1, cores // threads_per_core)
    except Exception:
        _log.debug("Physical core count detection failed, estimating from os.cpu_count()")
        physical = max(1, (os.cpu_count() or 4) // 2)
    return physical


def optimize_cpu() -> dict[str, int | str]:
    """Apply CPU-specific optimizations for PyTorch.

    Sets thread counts for OpenMP, MKL, and PyTorch based on hardware.
    Called automatically during engine startup.

    Returns dict with applied settings.
    """
    num_threads = _optimal_thread_count()
    features = _detect_cpu_features()

    # Set PyTorch thread count (interop can only be set before parallel work starts)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except RuntimeError:
        _log.debug("Interop threads already set or parallel work started, skipping")

    # Set environment for MKL and OpenMP (affects future library loads)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))

    # Enable TF32 on Ampere+ GPUs (if CUDA present)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    return {
        "num_threads": num_threads,
        "interop_threads": max(1, num_threads // 2),
        "cpu_features": features,
        "avx512": "avx512f" in features,
        "avx2": "avx2" in features,
    }


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

    # Apply CPU optimizations on first call
    cpu_opt = optimize_cpu()
    cpu_features = cpu_opt["cpu_features"]
    num_threads = cpu_opt["num_threads"]

    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
        backend = "cuda"
        preferred_real_dtype = torch.float64
        supports_float64 = True
        selection_reason = "Auto-selected CUDA because a CUDA device is available." if selected_automatically else "Using the requested CUDA device."
        gpu_mem = torch.cuda.get_device_properties(device).total_mem
        notes.append(f"GPU memory: {gpu_mem / 1e9:.1f} GB. TF32 enabled for Ampere+.")
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
        proc = platform.processor() or platform.machine()
        accelerator = proc
        backend = "cpu"
        preferred_real_dtype = torch.float64
        supports_float64 = True
        selection_reason = "Auto-selected CPU because no accelerator backend is available." if selected_automatically else "Using the requested CPU device."
        if cpu_opt["avx512"]:
            notes.append(f"CPU optimized: {num_threads} threads, AVX-512 enabled.")
        elif cpu_opt["avx2"]:
            notes.append(f"CPU optimized: {num_threads} threads, AVX2 enabled.")
        else:
            notes.append(f"CPU optimized: {num_threads} threads.")

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
        cpu_features=cpu_features,
        num_threads=num_threads,
    )


__all__ = [
    "TorchRuntime",
    "inspect_torch_runtime",
    "resolve_torch_device",
    "optimize_cpu",
]
