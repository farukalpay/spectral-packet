from __future__ import annotations

import torch

from spectral_packet_engine import inspect_torch_runtime, resolve_torch_device


def test_resolve_torch_device_supports_cpu_and_auto() -> None:
    assert resolve_torch_device("cpu").type == "cpu"
    assert resolve_torch_device("auto").type in {"cpu", "cuda", "mps"}


def test_inspect_torch_runtime_matches_selected_device() -> None:
    runtime = inspect_torch_runtime("cpu")

    assert runtime.backend == "cpu"
    assert runtime.device == torch.device("cpu")
    assert runtime.requested_device == "cpu"
    assert "cpu" in runtime.available_backends
    assert runtime.selected_automatically is False
    assert runtime.selection_reason
    assert runtime.accelerator
    assert runtime.preferred_real_dtype == torch.float64
    assert runtime.supports_float64 is True
    assert runtime.torch_version
    assert runtime.notes
