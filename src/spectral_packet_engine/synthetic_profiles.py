from __future__ import annotations

import math

import numpy as np
import torch

from spectral_packet_engine.runtime import resolve_torch_device
from spectral_packet_engine.table_io import ProfileTable


def _validate_positive_integer(name: str, value: int) -> int:
    integer_value = int(value)
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")
    return integer_value


def generate_synthetic_profile_table(
    *,
    num_profiles: int = 50,
    grid_points: int = 64,
    device: str = "cpu",
) -> ProfileTable:
    """Generate a deterministic profile table that is valid for engine workflows."""

    resolved_profiles = _validate_positive_integer("num_profiles", num_profiles)
    resolved_grid_points = _validate_positive_integer("grid_points", grid_points)
    if resolved_grid_points < 4:
        raise ValueError("grid_points must be at least 4 for stable Gaussian profile synthesis")

    torch_device = resolve_torch_device(device)
    grid = torch.linspace(0.0, 1.0, resolved_grid_points, dtype=torch.float64, device=torch_device)
    sample_times = torch.linspace(
        0.0,
        float(max(resolved_profiles - 1, 0)),
        resolved_profiles,
        dtype=torch.float64,
        device=torch_device,
    )

    centers = torch.linspace(0.2, 0.8, resolved_profiles, dtype=torch.float64, device=torch_device)
    phase = torch.linspace(0.0, 2.0 * math.pi, resolved_profiles, dtype=torch.float64, device=torch_device)
    widths = 0.045 + 0.02 * (1.0 + torch.sin(phase))

    profiles: list[torch.Tensor] = []
    for center, width in zip(centers, widths, strict=False):
        profile = torch.exp(-((grid - center) ** 2) / (2.0 * width**2))
        mass = torch.trapezoid(profile, grid)
        if not torch.isfinite(mass) or float(mass.item()) <= 0.0:
            raise RuntimeError("synthetic profile generation produced a non-finite normalization mass")
        profiles.append(profile / mass)

    profile_tensor = torch.stack(profiles)
    if not torch.isfinite(profile_tensor).all():
        raise RuntimeError("synthetic profile generation produced non-finite values")

    return ProfileTable(
        position_grid=np.asarray(grid.detach().cpu().numpy(), dtype=np.float64),
        sample_times=np.asarray(sample_times.detach().cpu().numpy(), dtype=np.float64),
        profiles=np.asarray(profile_tensor.detach().cpu().numpy(), dtype=np.float64),
        source="synthetic-profile-generator",
    )


__all__ = ["generate_synthetic_profile_table"]
