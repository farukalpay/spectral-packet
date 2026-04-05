from __future__ import annotations

import pytest
import torch

pooch = pytest.importorskip("pooch")
scipy = pytest.importorskip("scipy")

from spectral_packet_engine import (
    DensityPreprocessingConfig,
    InfiniteWell1D,
    InfiniteWellBasis,
    download_and_prepare_quantum_gas_transport_scan,
    profile_mean,
    project_profiles_onto_basis,
    reconstruct_profiles_from_basis,
    relative_l2_error,
)


def test_real_quantum_transport_scan_supports_low_error_modal_compression() -> None:
    prepared = download_and_prepare_quantum_gas_transport_scan(
        preprocessing=DensityPreprocessingConfig(
            aggregate="mean",
            nan_fill_value=0.0,
            clip_negative=True,
            normalize_each_profile=True,
            drop_nonpositive_mass=True,
        )
    )

    grid, times, profiles = prepared.to_torch(dtype=torch.float64)
    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    basis = InfiniteWellBasis(domain, num_modes=32)

    coefficients = project_profiles_onto_basis(profiles, grid, basis)
    reconstruction = reconstruct_profiles_from_basis(coefficients, grid, basis)
    error = relative_l2_error(profiles, reconstruction, grid)
    center = profile_mean(profiles, grid)

    assert profiles.shape == (31, 126)
    assert float(torch.mean(error)) < 0.02
    assert float(torch.max(error)) < 0.03
    assert float(torch.min(center)) > grid[0].item()
    assert float(torch.max(center)) < grid[-1].item()
