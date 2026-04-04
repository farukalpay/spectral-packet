from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor
from spectral_packet_engine.state import PacketState, SpectralState

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    quadrature_points: int = 4096

    def __post_init__(self) -> None:
        if self.quadrature_points < 2:
            raise ValueError("quadrature_points must be at least 2")


@dataclass(frozen=True, slots=True)
class StateProjector:
    basis: InfiniteWellBasis
    config: ProjectionConfig = ProjectionConfig()

    @property
    def domain(self) -> InfiniteWell1D:
        return self.basis.domain

    def quadrature_grid(self, *, num_points: int | None = None) -> Tensor:
        return self.domain.grid(num_points or self.config.quadrature_points)

    def project_packet(self, packet_state: PacketState) -> SpectralState:
        grid = self.quadrature_grid()
        values = packet_state.wavefunction(grid)
        coefficients = self.project_coefficients(values, grid)
        return SpectralState(domain=self.domain, coefficients=coefficients)

    def project_coefficients(self, wavefunction, grid) -> Tensor:
        spatial_grid = coerce_tensor(grid, dtype=self.domain.real_dtype, device=self.domain.device)
        values = coerce_tensor(wavefunction, dtype=self.domain.complex_dtype, device=self.domain.device)
        if values.shape[-1] != spatial_grid.shape[0]:
            raise ValueError("wavefunction values must align with the grid")
        basis_values = self.basis.evaluate(spatial_grid).to(dtype=values.dtype, device=values.device)
        return torch.trapezoid(values[..., :, None] * basis_values, spatial_grid, dim=-2)

    def project_wavefunction(self, wavefunction, grid) -> SpectralState:
        coefficients = self.project_coefficients(wavefunction, grid)
        if coefficients.ndim != 1:
            raise ValueError("project_wavefunction expects a single wavefunction, not a batch")
        return SpectralState(domain=self.domain, coefficients=coefficients)

    def reconstruct(self, spectral_state: SpectralState, grid) -> Tensor:
        return self.basis.reconstruct(spectral_state.coefficients, grid)

    def reconstruct_coefficients(self, coefficients, grid) -> Tensor:
        return self.basis.reconstruct(coefficients, grid)


__all__ = [
    "ProjectionConfig",
    "StateProjector",
]
