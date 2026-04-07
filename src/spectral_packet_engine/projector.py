from __future__ import annotations

from dataclasses import dataclass, field

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
    _quadrature_grid_cache: Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _quadrature_basis_cache: Tensor | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def domain(self) -> InfiniteWell1D:
        return self.basis.domain

    def _quadrature_cache(self) -> tuple[Tensor, Tensor]:
        cached_grid = self._quadrature_grid_cache
        cached_basis = self._quadrature_basis_cache
        if cached_grid is None or cached_basis is None:
            cached_grid = self.domain.grid(self.config.quadrature_points)
            cached_basis = self.basis.evaluate(cached_grid)
            object.__setattr__(self, "_quadrature_grid_cache", cached_grid)
            object.__setattr__(self, "_quadrature_basis_cache", cached_basis)
        return cached_grid, cached_basis

    def quadrature_grid(self, *, num_points: int | None = None) -> Tensor:
        if num_points is None or num_points == self.config.quadrature_points:
            cached_grid, _ = self._quadrature_cache()
            return cached_grid
        return self.domain.grid(num_points)

    def quadrature_basis_values(self, *, num_points: int | None = None) -> Tensor:
        if num_points is None or num_points == self.config.quadrature_points:
            _, cached_basis = self._quadrature_cache()
            return cached_basis
        return self.basis.evaluate(self.quadrature_grid(num_points=num_points))

    def project_packet(self, packet_state: PacketState) -> SpectralState:
        grid, basis_values = self._quadrature_cache()
        values = packet_state.wavefunction(grid)
        basis_values = basis_values.to(dtype=values.dtype, device=values.device)
        coefficients = torch.trapezoid(values[..., :, None] * basis_values, grid, dim=-2)
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

    def reconstruct_coefficients(self, coefficients, grid) -> Tensor:
        return self.basis.reconstruct(coefficients, grid)

    def reconstruct(self, spectral_state: SpectralState, grid) -> Tensor:
        """Convenience wrapper: extracts coefficients and delegates."""
        return self.reconstruct_coefficients(spectral_state.coefficients, grid)


__all__ = [
    "ProjectionConfig",
    "StateProjector",
]
