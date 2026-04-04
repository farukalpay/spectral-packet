from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_scalar_tensor, coerce_tensor

Tensor = torch.Tensor


def _coerce_modes(modes, *, dtype: torch.dtype, device: torch.device | str | None) -> Tensor:
    tensor = coerce_tensor(modes, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        raise ValueError("modes must be a one-dimensional tensor or sequence")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("modes must be finite")
    if not torch.all(tensor > 0).item():
        raise ValueError("modes must be positive")
    rounded = torch.round(tensor)
    if not torch.allclose(tensor, rounded):
        raise ValueError("modes must be integer-valued")
    return rounded


def sine_basis_matrix(domain: InfiniteWell1D, modes, x) -> Tensor:
    grid = coerce_tensor(x, dtype=domain.real_dtype, device=domain.device)
    if grid.ndim == 0:
        grid = grid.reshape(1)
    mode_numbers = _coerce_modes(modes, dtype=domain.real_dtype, device=domain.device)
    factor = torch.sqrt(torch.tensor(2.0, dtype=domain.real_dtype, device=domain.device) / domain.length)
    scaled = (grid[..., None] - domain.left) / domain.length
    return factor * torch.sin(torch.pi * scaled * mode_numbers)


def eigenenergy(
    domain: InfiniteWell1D,
    mode_index,
) -> Tensor:
    n = coerce_scalar_tensor(mode_index, dtype=domain.real_dtype, device=domain.device)
    if not torch.isclose(n, torch.round(n)).item() or not (n > 0).item():
        raise ValueError("mode index must be a positive integer")
    return (torch.pi**2 * domain.hbar**2 * n**2) / (2 * domain.mass * domain.length**2)


def eigenenergies(
    domain: InfiniteWell1D,
    modes,
) -> Tensor:
    mode_numbers = _coerce_modes(modes, dtype=domain.real_dtype, device=domain.device)
    return (torch.pi**2 * domain.hbar**2 * mode_numbers**2) / (2 * domain.mass * domain.length**2)


@dataclass(frozen=True, slots=True)
class InfiniteWellBasis:
    domain: InfiniteWell1D
    num_modes: int

    def __post_init__(self) -> None:
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")

    @property
    def mode_numbers(self) -> Tensor:
        return torch.arange(
            1,
            self.num_modes + 1,
            dtype=self.domain.real_dtype,
            device=self.domain.device,
        )

    @property
    def energies(self) -> Tensor:
        return eigenenergies(self.domain, self.mode_numbers)

    @property
    def wavenumbers(self) -> Tensor:
        return self.mode_numbers * torch.pi / self.domain.length

    def evaluate(self, x) -> Tensor:
        return sine_basis_matrix(self.domain, self.mode_numbers, x)

    def reconstruct(self, coefficients, x) -> Tensor:
        coeffs = coerce_tensor(coefficients, dtype=self.domain.complex_dtype, device=self.domain.device)
        basis = self.evaluate(x).to(dtype=coeffs.dtype, device=coeffs.device)
        if coeffs.ndim == 1:
            if coeffs.shape[0] != self.num_modes:
                raise ValueError("coefficient vector length does not match the basis")
            return basis @ coeffs
        if coeffs.ndim == 2:
            if coeffs.shape[-1] != self.num_modes:
                raise ValueError("coefficient matrix width does not match the basis")
            return coeffs @ basis.transpose(0, 1)
        raise ValueError("coefficients must be one- or two-dimensional")


__all__ = [
    "InfiniteWellBasis",
    "eigenenergy",
    "eigenenergies",
    "sine_basis_matrix",
]
