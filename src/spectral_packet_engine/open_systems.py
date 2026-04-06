from __future__ import annotations

"""Open-system and finite-resolution measurement contracts.

This module extends the existing density-matrix and quantum-channel layer
without turning the engine into a generic simulator.  It keeps the contract
matrix-based: closed-system Hamiltonians, Lindblad jump operators, and
instrument response matrices are explicit objects that spectroscopy,
transport, and control workflows can reuse.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from spectral_packet_engine.density_matrix import purity, von_neumann_entropy
from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor

_CDTYPE = torch.complex128
_RDTYPE = torch.float64


def _coerce_square_matrix(value: Any, *, name: str) -> Tensor:
    matrix = coerce_tensor(value, dtype=_CDTYPE)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _symmetrize_density_matrix(rho: Tensor) -> Tensor:
    return 0.5 * (rho + rho.conj().T)


@dataclass(frozen=True, slots=True)
class LindbladOperator:
    """One dissipative channel in a Lindblad master equation."""

    name: str
    matrix: Tensor
    rate: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Lindblad operator name must be non-empty")
        if self.rate < 0.0:
            raise ValueError("Lindblad operator rate must be non-negative")
        object.__setattr__(self, "matrix", _coerce_square_matrix(self.matrix, name=self.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate": self.rate,
            "shape": list(self.matrix.shape),
        }


@dataclass(frozen=True, slots=True)
class OpenSystemEvolutionSummary:
    """Density-matrix trajectory and diagnostics for an open-system run."""

    density_matrices: Tensor
    times: Tensor
    trace: Tensor
    purity: Tensor
    von_neumann_entropy: Tensor
    channels: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstrumentResponse:
    """Finite-resolution measurement response matrix."""

    response_matrix: Tensor
    observable_axis: str = "position"
    units: str = "bin"

    def __post_init__(self) -> None:
        matrix = coerce_tensor(self.response_matrix, dtype=_RDTYPE)
        if matrix.ndim != 2:
            raise ValueError("response_matrix must be two-dimensional")
        if torch.any(matrix < 0).item():
            raise ValueError("response_matrix entries must be non-negative")
        column_sum = matrix.sum(dim=0)
        if torch.any(column_sum <= 0).item():
            raise ValueError("response_matrix input columns must have positive mass")
        object.__setattr__(self, "response_matrix", matrix / column_sum[None, :])

    def apply(self, distribution: Tensor) -> Tensor:
        values = coerce_tensor(distribution, dtype=self.response_matrix.dtype, device=self.response_matrix.device)
        if values.shape[-1] != self.response_matrix.shape[1]:
            raise ValueError("distribution last dimension must match response_matrix input bins")
        flattened = values.reshape(-1, values.shape[-1])
        measured = flattened @ self.response_matrix.transpose(0, 1)
        return measured.reshape(*values.shape[:-1], self.response_matrix.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_axis": self.observable_axis,
            "units": self.units,
            "input_bins": int(self.response_matrix.shape[1]),
            "output_bins": int(self.response_matrix.shape[0]),
        }


@dataclass(frozen=True, slots=True)
class MeasurementNoiseModel:
    """Inspectable measurement-noise contract."""

    model: str = "independent-gaussian"
    scale: float = 0.0
    units: str = "same-as-observable"

    def __post_init__(self) -> None:
        model = self.model.strip().lower()
        object.__setattr__(self, "model", model)
        if model not in {"none", "independent-gaussian", "poisson-approx"}:
            raise ValueError("noise model must be one of: none, independent-gaussian, poisson-approx")
        if self.scale < 0.0:
            raise ValueError("noise scale must be non-negative")

    def uncertainty(self, measured: Tensor) -> Tensor:
        if self.model == "none" or self.scale == 0.0:
            return torch.zeros_like(measured)
        if self.model == "poisson-approx":
            return self.scale * torch.sqrt(torch.clamp(measured, min=0.0))
        return torch.full_like(measured, self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scale": self.scale,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class MeasurementResponseSummary:
    measured: Tensor
    uncertainty: Tensor
    response: dict[str, Any]
    noise_model: dict[str, Any]
    normalization_error: Tensor
    assumptions: tuple[str, ...]


def dephasing_lindblad_operator(
    dim: int,
    mode: int,
    *,
    rate: float,
    name: str | None = None,
) -> LindbladOperator:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if not (0 <= mode < dim):
        raise ValueError("mode must be a valid basis index")
    matrix = torch.zeros(dim, dim, dtype=_CDTYPE)
    matrix[mode, mode] = 1.0
    return LindbladOperator(name or f"dephasing[{mode}]", matrix, rate=rate)


def relaxation_lindblad_operator(
    dim: int,
    *,
    source: int,
    target: int,
    rate: float,
    name: str | None = None,
) -> LindbladOperator:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if not (0 <= source < dim and 0 <= target < dim):
        raise ValueError("source and target must be valid basis indices")
    if source == target:
        raise ValueError("source and target must differ for relaxation")
    matrix = torch.zeros(dim, dim, dtype=_CDTYPE)
    matrix[target, source] = 1.0
    return LindbladOperator(name or f"relaxation[{source}->{target}]", matrix, rate=rate)


def lindblad_rhs(
    rho: Tensor,
    hamiltonian: Tensor,
    channels: Sequence[LindbladOperator],
) -> Tensor:
    density = _coerce_square_matrix(rho, name="rho")
    h = _coerce_square_matrix(hamiltonian, name="hamiltonian").to(device=density.device)
    if h.shape != density.shape:
        raise ValueError("hamiltonian shape must match rho")
    rhs = -1j * (h @ density - density @ h)
    for channel in channels:
        jump = channel.matrix.to(dtype=density.dtype, device=density.device)
        if jump.shape != density.shape:
            raise ValueError(f"channel {channel.name!r} shape must match rho")
        jump_dag_jump = jump.conj().T @ jump
        dissipator = jump @ density @ jump.conj().T - 0.5 * (jump_dag_jump @ density + density @ jump_dag_jump)
        rhs = rhs + channel.rate * dissipator
    return rhs


def _rk4_lindblad_step(
    rho: Tensor,
    hamiltonian: Tensor,
    channels: Sequence[LindbladOperator],
    dt: Tensor,
) -> Tensor:
    k1 = lindblad_rhs(rho, hamiltonian, channels)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, hamiltonian, channels)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, hamiltonian, channels)
    k4 = lindblad_rhs(rho + dt * k3, hamiltonian, channels)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_lindblad(
    rho0: Tensor,
    hamiltonian: Tensor,
    channels: Sequence[LindbladOperator],
    times: Sequence[float] | Tensor,
    *,
    preserve_trace: bool = True,
) -> OpenSystemEvolutionSummary:
    density = _coerce_square_matrix(rho0, name="rho0")
    h = _coerce_square_matrix(hamiltonian, name="hamiltonian").to(dtype=density.dtype, device=density.device)
    if h.shape != density.shape:
        raise ValueError("hamiltonian shape must match rho0")
    time_grid = coerce_tensor(times, dtype=_RDTYPE, device=density.device)
    if time_grid.ndim != 1 or time_grid.numel() == 0:
        raise ValueError("times must be a non-empty one-dimensional grid")
    if time_grid.numel() > 1 and not torch.all(time_grid[1:] >= time_grid[:-1]).item():
        raise ValueError("times must be non-decreasing")

    current = _symmetrize_density_matrix(density)
    if preserve_trace:
        trace = torch.trace(current)
        if torch.abs(trace) > 0:
            current = current / trace
    trajectory: list[Tensor] = [current.detach()]
    for index in range(1, int(time_grid.numel())):
        dt = time_grid[index] - time_grid[index - 1]
        current = _rk4_lindblad_step(current, h, channels, dt)
        current = _symmetrize_density_matrix(current)
        if preserve_trace:
            trace = torch.trace(current)
            if torch.abs(trace) > 0:
                current = current / trace
        trajectory.append(current.detach())

    stacked = torch.stack(trajectory)
    trace_values = torch.stack([torch.trace(item).real for item in stacked])
    purity_values = torch.stack([purity(item) for item in stacked])
    entropy_values = torch.stack([von_neumann_entropy(item) for item in stacked])
    return OpenSystemEvolutionSummary(
        density_matrices=stacked,
        times=time_grid.detach(),
        trace=trace_values.detach(),
        purity=purity_values.detach(),
        von_neumann_entropy=entropy_values.detach(),
        channels=tuple(channel.to_dict() for channel in channels),
        assumptions=(
            "Evolution uses a fourth-order Runge-Kutta step over the explicit user-provided time grid.",
            "The density matrix is symmetrized after each step; trace renormalization is explicit and controlled by preserve_trace.",
            "This is an open-system reduced model over an explicit matrix basis, not a generic environment simulator.",
        ),
    )


def finite_resolution_response_matrix(
    coordinates: Sequence[float] | Tensor,
    *,
    sigma: float,
) -> Tensor:
    grid = coerce_tensor(coordinates, dtype=_RDTYPE)
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("coordinates must be a one-dimensional grid with at least two points")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    displacement = grid[:, None] - grid[None, :]
    response = torch.exp(-0.5 * (displacement / sigma) ** 2)
    return response / response.sum(dim=0, keepdim=True)


def apply_instrument_response(
    distribution: Tensor,
    response: InstrumentResponse | Tensor,
    *,
    noise_model: MeasurementNoiseModel | None = None,
) -> MeasurementResponseSummary:
    instrument = response if isinstance(response, InstrumentResponse) else InstrumentResponse(response)
    noise = noise_model or MeasurementNoiseModel(model="none", scale=0.0)
    measured = instrument.apply(distribution)
    uncertainty = noise.uncertainty(measured)
    input_mass = torch.sum(coerce_tensor(distribution, dtype=measured.dtype, device=measured.device), dim=-1)
    measured_mass = torch.sum(measured, dim=-1)
    return MeasurementResponseSummary(
        measured=measured.detach(),
        uncertainty=uncertainty.detach(),
        response=instrument.to_dict(),
        noise_model=noise.to_dict(),
        normalization_error=torch.abs(measured_mass - input_mass).detach(),
        assumptions=(
            "The response matrix is normalized over output bins for each input bin and applied along the last distribution axis.",
            "Noise output is an uncertainty summary; this function does not sample random noise.",
        ),
    )


__all__ = [
    "InstrumentResponse",
    "LindbladOperator",
    "MeasurementNoiseModel",
    "MeasurementResponseSummary",
    "OpenSystemEvolutionSummary",
    "apply_instrument_response",
    "dephasing_lindblad_operator",
    "evolve_lindblad",
    "finite_resolution_response_matrix",
    "lindblad_rhs",
    "relaxation_lindblad_operator",
]
