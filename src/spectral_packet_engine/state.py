from __future__ import annotations

from dataclasses import dataclass, field

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_scalar_tensor, coerce_tensor

Tensor = torch.Tensor


def _coerce_parameter_vector(
    value,
    *,
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> Tensor:
    if isinstance(value, torch.Tensor):
        inferred_dtype = value.dtype if dtype is None else dtype
        inferred_device = value.device if device is None else device
        tensor = coerce_tensor(value, dtype=inferred_dtype, device=inferred_device)
    else:
        tensor = coerce_tensor(value, dtype=dtype or torch.float64, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        raise ValueError("packet parameters must be one-dimensional")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("packet parameters must be finite")
    return tensor


@dataclass(frozen=True, slots=True)
class GaussianPacketParameters:
    center: Tensor
    width: Tensor
    wavenumber: Tensor
    phase: Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))

    def __post_init__(self) -> None:
        center = _coerce_parameter_vector(self.center, dtype=None, device=None)
        width = _coerce_parameter_vector(self.width, dtype=center.dtype, device=center.device)
        wavenumber = _coerce_parameter_vector(
            self.wavenumber,
            dtype=center.dtype,
            device=center.device,
        )
        phase = _coerce_parameter_vector(self.phase, dtype=center.dtype, device=center.device)

        if not (center.shape == width.shape == wavenumber.shape == phase.shape):
            raise ValueError("center, width, wavenumber, and phase must have matching shapes")
        if not torch.all(width > 0).item():
            raise ValueError("packet widths must be positive")

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "wavenumber", wavenumber)
        object.__setattr__(self, "phase", phase)

    @property
    def packet_count(self) -> int:
        return int(self.center.shape[0])

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> "GaussianPacketParameters":
        return GaussianPacketParameters(
            center=self.center.to(dtype=dtype, device=device),
            width=self.width.to(dtype=dtype, device=device),
            wavenumber=self.wavenumber.to(dtype=dtype, device=device),
            phase=self.phase.to(dtype=dtype, device=device),
        )

    @classmethod
    def single(
        cls,
        *,
        center,
        width,
        wavenumber,
        phase=0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "GaussianPacketParameters":
        center_tensor = coerce_tensor(center, dtype=dtype, device=device)
        width_tensor = coerce_tensor(width, dtype=dtype, device=device)
        wavenumber_tensor = coerce_tensor(wavenumber, dtype=dtype, device=device)
        phase_tensor = coerce_tensor(phase, dtype=dtype, device=device)
        return cls(
            center=torch.atleast_1d(center_tensor),
            width=torch.atleast_1d(width_tensor),
            wavenumber=torch.atleast_1d(wavenumber_tensor),
            phase=torch.atleast_1d(phase_tensor),
        )


@dataclass(frozen=True, slots=True)
class PacketState:
    domain: InfiniteWell1D
    parameters: GaussianPacketParameters
    weights: Tensor | None = None
    normalize_components: bool = True

    def __post_init__(self) -> None:
        parameters = self.parameters.to(dtype=self.domain.real_dtype, device=self.domain.device)
        weights = self.weights
        if weights is None:
            weights = torch.ones(
                parameters.packet_count,
                dtype=self.domain.complex_dtype,
                device=self.domain.device,
            )
        else:
            weights = coerce_tensor(weights, dtype=self.domain.complex_dtype, device=self.domain.device)
            if weights.ndim == 0:
                weights = weights.repeat(parameters.packet_count)
            if weights.ndim != 1:
                raise ValueError("packet weights must be one-dimensional")
            if weights.shape[0] != parameters.packet_count:
                raise ValueError("packet weights must match the number of packet parameters")
            if not torch.isfinite(weights.real).all().item() or not torch.isfinite(weights.imag).all().item():
                raise ValueError("packet weights must be finite")

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "weights", weights)

    @property
    def packet_count(self) -> int:
        return self.parameters.packet_count

    def component_normalizations(self) -> Tensor:
        widths = self.parameters.width
        centers = self.parameters.center
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=self.domain.real_dtype, device=self.domain.device))
        interval_mass = widths * torch.sqrt(
            torch.tensor(torch.pi / 2.0, dtype=self.domain.real_dtype, device=self.domain.device)
        )
        interval_mass = interval_mass * (
            torch.erf((self.domain.right - centers) / (sqrt_two * widths))
            + torch.erf((centers - self.domain.left) / (sqrt_two * widths))
        )
        if torch.any(interval_mass <= 0).item():
            raise ValueError("packet normalization integral must be positive")
        return torch.rsqrt(interval_mass).to(dtype=self.domain.complex_dtype)

    def component_wavefunctions(self, x) -> Tensor:
        grid = coerce_tensor(x, dtype=self.domain.real_dtype, device=self.domain.device)
        flat_grid = grid.reshape(-1)
        centers = self.parameters.center[:, None]
        widths = self.parameters.width[:, None]
        wavenumbers = self.parameters.wavenumber[:, None]
        phases = self.parameters.phase[:, None]

        envelope = torch.exp(-((flat_grid[None, :] - centers) ** 2) / (4 * widths**2))
        oscillation = torch.exp(1j * (wavenumbers * flat_grid[None, :] + phases))
        components = envelope.to(dtype=self.domain.complex_dtype) * oscillation.to(dtype=self.domain.complex_dtype)
        support = self.domain.contains(flat_grid)[None, :]
        components = torch.where(support, components, torch.zeros_like(components))
        if self.normalize_components:
            components = self.component_normalizations()[:, None] * components
        return components.reshape(self.packet_count, *grid.shape)

    def wavefunction(self, x) -> Tensor:
        components = self.component_wavefunctions(x)
        reshaped_weights = self.weights.reshape(self.packet_count, *([1] * (components.ndim - 1)))
        return torch.sum(reshaped_weights * components, dim=0)


@dataclass(frozen=True, slots=True)
class SpectralState:
    domain: InfiniteWell1D
    coefficients: Tensor

    def __post_init__(self) -> None:
        coefficients = coerce_tensor(
            self.coefficients,
            dtype=self.domain.complex_dtype,
            device=self.domain.device,
        )
        if coefficients.ndim != 1:
            raise ValueError("spectral coefficients must be one-dimensional")
        if not torch.isfinite(coefficients.real).all().item() or not torch.isfinite(coefficients.imag).all().item():
            raise ValueError("spectral coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def num_modes(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def norm_squared(self) -> Tensor:
        return torch.sum(torch.abs(self.coefficients) ** 2)

    def wavefunction(self, x) -> Tensor:
        from spectral_packet_engine.basis import InfiniteWellBasis

        basis = InfiniteWellBasis(self.domain, self.num_modes)
        return basis.reconstruct(self.coefficients, x)


def make_truncated_gaussian_packet(
    domain: InfiniteWell1D,
    *,
    center,
    width,
    wavenumber,
    phase=0.0,
    weight=1.0 + 0.0j,
) -> PacketState:
    parameters = GaussianPacketParameters.single(
        center=center,
        width=width,
        wavenumber=wavenumber,
        phase=phase,
        dtype=domain.real_dtype,
        device=domain.device,
    )
    weight_tensor = coerce_scalar_tensor(
        torch.as_tensor(weight, dtype=domain.complex_dtype, device=domain.device),
        dtype=domain.complex_dtype,
        device=domain.device,
    ).reshape(1)
    return PacketState(domain=domain, parameters=parameters, weights=weight_tensor)


__all__ = [
    "GaussianPacketParameters",
    "PacketState",
    "SpectralState",
    "make_truncated_gaussian_packet",
]
