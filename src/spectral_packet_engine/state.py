from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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


def _strict_domain_support(domain: InfiniteWell1D, grid: Tensor) -> Tensor:
    return (grid > domain.left) & (grid < domain.right)


@dataclass(frozen=True, slots=True)
class PacketSupportDiagnostics:
    inside_probability_mass: Tensor
    outside_probability_mass: Tensor
    left_boundary_density: Tensor
    right_boundary_density: Tensor
    boundary_density_mismatch: Tensor


@runtime_checkable
class PacketParameterization(Protocol):
    @property
    def packet_count(self) -> int: ...

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> "PacketParameterization": ...

    def component_wavefunctions(
        self,
        domain: InfiniteWell1D,
        x,
        *,
        normalize: bool = True,
    ) -> Tensor: ...

    def support_diagnostics(self, domain: InfiniteWell1D) -> PacketSupportDiagnostics: ...


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

    def _raw_interval_mass(self, domain: InfiniteWell1D) -> Tensor:
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=domain.real_dtype, device=domain.device))
        interval_mass = self.width * torch.sqrt(
            torch.tensor(torch.pi / 2.0, dtype=domain.real_dtype, device=domain.device)
        )
        return interval_mass * (
            torch.erf((domain.right - self.center) / (sqrt_two * self.width))
            - torch.erf((domain.left - self.center) / (sqrt_two * self.width))
        )

    def _inside_probability_mass(self, domain: InfiniteWell1D) -> Tensor:
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=domain.real_dtype, device=domain.device))
        right_term = torch.erf((domain.right - self.center) / (sqrt_two * self.width))
        left_term = torch.erf((domain.left - self.center) / (sqrt_two * self.width))
        inside = 0.5 * (right_term - left_term)
        return torch.clamp(inside, min=0.0, max=1.0)

    def component_normalizations(self, domain: InfiniteWell1D) -> Tensor:
        interval_mass = self._raw_interval_mass(domain)
        if torch.any(interval_mass <= 0).item():
            raise ValueError("packet normalization integral must be positive inside the bounded domain")
        scale = torch.rsqrt(
            torch.clamp(
                interval_mass,
                min=torch.finfo(domain.real_dtype).tiny,
            )
        )
        return scale.to(dtype=domain.complex_dtype)

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

    def component_wavefunctions(
        self,
        domain: InfiniteWell1D,
        x,
        *,
        normalize: bool = True,
    ) -> Tensor:
        grid = coerce_tensor(x, dtype=domain.real_dtype, device=domain.device)
        flat_grid = grid.reshape(-1)
        centers = self.center[:, None]
        widths = self.width[:, None]
        wavenumbers = self.wavenumber[:, None]
        phases = self.phase[:, None]

        envelope = torch.exp(-((flat_grid[None, :] - centers) ** 2) / (4 * widths**2))
        oscillation = torch.exp(1j * (wavenumbers * flat_grid[None, :] + phases))
        components = envelope.to(dtype=domain.complex_dtype) * oscillation.to(dtype=domain.complex_dtype)
        support = _strict_domain_support(domain, flat_grid)[None, :]
        components = torch.where(support, components, torch.zeros_like(components))
        if normalize:
            components = self.component_normalizations(domain)[:, None] * components
        return components.reshape(self.packet_count, *grid.shape)

    def support_diagnostics(self, domain: InfiniteWell1D) -> PacketSupportDiagnostics:
        inside_mass = self._inside_probability_mass(domain)
        outside_mass = torch.clamp(1.0 - inside_mass, min=0.0, max=1.0)
        density_prefactor = torch.rsqrt(
            2.0 * torch.pi * self.width**2
        )
        left_density = density_prefactor * torch.exp(-((domain.left - self.center) ** 2) / (2 * self.width**2))
        right_density = density_prefactor * torch.exp(-((domain.right - self.center) ** 2) / (2 * self.width**2))
        return PacketSupportDiagnostics(
            inside_probability_mass=inside_mass,
            outside_probability_mass=outside_mass,
            left_boundary_density=left_density,
            right_boundary_density=right_density,
            boundary_density_mismatch=left_density + right_density,
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
class PlaneWavePacketParameters:
    wavenumber: Tensor
    phase: Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))

    def __post_init__(self) -> None:
        wavenumber = _coerce_parameter_vector(self.wavenumber, dtype=None, device=None)
        phase = _coerce_parameter_vector(self.phase, dtype=wavenumber.dtype, device=wavenumber.device)
        if wavenumber.shape != phase.shape:
            raise ValueError("wavenumber and phase must have matching shapes")

        object.__setattr__(self, "wavenumber", wavenumber)
        object.__setattr__(self, "phase", phase)

    @property
    def packet_count(self) -> int:
        return int(self.wavenumber.shape[0])

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> "PlaneWavePacketParameters":
        return PlaneWavePacketParameters(
            wavenumber=self.wavenumber.to(dtype=dtype, device=device),
            phase=self.phase.to(dtype=dtype, device=device),
        )

    def component_wavefunctions(
        self,
        domain: InfiniteWell1D,
        x,
        *,
        normalize: bool = True,
    ) -> Tensor:
        grid = coerce_tensor(x, dtype=domain.real_dtype, device=domain.device)
        flat_grid = grid.reshape(-1)
        wavenumbers = self.wavenumber[:, None]
        phases = self.phase[:, None]

        components = torch.exp(1j * (wavenumbers * flat_grid[None, :] + phases)).to(dtype=domain.complex_dtype)
        support = _strict_domain_support(domain, flat_grid)[None, :]
        components = torch.where(support, components, torch.zeros_like(components))
        if normalize:
            normalization = torch.rsqrt(domain.length).to(dtype=domain.complex_dtype)
            components = normalization * components
        return components.reshape(self.packet_count, *grid.shape)

    def support_diagnostics(self, domain: InfiniteWell1D) -> PacketSupportDiagnostics:
        density = torch.full(
            self.wavenumber.shape,
            (1.0 / domain.length).item(),
            dtype=domain.real_dtype,
            device=domain.device,
        )
        zero = torch.zeros_like(density)
        return PacketSupportDiagnostics(
            inside_probability_mass=torch.ones_like(density),
            outside_probability_mass=zero,
            left_boundary_density=density,
            right_boundary_density=density,
            boundary_density_mismatch=2.0 * density,
        )

    @classmethod
    def single(
        cls,
        *,
        wavenumber,
        phase=0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "PlaneWavePacketParameters":
        wavenumber_tensor = coerce_tensor(wavenumber, dtype=dtype, device=device)
        phase_tensor = coerce_tensor(phase, dtype=dtype, device=device)
        return cls(
            wavenumber=torch.atleast_1d(wavenumber_tensor),
            phase=torch.atleast_1d(phase_tensor),
        )


@dataclass(frozen=True, slots=True)
class WindowedPlaneWavePacketParameters:
    center: Tensor
    window_width: Tensor
    wavenumber: Tensor
    phase: Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))

    def __post_init__(self) -> None:
        center = _coerce_parameter_vector(self.center, dtype=None, device=None)
        window_width = _coerce_parameter_vector(self.window_width, dtype=center.dtype, device=center.device)
        wavenumber = _coerce_parameter_vector(
            self.wavenumber,
            dtype=center.dtype,
            device=center.device,
        )
        phase = _coerce_parameter_vector(self.phase, dtype=center.dtype, device=center.device)

        if not (center.shape == window_width.shape == wavenumber.shape == phase.shape):
            raise ValueError("center, window_width, wavenumber, and phase must have matching shapes")
        if not torch.all(window_width > 0).item():
            raise ValueError("window widths must be positive")

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "window_width", window_width)
        object.__setattr__(self, "wavenumber", wavenumber)
        object.__setattr__(self, "phase", phase)

    @property
    def packet_count(self) -> int:
        return int(self.center.shape[0])

    def _support_edges(self) -> tuple[Tensor, Tensor]:
        half_width = 0.5 * self.window_width
        return self.center - half_width, self.center + half_width

    def _full_support_mass(self) -> Tensor:
        return 3.0 * self.window_width / 8.0

    def _normalized_coordinate(self, x: Tensor) -> Tensor:
        return 2.0 * (x - self.center[:, None]) / self.window_width[:, None]

    def _density_antiderivative(self, u: Tensor) -> Tensor:
        pi = torch.tensor(torch.pi, dtype=u.dtype, device=u.device)
        return (
            (3.0 * u / 8.0)
            + torch.sin(pi * u) / (2.0 * pi)
            + torch.sin(2.0 * pi * u) / (16.0 * pi)
        )

    def _clipped_support_mass(self, domain: InfiniteWell1D) -> Tensor:
        support_left, support_right = self._support_edges()
        clipped_left = torch.maximum(support_left, domain.left)
        clipped_right = torch.minimum(support_right, domain.right)
        zero = torch.zeros_like(clipped_left)
        valid = clipped_right > clipped_left
        left_u = 2.0 * (clipped_left - self.center) / self.window_width
        right_u = 2.0 * (clipped_right - self.center) / self.window_width
        clipped_mass = 0.5 * self.window_width * (
            self._density_antiderivative(right_u) - self._density_antiderivative(left_u)
        )
        return torch.where(valid, clipped_mass, zero)

    def component_normalizations(self, domain: InfiniteWell1D) -> Tensor:
        interval_mass = self._clipped_support_mass(domain)
        if torch.any(interval_mass <= 0).item():
            raise ValueError("windowed plane-wave support must overlap the bounded domain")
        scale = torch.rsqrt(
            torch.clamp(
                interval_mass,
                min=torch.finfo(domain.real_dtype).tiny,
            )
        )
        return scale.to(dtype=domain.complex_dtype)

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> "WindowedPlaneWavePacketParameters":
        return WindowedPlaneWavePacketParameters(
            center=self.center.to(dtype=dtype, device=device),
            window_width=self.window_width.to(dtype=dtype, device=device),
            wavenumber=self.wavenumber.to(dtype=dtype, device=device),
            phase=self.phase.to(dtype=dtype, device=device),
        )

    def component_wavefunctions(
        self,
        domain: InfiniteWell1D,
        x,
        *,
        normalize: bool = True,
    ) -> Tensor:
        grid = coerce_tensor(x, dtype=domain.real_dtype, device=domain.device)
        flat_grid = grid.reshape(-1)
        normalized_coordinate = self._normalized_coordinate(flat_grid)
        within_window = torch.abs(normalized_coordinate) <= 1.0
        envelope = torch.where(
            within_window,
            torch.cos(0.5 * torch.pi * normalized_coordinate) ** 2,
            torch.zeros_like(normalized_coordinate),
        )
        oscillation = torch.exp(
            1j * (
                self.wavenumber[:, None] * flat_grid[None, :]
                + self.phase[:, None]
            )
        ).to(dtype=domain.complex_dtype)
        components = envelope.to(dtype=domain.complex_dtype) * oscillation
        support = _strict_domain_support(domain, flat_grid)[None, :]
        components = torch.where(support, components, torch.zeros_like(components))
        if normalize:
            components = self.component_normalizations(domain)[:, None] * components
        return components.reshape(self.packet_count, *grid.shape)

    def support_diagnostics(self, domain: InfiniteWell1D) -> PacketSupportDiagnostics:
        full_mass = self._full_support_mass()
        inside_mass = torch.clamp(self._clipped_support_mass(domain) / full_mass, min=0.0, max=1.0)
        outside_mass = torch.clamp(1.0 - inside_mass, min=0.0, max=1.0)
        left_u = 2.0 * (domain.left - self.center) / self.window_width
        right_u = 2.0 * (domain.right - self.center) / self.window_width
        left_density = torch.where(
            torch.abs(left_u) <= 1.0,
            (torch.cos(0.5 * torch.pi * left_u) ** 4) / full_mass,
            torch.zeros_like(left_u),
        )
        right_density = torch.where(
            torch.abs(right_u) <= 1.0,
            (torch.cos(0.5 * torch.pi * right_u) ** 4) / full_mass,
            torch.zeros_like(right_u),
        )
        return PacketSupportDiagnostics(
            inside_probability_mass=inside_mass,
            outside_probability_mass=outside_mass,
            left_boundary_density=left_density,
            right_boundary_density=right_density,
            boundary_density_mismatch=left_density + right_density,
        )

    @classmethod
    def single(
        cls,
        *,
        center,
        window_width,
        wavenumber,
        phase=0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "WindowedPlaneWavePacketParameters":
        center_tensor = coerce_tensor(center, dtype=dtype, device=device)
        width_tensor = coerce_tensor(window_width, dtype=dtype, device=device)
        wavenumber_tensor = coerce_tensor(wavenumber, dtype=dtype, device=device)
        phase_tensor = coerce_tensor(phase, dtype=dtype, device=device)
        return cls(
            center=torch.atleast_1d(center_tensor),
            window_width=torch.atleast_1d(width_tensor),
            wavenumber=torch.atleast_1d(wavenumber_tensor),
            phase=torch.atleast_1d(phase_tensor),
        )


@dataclass(frozen=True, slots=True)
class PacketState:
    domain: InfiniteWell1D
    parameters: PacketParameterization
    weights: Tensor | None = None
    normalize_components: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, PacketParameterization):
            raise TypeError("packet parameters must implement the packet parameterization protocol")
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

    def component_wavefunctions(self, x) -> Tensor:
        return self.parameters.component_wavefunctions(
            self.domain,
            x,
            normalize=self.normalize_components,
        )

    def support_diagnostics(self) -> PacketSupportDiagnostics:
        return self.parameters.support_diagnostics(self.domain)

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


def make_plane_wave_packet(
    domain: InfiniteWell1D,
    *,
    wavenumber,
    phase=0.0,
    weight=1.0 + 0.0j,
) -> PacketState:
    parameters = PlaneWavePacketParameters.single(
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


def make_windowed_plane_wave_packet(
    domain: InfiniteWell1D,
    *,
    center,
    window_width,
    wavenumber,
    phase=0.0,
    weight=1.0 + 0.0j,
) -> PacketState:
    parameters = WindowedPlaneWavePacketParameters.single(
        center=center,
        window_width=window_width,
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


def make_box_mode_spectral_state(
    domain: InfiniteWell1D,
    *,
    mode_index: int,
    num_modes: int | None = None,
    weight=1.0 + 0.0j,
) -> SpectralState:
    resolved_num_modes = int(mode_index if num_modes is None else num_modes)
    if int(mode_index) < 1:
        raise ValueError("mode_index must be at least 1")
    if resolved_num_modes < int(mode_index):
        raise ValueError("num_modes must be at least mode_index")
    coefficients = torch.zeros(
        resolved_num_modes,
        dtype=domain.complex_dtype,
        device=domain.device,
    )
    coefficients[int(mode_index) - 1] = coerce_scalar_tensor(
        torch.as_tensor(weight, dtype=domain.complex_dtype, device=domain.device),
        dtype=domain.complex_dtype,
        device=domain.device,
    )
    return SpectralState(domain=domain, coefficients=coefficients)


__all__ = [
    "GaussianPacketParameters",
    "PacketParameterization",
    "PacketSupportDiagnostics",
    "PacketState",
    "PlaneWavePacketParameters",
    "SpectralState",
    "WindowedPlaneWavePacketParameters",
    "make_box_mode_spectral_state",
    "make_plane_wave_packet",
    "make_truncated_gaussian_packet",
    "make_windowed_plane_wave_packet",
]
