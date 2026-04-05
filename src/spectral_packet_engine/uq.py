from __future__ import annotations

"""Shared local uncertainty and identifiability utilities.

This module keeps the reusable local-Gaussian inference logic separate from
packet-specific inverse estimation and potential-family calibration wrappers.
"""

from dataclasses import dataclass

import torch


def softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
    return clamped + torch.log(-torch.expm1(-clamped))


def real_view(value: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(value):
        return torch.stack((value.real, value.imag), dim=-1)
    return value


def flatten_real_view(value: torch.Tensor) -> torch.Tensor:
    return real_view(value).reshape(-1)


def _central_normal_quantile(
    confidence_level: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    level = torch.as_tensor(confidence_level, dtype=dtype, device=device)
    sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    return sqrt_two * torch.erfinv(level)


def covariance_to_correlation(covariance: torch.Tensor) -> torch.Tensor:
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))
    denominator = standard_deviation[:, None] * standard_deviation[None, :]
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    correlation = covariance / safe_denominator
    return torch.where(denominator > 0, correlation, torch.zeros_like(correlation))


def linearized_covariance(
    output_jacobian: torch.Tensor,
    parameter_covariance: torch.Tensor,
) -> torch.Tensor:
    jacobian = torch.as_tensor(output_jacobian)
    covariance = jacobian @ parameter_covariance @ jacobian.transpose(0, 1)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coordinates: list[int] = []
    remainder = int(index)
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder //= size
    return tuple(reversed(coordinates))


def _coerce_observation_jacobian(
    observation_jacobian: torch.Tensor,
    *,
    num_parameters: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(
        observation_jacobian,
        dtype=dtype,
        device=device,
    ).reshape(-1, num_parameters)


@dataclass(frozen=True, slots=True)
class PosteriorConfig:
    enabled: bool = True
    noise_scale: float | None = None
    noise_floor: float = 1e-6
    fisher_damping: float = 1e-6
    confidence_level: float = 0.95
    compute_coefficient_posterior: bool = True
    compute_sensitivity: bool = True
    compute_observation_posterior: bool = True
    compute_observation_information: bool = True

    def __post_init__(self) -> None:
        if self.noise_scale is not None and self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive when provided")
        if self.noise_floor <= 0:
            raise ValueError("noise_floor must be positive")
        if self.fisher_damping < 0:
            raise ValueError("fisher_damping must be non-negative")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class ParameterPosteriorSummary:
    parameter_names: tuple[str, ...]
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    confidence_level: float
    confidence_interval_low: torch.Tensor
    confidence_interval_high: torch.Tensor
    covariance: torch.Tensor
    correlation: torch.Tensor
    fisher_information: torch.Tensor
    information_eigenvalues: torch.Tensor
    effective_rank: int
    condition_number: float
    identifiability_score: float
    noise_scale: float
    residual_rms: float


@dataclass(frozen=True, slots=True)
class SensitivityMapSummary:
    parameter_names: tuple[str, ...]
    observation_shape: tuple[int, ...]
    gradient: torch.Tensor
    one_sigma_effect: torch.Tensor
    rms_one_sigma_effect: torch.Tensor
    peak_one_sigma_effect: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObservationPosteriorSummary:
    observation_shape: tuple[int, ...]
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    confidence_level: float
    confidence_interval_low: torch.Tensor
    confidence_interval_high: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObservationInformationSummary:
    observation_shape: tuple[int, ...]
    information_density: torch.Tensor
    normalized_information_density: torch.Tensor
    total_information: float
    effective_observation_count: float
    peak_index: tuple[int, ...]
    peak_information_density: float


def summarize_parameter_posterior(
    *,
    parameter_names: tuple[str, ...],
    mean: torch.Tensor,
    residual_vector: torch.Tensor,
    observation_jacobian: torch.Tensor,
    posterior_config: PosteriorConfig,
) -> ParameterPosteriorSummary:
    mean_vector = torch.as_tensor(mean).reshape(-1)
    residual_flat = torch.as_tensor(
        residual_vector,
        dtype=mean_vector.dtype,
        device=mean_vector.device,
    ).reshape(-1)
    jacobian = torch.as_tensor(
        observation_jacobian,
        dtype=mean_vector.dtype,
        device=mean_vector.device,
    ).reshape(-1, mean_vector.shape[0])
    residual_rms_tensor = torch.sqrt(torch.mean(residual_flat**2))
    if posterior_config.noise_scale is None:
        noise_scale = torch.clamp(
            residual_rms_tensor.detach(),
            min=posterior_config.noise_floor,
        )
    else:
        noise_scale = torch.as_tensor(
            posterior_config.noise_scale,
            dtype=mean_vector.dtype,
            device=mean_vector.device,
        )

    fisher_information = (jacobian.transpose(0, 1) @ jacobian) / (noise_scale**2)
    fisher_information = 0.5 * (fisher_information + fisher_information.transpose(0, 1))
    identity = torch.eye(
        mean_vector.shape[0],
        dtype=fisher_information.dtype,
        device=fisher_information.device,
    )
    precision = fisher_information + posterior_config.fisher_damping * identity
    covariance = torch.linalg.pinv(precision, hermitian=True)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))
    z_score = _central_normal_quantile(
        posterior_config.confidence_level,
        dtype=standard_deviation.dtype,
        device=standard_deviation.device,
    )
    confidence_interval_low = mean_vector - z_score * standard_deviation
    confidence_interval_high = mean_vector + z_score * standard_deviation
    information_eigenvalues = torch.linalg.eigvalsh(fisher_information).detach()
    tolerance = (
        torch.finfo(information_eigenvalues.dtype).eps
        * max(float(torch.max(torch.abs(information_eigenvalues)).item()), 1.0)
        * max(int(mean_vector.numel()), 1)
    )
    positive_eigenvalues = information_eigenvalues[information_eigenvalues > tolerance]
    effective_rank = int(positive_eigenvalues.numel())
    if effective_rank != int(mean_vector.numel()) or effective_rank == 0:
        condition_number = float("inf")
        identifiability_score = 0.0
    else:
        max_eigenvalue = float(torch.max(positive_eigenvalues).item())
        min_eigenvalue = float(torch.min(positive_eigenvalues).item())
        condition_number = max_eigenvalue / min_eigenvalue
        identifiability_score = min_eigenvalue / max_eigenvalue

    return ParameterPosteriorSummary(
        parameter_names=parameter_names,
        mean=mean_vector.detach(),
        standard_deviation=standard_deviation.detach(),
        confidence_level=posterior_config.confidence_level,
        confidence_interval_low=confidence_interval_low.detach(),
        confidence_interval_high=confidence_interval_high.detach(),
        covariance=covariance.detach(),
        correlation=covariance_to_correlation(covariance).detach(),
        fisher_information=fisher_information.detach(),
        information_eigenvalues=information_eigenvalues,
        effective_rank=effective_rank,
        condition_number=condition_number,
        identifiability_score=float(identifiability_score),
        noise_scale=float(noise_scale.detach()),
        residual_rms=float(residual_rms_tensor.detach()),
    )


def summarize_sensitivity_map(
    *,
    parameter_names: tuple[str, ...],
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    parameter_standard_deviation: torch.Tensor,
) -> SensitivityMapSummary:
    observation_shape = tuple(int(size) for size in real_view(observation).shape)
    jacobian = torch.as_tensor(observation_jacobian).reshape(-1, len(parameter_names))
    standard_deviation = torch.as_tensor(
        parameter_standard_deviation,
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).reshape(len(parameter_names))
    gradient = jacobian.transpose(0, 1).reshape(
        len(parameter_names),
        *observation_shape,
    )
    broadcast_shape = (len(parameter_names),) + (1,) * len(observation_shape)
    one_sigma_effect = gradient * standard_deviation.reshape(broadcast_shape)
    flattened_effect = torch.abs(one_sigma_effect).reshape(len(parameter_names), -1)
    return SensitivityMapSummary(
        parameter_names=parameter_names,
        observation_shape=observation_shape,
        gradient=gradient.detach(),
        one_sigma_effect=one_sigma_effect.detach(),
        rms_one_sigma_effect=torch.sqrt(torch.mean(flattened_effect**2, dim=-1)).detach(),
        peak_one_sigma_effect=torch.max(flattened_effect, dim=-1).values.detach(),
    )


def summarize_observation_posterior(
    *,
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    parameter_covariance: torch.Tensor,
    confidence_level: float,
) -> ObservationPosteriorSummary:
    mean = real_view(torch.as_tensor(observation))
    observation_shape = tuple(int(size) for size in mean.shape)
    parameter_covariance_tensor = torch.as_tensor(
        parameter_covariance,
        dtype=mean.dtype,
        device=mean.device,
    )
    covariance = linearized_covariance(
        _coerce_observation_jacobian(
            observation_jacobian,
            num_parameters=int(parameter_covariance_tensor.shape[0]),
            dtype=mean.dtype,
            device=mean.device,
        ),
        parameter_covariance_tensor,
    )
    standard_deviation = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0)).reshape(observation_shape)
    z_score = _central_normal_quantile(
        confidence_level,
        dtype=standard_deviation.dtype,
        device=standard_deviation.device,
    )
    confidence_interval_low = mean - z_score * standard_deviation
    confidence_interval_high = mean + z_score * standard_deviation
    return ObservationPosteriorSummary(
        observation_shape=observation_shape,
        mean=mean.detach(),
        standard_deviation=standard_deviation.detach(),
        confidence_level=confidence_level,
        confidence_interval_low=confidence_interval_low.detach(),
        confidence_interval_high=confidence_interval_high.detach(),
    )


def summarize_observation_information(
    *,
    observation: torch.Tensor,
    observation_jacobian: torch.Tensor,
    noise_scale: float,
) -> ObservationInformationSummary:
    mean = real_view(torch.as_tensor(observation))
    observation_shape = tuple(int(size) for size in mean.shape)
    raw_jacobian = torch.as_tensor(observation_jacobian)
    jacobian = _coerce_observation_jacobian(
        observation_jacobian,
        num_parameters=int(raw_jacobian.shape[-1]),
        dtype=mean.dtype,
        device=mean.device,
    )
    noise = torch.as_tensor(noise_scale, dtype=jacobian.dtype, device=jacobian.device).reshape(())
    if float(noise.item()) <= 0.0:
        raise ValueError("noise_scale must be positive")
    information_density = torch.sum(jacobian**2, dim=-1).reshape(observation_shape) / (noise**2)
    total_information_tensor = torch.sum(information_density)
    total_information = float(total_information_tensor.detach())
    if total_information <= 0.0:
        normalized = torch.zeros_like(information_density)
        effective_observation_count = 0.0
        peak_index = _unravel_index(0, observation_shape)
        peak_information_density = 0.0
    else:
        normalized = information_density / total_information_tensor
        flat_normalized = normalized.reshape(-1)
        positive_mass = flat_normalized[flat_normalized > 0]
        if positive_mass.numel() == 0:
            effective_observation_count = 0.0
        else:
            entropy = -torch.sum(positive_mass * torch.log(positive_mass))
            effective_observation_count = float(torch.exp(entropy).detach())
        peak_flat_index = int(torch.argmax(information_density.reshape(-1)).item())
        peak_index = _unravel_index(peak_flat_index, observation_shape)
        peak_information_density = float(information_density.reshape(-1)[peak_flat_index].detach())
    return ObservationInformationSummary(
        observation_shape=observation_shape,
        information_density=information_density.detach(),
        normalized_information_density=normalized.detach(),
        total_information=total_information,
        effective_observation_count=effective_observation_count,
        peak_index=peak_index,
        peak_information_density=peak_information_density,
    )


__all__ = [
    "ObservationInformationSummary",
    "ObservationPosteriorSummary",
    "ParameterPosteriorSummary",
    "PosteriorConfig",
    "SensitivityMapSummary",
    "covariance_to_correlation",
    "flatten_real_view",
    "linearized_covariance",
    "real_view",
    "softplus_inverse",
    "summarize_observation_information",
    "summarize_observation_posterior",
    "summarize_parameter_posterior",
    "summarize_sensitivity_map",
]
