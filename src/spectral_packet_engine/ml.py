from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import math
import platform
import sys
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np
import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.profiles import (
    profile_mean,
    profile_variance,
    project_profiles_onto_basis,
    reconstruct_profiles_from_basis,
    relative_l2_error,
)
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime, resolve_torch_device
from spectral_packet_engine.tf_surrogate import (
    TensorFlowModalRegressor,
    TensorFlowRegressorConfig,
    configure_tensorflow_runtime,
    inspect_tensorflow_host,
    tensorflow_is_available,
)


ModalBackend = Literal["auto", "torch", "jax", "tensorflow"]
BackendProjectStatus = Literal["stable", "beta", "experimental"]


def jax_is_available() -> bool:
    return importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None


@dataclass(frozen=True, slots=True)
class MLBackendRuntime:
    backend: str
    available: bool
    version: str | None
    system: str
    machine: str
    python_version: str
    accelerator: str
    device_types: tuple[str, ...]
    preferred_device: str | None
    recommended_runtime: str
    package_extra: str | None
    project_status: BackendProjectStatus
    project_supported: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModalBackendResolution:
    requested_backend: str
    resolved_backend: str | None
    available_backends: tuple[str, ...]
    fallback_order: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MLBackendReport:
    requested_device: str
    requested_backend: str
    preferred_backend: str | None
    runtime_available_backends: tuple[str, ...]
    project_supported_backends: tuple[str, ...]
    backends: dict[str, MLBackendRuntime]
    resolution: ModalBackendResolution

    @property
    def available_backends(self) -> tuple[str, ...]:
        return self.runtime_available_backends


def _python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _safe_package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _coerce_requested_device(preferred: str | torch.device | None) -> str:
    if isinstance(preferred, torch.device):
        return preferred.type
    if preferred is None:
        return "auto"
    return str(preferred)


def _project_backend_metadata(
    backend: str,
    *,
    system: str,
    machine: str,
) -> tuple[str | None, BackendProjectStatus, bool, tuple[str, ...]]:
    python_version = (sys.version_info.major, sys.version_info.minor)
    notes: list[str] = []

    if backend == "torch":
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            notes.append("PyTorch is the primary backend for the project on Apple Silicon.")
        else:
            notes.append("PyTorch is the primary backend for the project.")
        return None, "stable", True, tuple(notes)

    if backend == "jax":
        if system == "Windows":
            notes.append(
                "The project does not install JAX on Windows through extras; upstream CPU wheels are experimental."
            )
            return "ml-jax", "beta", False, tuple(notes)
        notes.append("JAX is a beta backend in the project.")
        return "ml-jax", "beta", True, tuple(notes)

    if backend == "tensorflow":
        supported = python_version < (3, 14)
        if not supported:
            notes.append("Project TensorFlow extras are disabled on Python 3.14 and newer.")
        if system == "Darwin":
            notes.append(
                "TensorFlow on macOS is a compatibility path; the official TensorFlow pip page does not provide official macOS GPU support."
            )
        elif system == "Windows":
            notes.append(
                "Native Windows GPU TensorFlow is not a primary path; use CPU or WSL2 for GPU workloads."
            )
        else:
            notes.append("TensorFlow is an experimental compatibility backend in the project.")
        return "ml", "experimental", supported, tuple(notes)

    raise ValueError(f"unsupported backend policy: {backend}")


def _build_backend_resolution(
    backends: Mapping[str, MLBackendRuntime],
    *,
    requested_backend: ModalBackend = "auto",
) -> ModalBackendResolution:
    fallback_order = ("torch", "jax", "tensorflow")
    available_backends = tuple(
        backend
        for backend in fallback_order
        if backends[backend].available
    )
    project_supported_backends = tuple(
        backend
        for backend in available_backends
        if backends[backend].project_supported
    )

    if requested_backend != "auto":
        runtime = backends[str(requested_backend)]
        if runtime.available:
            warnings: list[str] = []
            if not runtime.project_supported:
                warnings.append(
                    f"{requested_backend} is available at runtime but outside the project's primary supported install surface on this platform."
                )
            return ModalBackendResolution(
                requested_backend=str(requested_backend),
                resolved_backend=str(requested_backend),
                available_backends=available_backends,
                fallback_order=fallback_order,
                reason=f"Using explicitly requested backend '{requested_backend}'.",
                warnings=tuple(warnings),
            )
        return ModalBackendResolution(
            requested_backend=str(requested_backend),
            resolved_backend=None,
            available_backends=available_backends,
            fallback_order=fallback_order,
            reason=f"Requested backend '{requested_backend}' is unavailable in this environment.",
            warnings=runtime.notes,
        )

    if project_supported_backends:
        resolved_backend = project_supported_backends[0]
        return ModalBackendResolution(
            requested_backend="auto",
            resolved_backend=resolved_backend,
            available_backends=available_backends,
            fallback_order=fallback_order,
            reason=f"Auto-selected '{resolved_backend}' as the first runtime-available backend within the project's supported surface.",
            warnings=(),
        )

    if available_backends:
        resolved_backend = available_backends[0]
        return ModalBackendResolution(
            requested_backend="auto",
            resolved_backend=resolved_backend,
            available_backends=available_backends,
            fallback_order=fallback_order,
            reason=f"Auto-selected '{resolved_backend}' as the first runtime-available backend.",
            warnings=(
                f"{resolved_backend} is available at runtime but outside the project's primary supported install surface on this platform.",
            ),
        )

    return ModalBackendResolution(
        requested_backend="auto",
        resolved_backend=None,
        available_backends=(),
        fallback_order=fallback_order,
        reason="No ML backend is available in this environment.",
        warnings=(),
    )


@dataclass(frozen=True, slots=True)
class ModalSurrogateConfig:
    profile_hidden_units: tuple[int, ...] = (256, 128)
    time_hidden_units: tuple[int, ...] = (16,)
    residual_blocks: int = 1
    dropout_rate: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 40
    validation_fraction: float = 0.25
    random_seed: int = 42
    early_stopping_patience: int = 8
    coefficient_loss_weight: float = 1.0
    moment_loss_weight: float = 0.25
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    device: str = "auto"
    enable_xla: bool = True
    enable_mixed_precision: bool = True
    jit_compile: bool = True

    def __post_init__(self) -> None:
        if not self.profile_hidden_units:
            raise ValueError("profile_hidden_units must not be empty")
        if any(width <= 0 for width in self.profile_hidden_units):
            raise ValueError("profile_hidden_units must contain positive widths")
        if any(width <= 0 for width in self.time_hidden_units):
            raise ValueError("time_hidden_units must contain positive widths")
        if self.residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must lie in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.coefficient_loss_weight <= 0:
            raise ValueError("coefficient_loss_weight must be positive")
        if self.moment_loss_weight < 0:
            raise ValueError("moment_loss_weight must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")


@dataclass(frozen=True, slots=True)
class ModalRegressionResult:
    backend: str
    runtime: MLBackendRuntime
    train_size: int
    validation_size: int
    history: dict[str, list[float]]
    epochs_ran: int
    best_epoch: int
    best_validation_loss: float
    parameter_count: int
    training_seconds: float
    training_profiles_per_second: float
    validation_prediction_seconds: float
    validation_inference_profiles_per_second: float
    validation_profile_relative_l2: float
    validation_coefficient_mse: float
    validation_moment_mae: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime": self.runtime.to_dict(),
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "history": self.history,
            "epochs_ran": self.epochs_ran,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "parameter_count": self.parameter_count,
            "training_seconds": self.training_seconds,
            "training_profiles_per_second": self.training_profiles_per_second,
            "validation_prediction_seconds": self.validation_prediction_seconds,
            "validation_inference_profiles_per_second": self.validation_inference_profiles_per_second,
            "validation_profile_relative_l2": self.validation_profile_relative_l2,
            "validation_coefficient_mse": self.validation_coefficient_mse,
            "validation_moment_mae": self.validation_moment_mae,
        }


@dataclass(frozen=True, slots=True)
class _PreparedModalData:
    profiles: np.ndarray
    sample_times: np.ndarray
    coefficients: np.ndarray
    moments: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray
    profile_mean: np.ndarray
    profile_scale: np.ndarray
    time_mean: np.ndarray
    time_scale: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_scale: np.ndarray
    moment_mean: np.ndarray
    moment_scale: np.ndarray
    grid: torch.Tensor


class ModalRegressor(Protocol):
    backend: str
    basis: InfiniteWellBasis
    config: ModalSurrogateConfig
    runtime: MLBackendRuntime | None

    def fit(self, density_profiles, position_grid, sample_times=None) -> ModalRegressionResult: ...
    def predict_coefficients(self, density_profiles, sample_times=None) -> np.ndarray: ...
    def predict_moments(self, density_profiles, sample_times=None) -> np.ndarray: ...
    def reconstruct_profiles(self, density_profiles, position_grid, sample_times=None) -> torch.Tensor: ...
    def export(self, path: str | Path) -> Path: ...


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _split_indices(num_samples: int, validation_fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("at least two samples are required for a train/validation split")
    rng = np.random.default_rng(random_seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    validation_size = max(1, int(round(num_samples * validation_fraction)))
    if validation_size >= num_samples:
        validation_size = num_samples - 1
    validation_indices = np.sort(indices[:validation_size])
    training_indices = np.sort(indices[validation_size:])
    return training_indices, validation_indices


def _prepare_modal_data(
    basis: InfiniteWellBasis,
    config: ModalSurrogateConfig,
    density_profiles,
    position_grid,
    sample_times=None,
) -> _PreparedModalData:
    grid = torch.as_tensor(position_grid, dtype=basis.domain.real_dtype)
    profiles = torch.as_tensor(density_profiles, dtype=basis.domain.real_dtype)
    if profiles.ndim != 2:
        raise ValueError("density_profiles must be two-dimensional [sample, position]")
    if sample_times is None:
        sample_times_t = torch.zeros((profiles.shape[0], 1), dtype=basis.domain.real_dtype)
    else:
        sample_times_t = torch.as_tensor(sample_times, dtype=basis.domain.real_dtype).reshape(-1, 1)
        if sample_times_t.shape[0] != profiles.shape[0]:
            raise ValueError("sample_times must match the number of profiles")

    coefficients = project_profiles_onto_basis(profiles, grid, basis)
    mean_position = profile_mean(profiles, grid).reshape(-1, 1)
    width = torch.sqrt(profile_variance(profiles, grid)).reshape(-1, 1)
    moments = torch.cat([mean_position, width], dim=1)

    profile_features = profiles.detach().cpu().numpy().astype(np.float32, copy=False)
    time_features = sample_times_t.detach().cpu().numpy().astype(np.float32, copy=False)
    coefficient_targets = coefficients.detach().cpu().numpy().astype(np.float32, copy=False)
    moment_targets = moments.detach().cpu().numpy().astype(np.float32, copy=False)

    training_indices, validation_indices = _split_indices(
        profile_features.shape[0],
        config.validation_fraction,
        config.random_seed,
    )

    x_train = profile_features[training_indices]
    t_train = time_features[training_indices]
    y_train = coefficient_targets[training_indices]
    m_train = moment_targets[training_indices]

    profile_feature_mean, profile_feature_scale = _standardize(x_train)
    time_feature_mean, time_feature_scale = _standardize(t_train)
    coefficient_feature_mean, coefficient_feature_scale = _standardize(y_train)
    moment_feature_mean, moment_feature_scale = _standardize(m_train)

    return _PreparedModalData(
        profiles=profile_features,
        sample_times=time_features,
        coefficients=coefficient_targets,
        moments=moment_targets,
        training_indices=training_indices,
        validation_indices=validation_indices,
        profile_mean=profile_feature_mean,
        profile_scale=profile_feature_scale,
        time_mean=time_feature_mean,
        time_scale=time_feature_scale,
        coefficient_mean=coefficient_feature_mean,
        coefficient_scale=coefficient_feature_scale,
        moment_mean=moment_feature_mean,
        moment_scale=moment_feature_scale,
        grid=grid.detach().cpu(),
    )


def _validation_metrics(
    basis: InfiniteWellBasis,
    runtime: MLBackendRuntime,
    validation_profiles: np.ndarray,
    predicted_coefficients: np.ndarray,
    predicted_moments: np.ndarray,
    target_coefficients: np.ndarray,
    target_moments: np.ndarray,
    grid: torch.Tensor,
) -> tuple[float, float, float]:
    del runtime
    predicted_profiles = reconstruct_profiles_from_basis(
        torch.as_tensor(predicted_coefficients, dtype=basis.domain.real_dtype),
        grid.to(dtype=basis.domain.real_dtype),
        basis,
    )
    validation_error = relative_l2_error(
        torch.as_tensor(validation_profiles, dtype=basis.domain.real_dtype),
        predicted_profiles,
        grid.to(dtype=basis.domain.real_dtype),
    )
    coefficient_mse = float(np.mean((predicted_coefficients - target_coefficients) ** 2))
    moment_mae = float(np.mean(np.abs(predicted_moments - target_moments)))
    return float(torch.mean(validation_error)), coefficient_mse, moment_mae


def inspect_torch_backend(preferred_device: str | torch.device | None = "auto") -> MLBackendRuntime:
    runtime = inspect_torch_runtime(preferred_device)
    package_extra, project_status, project_supported, policy_notes = _project_backend_metadata(
        "torch",
        system=runtime.system,
        machine=runtime.machine,
    )
    notes = list(policy_notes) + list(runtime.notes)
    if runtime.backend == "mps":
        notes.append("Apple Metal uses float32 for training-oriented workloads.")
    if runtime.backend == "cpu":
        notes.append("CPU execution is the baseline fallback and fully supported.")
    return MLBackendRuntime(
        backend="torch",
        available=True,
        version=runtime.torch_version,
        system=runtime.system,
        machine=runtime.machine,
        python_version=_python_version_string(),
        accelerator=runtime.accelerator,
        device_types=(runtime.backend,),
        preferred_device=str(runtime.device),
        recommended_runtime="Use the core package on CPU, CUDA, or Apple Metal depending on hardware.",
        package_extra=package_extra,
        project_status=project_status,
        project_supported=project_supported,
        notes=tuple(notes),
    )


def inspect_jax_backend(preferred_device: str = "auto") -> MLBackendRuntime:
    system = platform.system()
    machine = platform.machine().lower()
    package_extra, project_status, project_supported, policy_notes = _project_backend_metadata(
        "jax",
        system=system,
        machine=machine,
    )
    if not jax_is_available():
        return MLBackendRuntime(
            backend="jax",
            available=False,
            version=None,
            system=system,
            machine=machine,
            python_version=_python_version_string(),
            accelerator="unavailable",
            device_types=(),
            preferred_device=None,
            recommended_runtime="Install the 'ml-jax' extra in a compatible environment.",
            package_extra=package_extra,
            project_status=project_status,
            project_supported=project_supported,
            notes=policy_notes + ("JAX is not installed.",),
        )
    try:
        import jax
        devices = list(jax.devices())
    except Exception as exc:
        return MLBackendRuntime(
            backend="jax",
            available=False,
            version=_safe_package_version("jax"),
            system=system,
            machine=machine,
            python_version=_python_version_string(),
            accelerator="unavailable",
            device_types=(),
            preferred_device=None,
            recommended_runtime="Install the 'ml-jax' extra in a clean environment with compatible JAX and JAXLIB wheels.",
            package_extra=package_extra,
            project_status=project_status,
            project_supported=project_supported,
            notes=policy_notes + (f"JAX import failed: {type(exc).__name__}: {exc}",),
        )

    if not devices:
        return MLBackendRuntime(
            backend="jax",
            available=False,
            version=jax.__version__,
            system=system,
            machine=machine,
            python_version=_python_version_string(),
            accelerator="unavailable",
            device_types=(),
            preferred_device=None,
            recommended_runtime="Ensure JAX can see a compatible backend device.",
            package_extra=package_extra,
            project_status=project_status,
            project_supported=project_supported,
            notes=policy_notes + ("JAX is installed but reports no devices.",),
        )
    preferred = preferred_device.lower()
    selected = devices[0]
    if preferred != "auto":
        for device in devices:
            if device.platform == preferred:
                selected = device
                break
        else:
            raise RuntimeError(f"JAX device '{preferred_device}' is not available")
    else:
        for token in ("gpu", "tpu", "cpu"):
            for device in devices:
                if device.platform == token:
                    selected = device
                    break
            if selected.platform == token:
                break
    notes = list(policy_notes)
    if selected.platform == "cpu":
        notes.append("JAX is running on CPU in this environment.")
    return MLBackendRuntime(
        backend="jax",
        available=True,
        version=jax.__version__,
        system=system,
        machine=machine,
        python_version=_python_version_string(),
        accelerator=getattr(selected, "device_kind", selected.platform),
        device_types=tuple(sorted({device.platform for device in devices})),
        preferred_device=selected.platform,
        recommended_runtime="Use JAX for CPU/GPU/XLA-oriented surrogate experiments when the platform is supported.",
        package_extra=package_extra,
        project_status=project_status,
        project_supported=project_supported,
        notes=tuple(notes),
    )


def inspect_tensorflow_backend() -> MLBackendRuntime:
    host = inspect_tensorflow_host()
    package_extra, project_status, project_supported, policy_notes = _project_backend_metadata(
        "tensorflow",
        system=host.system,
        machine=host.machine,
    )
    if not tensorflow_is_available():
        return MLBackendRuntime(
            backend="tensorflow",
            available=False,
            version=None,
            system=host.system,
            machine=host.machine,
            python_version=host.python_version,
            accelerator="unavailable",
            device_types=(),
            preferred_device=None,
            recommended_runtime=host.recommended_runtime,
            package_extra=package_extra,
            project_status=project_status,
            project_supported=project_supported,
            notes=policy_notes + ("TensorFlow is not installed.",),
        )
    try:
        runtime = configure_tensorflow_runtime(
            enable_memory_growth=False,
            enable_xla=False,
            enable_mixed_precision=False,
        )
    except Exception as exc:
        return MLBackendRuntime(
            backend="tensorflow",
            available=False,
            version=_safe_package_version("tensorflow"),
            system=host.system,
            machine=host.machine,
            python_version=host.python_version,
            accelerator="unavailable",
            device_types=(),
            preferred_device=None,
            recommended_runtime=host.recommended_runtime,
            package_extra=package_extra,
            project_status=project_status,
            project_supported=project_supported,
            notes=policy_notes + (f"TensorFlow runtime inspection failed: {type(exc).__name__}: {exc}",),
        )
    accelerator = runtime.gpu_devices[0] if runtime.gpu_devices else runtime.host.recommended_accelerator
    notes = list(policy_notes)
    if not runtime.gpu_devices:
        notes.append("TensorFlow is available, but no GPU devices were detected.")
    return MLBackendRuntime(
        backend="tensorflow",
        available=True,
        version=runtime.version,
        system=runtime.host.system,
        machine=runtime.host.machine,
        python_version=runtime.host.python_version,
        accelerator=accelerator,
        device_types=runtime.visible_device_types,
        preferred_device=runtime.visible_device_types[0] if runtime.visible_device_types else None,
        recommended_runtime=runtime.host.recommended_runtime,
        package_extra=package_extra,
        project_status=project_status,
        project_supported=project_supported,
        notes=tuple(notes),
    )


def inspect_ml_backends(
    preferred_torch_device: str | torch.device | None = "auto",
    *,
    requested_backend: ModalBackend = "auto",
) -> MLBackendReport:
    requested_device = _coerce_requested_device(preferred_torch_device)
    backends = {
        "torch": inspect_torch_backend(preferred_torch_device),
        "jax": inspect_jax_backend(requested_device),
        "tensorflow": inspect_tensorflow_backend(),
    }
    runtime_available_backends = tuple(
        backend_name
        for backend_name, runtime in backends.items()
        if runtime.available
    )
    project_supported_backends = tuple(
        backend_name
        for backend_name, runtime in backends.items()
        if runtime.available and runtime.project_supported
    )
    preferred_backend = _build_backend_resolution(backends, requested_backend="auto").resolved_backend
    resolution = _build_backend_resolution(backends, requested_backend=requested_backend)
    return MLBackendReport(
        requested_device=requested_device,
        requested_backend=str(requested_backend),
        preferred_backend=preferred_backend,
        runtime_available_backends=runtime_available_backends,
        project_supported_backends=project_supported_backends,
        backends=backends,
        resolution=resolution,
    )


def resolve_modal_backend(preferred: ModalBackend = "auto") -> str:
    report = inspect_ml_backends(requested_backend=preferred)
    resolution = report.resolution
    if resolution.resolved_backend is None:
        if preferred != "auto":
            runtime = report.backends[str(preferred)]
            raise ModuleNotFoundError(
                f"The requested ML backend '{preferred}' is unavailable. {runtime.recommended_runtime}"
            )
        raise ModuleNotFoundError("No supported ML backend is available in this environment.")
    return resolution.resolved_backend


def _to_tensorflow_config(config: ModalSurrogateConfig) -> TensorFlowRegressorConfig:
    return TensorFlowRegressorConfig(
        profile_hidden_units=config.profile_hidden_units,
        time_hidden_units=config.time_hidden_units,
        residual_blocks=config.residual_blocks,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_fraction=config.validation_fraction,
        random_seed=config.random_seed,
        early_stopping_patience=config.early_stopping_patience,
        coefficient_loss_weight=config.coefficient_loss_weight,
        moment_loss_weight=config.moment_loss_weight,
        enable_xla=config.enable_xla,
        enable_mixed_precision=config.enable_mixed_precision,
    )


class TensorFlowModalRegressorAdapter:
    backend = "tensorflow"

    def __init__(self, basis: InfiniteWellBasis, *, config: ModalSurrogateConfig = ModalSurrogateConfig()) -> None:
        self.basis = basis
        self.config = config
        self.runtime: MLBackendRuntime | None = None
        self._regressor = TensorFlowModalRegressor(basis, config=_to_tensorflow_config(config))

    def fit(self, density_profiles, position_grid, sample_times=None) -> ModalRegressionResult:
        result = self._regressor.fit(density_profiles, position_grid, sample_times=sample_times)
        self.runtime = inspect_tensorflow_backend()
        return ModalRegressionResult(
            backend="tensorflow",
            runtime=self.runtime,
            train_size=result.train_size,
            validation_size=result.validation_size,
            history=result.history,
            epochs_ran=result.epochs_ran,
            best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            parameter_count=result.parameter_count,
            training_seconds=result.training_seconds,
            training_profiles_per_second=result.training_profiles_per_second,
            validation_prediction_seconds=result.validation_prediction_seconds,
            validation_inference_profiles_per_second=result.validation_inference_profiles_per_second,
            validation_profile_relative_l2=result.validation_profile_relative_l2,
            validation_coefficient_mse=result.validation_coefficient_mse,
            validation_moment_mae=result.validation_moment_mae,
        )

    def predict_coefficients(self, density_profiles, sample_times=None) -> np.ndarray:
        return self._regressor.predict_coefficients(density_profiles, sample_times=sample_times)

    def predict_moments(self, density_profiles, sample_times=None) -> np.ndarray:
        return self._regressor.predict_moments(density_profiles, sample_times=sample_times)

    def reconstruct_profiles(self, density_profiles, position_grid, sample_times=None) -> torch.Tensor:
        return self._regressor.reconstruct_profiles(density_profiles, position_grid, sample_times=sample_times)

    def export(self, path: str | Path) -> Path:
        return self._regressor.export(path)


class _PyTorchResidualBlock(torch.nn.Module):
    def __init__(self, width: int, dropout_rate: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.linear1 = torch.nn.Linear(width, width)
        self.linear2 = torch.nn.Linear(width, width)
        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate > 0 else torch.nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.norm(inputs)
        outputs = torch.nn.functional.gelu(self.linear1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        return torch.nn.functional.gelu(residual + outputs)


class _PyTorchModalNetwork(torch.nn.Module):
    def __init__(self, profile_dim: int, num_modes: int, config: ModalSurrogateConfig) -> None:
        super().__init__()
        profile_layers: list[torch.nn.Module] = []
        in_features = profile_dim
        for width in config.profile_hidden_units:
            profile_layers.extend(
                [
                    torch.nn.Linear(in_features, width),
                    torch.nn.LayerNorm(width),
                    torch.nn.GELU(),
                ]
            )
            in_features = width
        self.profile_branch = torch.nn.Sequential(*profile_layers)

        time_layers: list[torch.nn.Module] = []
        in_features = 1
        for width in config.time_hidden_units:
            time_layers.extend(
                [
                    torch.nn.Linear(in_features, width),
                    torch.nn.GELU(),
                ]
            )
            in_features = width
        self.time_branch = torch.nn.Sequential(*time_layers) if time_layers else torch.nn.Identity()
        trunk_width = config.profile_hidden_units[-1]
        self.trunk = torch.nn.Linear(trunk_width + (config.time_hidden_units[-1] if config.time_hidden_units else 1), trunk_width)
        self.blocks = torch.nn.ModuleList(
            [_PyTorchResidualBlock(trunk_width, config.dropout_rate) for _ in range(config.residual_blocks)]
        )
        self.coefficient_head = torch.nn.Linear(trunk_width, num_modes)
        self.moment_head = torch.nn.Linear(trunk_width, 2)

    def forward(self, profile_inputs: torch.Tensor, time_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        profile_features = self.profile_branch(profile_inputs)
        time_features = self.time_branch(time_inputs)
        combined = torch.cat([profile_features, time_features], dim=-1)
        trunk = torch.nn.functional.gelu(self.trunk(combined))
        for block in self.blocks:
            trunk = block(trunk)
        return self.coefficient_head(trunk), self.moment_head(trunk)


class PyTorchModalRegressor:
    backend = "torch"

    def __init__(self, basis: InfiniteWellBasis, *, config: ModalSurrogateConfig = ModalSurrogateConfig()) -> None:
        self.basis = basis
        self.config = config
        self.runtime: MLBackendRuntime | None = None
        self.model: _PyTorchModalNetwork | None = None
        self.profile_mean: np.ndarray | None = None
        self.profile_scale: np.ndarray | None = None
        self.time_mean: np.ndarray | None = None
        self.time_scale: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None
        self.moment_mean: np.ndarray | None = None
        self.moment_scale: np.ndarray | None = None
        self._device: torch.device | None = None

    def fit(self, density_profiles, position_grid, sample_times=None) -> ModalRegressionResult:
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        prepared = _prepare_modal_data(self.basis, self.config, density_profiles, position_grid, sample_times)
        runtime = inspect_torch_backend(self.config.device)
        self.runtime = runtime
        self._device = resolve_torch_device(self.config.device)

        self.profile_mean = prepared.profile_mean
        self.profile_scale = prepared.profile_scale
        self.time_mean = prepared.time_mean
        self.time_scale = prepared.time_scale
        self.coefficient_mean = prepared.coefficient_mean
        self.coefficient_scale = prepared.coefficient_scale
        self.moment_mean = prepared.moment_mean
        self.moment_scale = prepared.moment_scale

        x_scaled = (prepared.profiles - self.profile_mean) / self.profile_scale
        t_scaled = (prepared.sample_times - self.time_mean) / self.time_scale
        y_scaled = (prepared.coefficients - self.coefficient_mean) / self.coefficient_scale
        m_scaled = (prepared.moments - self.moment_mean) / self.moment_scale

        train_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_scaled[prepared.training_indices], dtype=torch.float32),
            torch.as_tensor(t_scaled[prepared.training_indices], dtype=torch.float32),
            torch.as_tensor(y_scaled[prepared.training_indices], dtype=torch.float32),
            torch.as_tensor(m_scaled[prepared.training_indices], dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.random_seed),
        )

        val_profile = torch.as_tensor(x_scaled[prepared.validation_indices], dtype=torch.float32, device=self._device)
        val_time = torch.as_tensor(t_scaled[prepared.validation_indices], dtype=torch.float32, device=self._device)
        val_coefficients = torch.as_tensor(y_scaled[prepared.validation_indices], dtype=torch.float32, device=self._device)
        val_moments = torch.as_tensor(m_scaled[prepared.validation_indices], dtype=torch.float32, device=self._device)

        self.model = _PyTorchModalNetwork(prepared.profiles.shape[1], self.basis.num_modes, self.config).to(self._device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        mse = torch.nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        training_start = perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            samples_seen = 0
            for batch_profiles, batch_times, batch_coefficients, batch_moments in train_loader:
                batch_profiles = batch_profiles.to(self._device)
                batch_times = batch_times.to(self._device)
                batch_coefficients = batch_coefficients.to(self._device)
                batch_moments = batch_moments.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                pred_coefficients, pred_moments = self.model(batch_profiles, batch_times)
                loss = (
                    self.config.coefficient_loss_weight * mse(pred_coefficients, batch_coefficients)
                    + self.config.moment_loss_weight * mse(pred_moments, batch_moments)
                )
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                batch_size = int(batch_profiles.shape[0])
                running_loss += float(loss.detach()) * batch_size
                samples_seen += batch_size

            self.model.eval()
            with torch.no_grad():
                val_pred_coefficients, val_pred_moments = self.model(val_profile, val_time)
                val_loss = (
                    self.config.coefficient_loss_weight * mse(val_pred_coefficients, val_coefficients)
                    + self.config.moment_loss_weight * mse(val_pred_moments, val_moments)
                )
            epoch_loss = running_loss / max(samples_seen, 1)
            epoch_val_loss = float(val_loss.detach().cpu())
            history["loss"].append(epoch_loss)
            history["val_loss"].append(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break

        training_seconds = perf_counter() - training_start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        epochs_ran = len(history["loss"])

        prediction_start = perf_counter()
        predicted_coefficients = self.predict_coefficients(
            prepared.profiles[prepared.validation_indices],
            sample_times=prepared.sample_times[prepared.validation_indices].reshape(-1),
        )
        predicted_moments = self.predict_moments(
            prepared.profiles[prepared.validation_indices],
            sample_times=prepared.sample_times[prepared.validation_indices].reshape(-1),
        )
        prediction_seconds = perf_counter() - prediction_start

        validation_profile_relative_l2, validation_coefficient_mse, validation_moment_mae = _validation_metrics(
            self.basis,
            runtime,
            prepared.profiles[prepared.validation_indices],
            predicted_coefficients,
            predicted_moments,
            prepared.coefficients[prepared.validation_indices],
            prepared.moments[prepared.validation_indices],
            prepared.grid,
        )

        train_size = int(prepared.training_indices.shape[0])
        validation_size = int(prepared.validation_indices.shape[0])
        return ModalRegressionResult(
            backend="torch",
            runtime=runtime,
            train_size=train_size,
            validation_size=validation_size,
            history=history,
            epochs_ran=epochs_ran,
            best_epoch=best_epoch,
            best_validation_loss=float(best_loss),
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            training_seconds=float(training_seconds),
            training_profiles_per_second=float((train_size * max(epochs_ran, 1)) / max(training_seconds, 1e-9)),
            validation_prediction_seconds=float(prediction_seconds),
            validation_inference_profiles_per_second=float(validation_size / max(prediction_seconds, 1e-9)),
            validation_profile_relative_l2=validation_profile_relative_l2,
            validation_coefficient_mse=validation_coefficient_mse,
            validation_moment_mae=validation_moment_mae,
        )

    def _check_ready(self) -> None:
        if self.model is None or self._device is None:
            raise RuntimeError("fit must be called before prediction")
        if any(
            value is None
            for value in (
                self.profile_mean,
                self.profile_scale,
                self.time_mean,
                self.time_scale,
                self.coefficient_mean,
                self.coefficient_scale,
                self.moment_mean,
                self.moment_scale,
            )
        ):
            raise RuntimeError("fit must be called before prediction")

    def _scaled_inputs(self, density_profiles, sample_times=None) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_ready()
        profile_array = np.asarray(density_profiles, dtype=np.float32)
        if profile_array.ndim == 1:
            profile_array = profile_array[None, :]
        if sample_times is None:
            time_array = np.zeros((profile_array.shape[0], 1), dtype=np.float32)
        else:
            time_array = np.asarray(sample_times, dtype=np.float32).reshape(-1, 1)
            if time_array.shape[0] != profile_array.shape[0]:
                raise ValueError("sample_times must match the number of profiles")
        profile_inputs = (profile_array - self.profile_mean) / self.profile_scale
        time_inputs = (time_array - self.time_mean) / self.time_scale
        return (
            torch.as_tensor(profile_inputs, dtype=torch.float32, device=self._device),
            torch.as_tensor(time_inputs, dtype=torch.float32, device=self._device),
        )

    def predict_coefficients(self, density_profiles, sample_times=None) -> np.ndarray:
        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        self.model.eval()
        with torch.no_grad():
            coefficients, _ = self.model(profile_inputs, time_inputs)
        output = coefficients.detach().cpu().numpy() * self.coefficient_scale + self.coefficient_mean
        return np.asarray(output, dtype=np.float32)

    def predict_moments(self, density_profiles, sample_times=None) -> np.ndarray:
        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        self.model.eval()
        with torch.no_grad():
            _, moments = self.model(profile_inputs, time_inputs)
        output = moments.detach().cpu().numpy() * self.moment_scale + self.moment_mean
        return np.asarray(output, dtype=np.float32)

    def reconstruct_profiles(self, density_profiles, position_grid, sample_times=None) -> torch.Tensor:
        predicted_coefficients = self.predict_coefficients(density_profiles, sample_times=sample_times)
        return reconstruct_profiles_from_basis(
            torch.as_tensor(predicted_coefficients, dtype=self.basis.domain.real_dtype),
            torch.as_tensor(position_grid, dtype=self.basis.domain.real_dtype),
            self.basis,
        )

    def export(self, path: str | Path) -> Path:
        self._check_ready()
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": "torch",
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "profile_mean": self.profile_mean,
                "profile_scale": self.profile_scale,
                "time_mean": self.time_mean,
                "time_scale": self.time_scale,
                "coefficient_mean": self.coefficient_mean,
                "coefficient_scale": self.coefficient_scale,
                "moment_mean": self.moment_mean,
                "moment_scale": self.moment_scale,
            },
            export_path,
        )
        return export_path


def _jax_initialize_parameters(
    input_dim: int,
    time_dim: int,
    num_modes: int,
    config: ModalSurrogateConfig,
    *,
    random_seed: int,
):
    import jax
    import jax.numpy as jnp

    def init_linear(key, in_dim: int, out_dim: int):
        limit = math.sqrt(2.0 / max(in_dim, 1))
        weights = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * limit
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        return {"w": weights, "b": bias}

    def split_many(key, count: int):
        return jax.random.split(key, count + 1)[1:]

    key = jax.random.PRNGKey(random_seed)
    keys = list(split_many(key, len(config.profile_hidden_units) + len(config.time_hidden_units) + 2 * config.residual_blocks + 3))
    index = 0

    profile_layers = []
    current_dim = input_dim
    for width in config.profile_hidden_units:
        profile_layers.append(init_linear(keys[index], current_dim, width))
        current_dim = width
        index += 1

    time_layers = []
    current_time_dim = time_dim
    for width in config.time_hidden_units:
        time_layers.append(init_linear(keys[index], current_time_dim, width))
        current_time_dim = width
        index += 1

    trunk_width = config.profile_hidden_units[-1]
    trunk_input_dim = trunk_width + (config.time_hidden_units[-1] if config.time_hidden_units else time_dim)
    trunk_input = init_linear(keys[index], trunk_input_dim, trunk_width)
    index += 1

    residual_blocks = []
    for _ in range(config.residual_blocks):
        block = (
            init_linear(keys[index], trunk_width, trunk_width),
            init_linear(keys[index + 1], trunk_width, trunk_width),
        )
        residual_blocks.append(block)
        index += 2

    coefficient_head = init_linear(keys[index], trunk_width, num_modes)
    moment_head = init_linear(keys[index + 1], trunk_width, 2)
    return {
        "profile_layers": tuple(profile_layers),
        "time_layers": tuple(time_layers),
        "trunk_input": trunk_input,
        "residual_blocks": tuple(residual_blocks),
        "coefficient_head": coefficient_head,
        "moment_head": moment_head,
    }


def _jax_forward(parameters, profile_inputs, time_inputs, *, training: bool, dropout_rate: float, rng_key=None):
    import jax
    import jax.numpy as jnp

    def linear(inputs, layer):
        return jnp.dot(inputs, layer["w"]) + layer["b"]

    profile = profile_inputs
    for layer in parameters["profile_layers"]:
        profile = jax.nn.gelu(linear(profile, layer))

    time_branch = time_inputs
    for layer in parameters["time_layers"]:
        time_branch = jax.nn.gelu(linear(time_branch, layer))

    trunk = jax.nn.gelu(linear(jnp.concatenate([profile, time_branch], axis=-1), parameters["trunk_input"]))
    residual_key = rng_key
    for first_layer, second_layer in parameters["residual_blocks"]:
        block = jax.nn.gelu(linear(trunk, first_layer))
        if training and dropout_rate > 0:
            if residual_key is None:
                raise RuntimeError("rng_key is required when dropout is enabled")
            residual_key, block_key = jax.random.split(residual_key)
            keep_probability = 1.0 - dropout_rate
            mask = jax.random.bernoulli(block_key, keep_probability, block.shape)
            block = jnp.where(mask, block / keep_probability, 0.0)
        block = linear(block, second_layer)
        trunk = jax.nn.gelu(trunk + block)

    coefficients = linear(trunk, parameters["coefficient_head"])
    moments = linear(trunk, parameters["moment_head"])
    return coefficients, moments


class JAXModalRegressor:
    backend = "jax"

    def __init__(self, basis: InfiniteWellBasis, *, config: ModalSurrogateConfig = ModalSurrogateConfig()) -> None:
        self.basis = basis
        self.config = config
        self.runtime: MLBackendRuntime | None = None
        self.parameters: Any = None
        self.profile_mean: np.ndarray | None = None
        self.profile_scale: np.ndarray | None = None
        self.time_mean: np.ndarray | None = None
        self.time_scale: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None
        self.moment_mean: np.ndarray | None = None
        self.moment_scale: np.ndarray | None = None

    def fit(self, density_profiles, position_grid, sample_times=None) -> ModalRegressionResult:
        if not jax_is_available():
            raise ModuleNotFoundError("JAX is not installed. Install the 'ml-jax' extra.")

        import jax
        import jax.numpy as jnp
        from jax.example_libraries import optimizers

        prepared = _prepare_modal_data(self.basis, self.config, density_profiles, position_grid, sample_times)
        self.runtime = inspect_jax_backend()

        self.profile_mean = prepared.profile_mean
        self.profile_scale = prepared.profile_scale
        self.time_mean = prepared.time_mean
        self.time_scale = prepared.time_scale
        self.coefficient_mean = prepared.coefficient_mean
        self.coefficient_scale = prepared.coefficient_scale
        self.moment_mean = prepared.moment_mean
        self.moment_scale = prepared.moment_scale

        x_scaled = (prepared.profiles - self.profile_mean) / self.profile_scale
        t_scaled = (prepared.sample_times - self.time_mean) / self.time_scale
        y_scaled = (prepared.coefficients - self.coefficient_mean) / self.coefficient_scale
        m_scaled = (prepared.moments - self.moment_mean) / self.moment_scale

        self.parameters = _jax_initialize_parameters(
            x_scaled.shape[1],
            t_scaled.shape[1],
            self.basis.num_modes,
            self.config,
            random_seed=self.config.random_seed,
        )
        optimizer_init, optimizer_update, get_params = optimizers.adam(self.config.learning_rate)
        opt_state = optimizer_init(self.parameters)

        def loss_fn(params, batch_profiles, batch_times, batch_coefficients, batch_moments, rng_key):
            pred_coefficients, pred_moments = _jax_forward(
                params,
                batch_profiles,
                batch_times,
                training=True,
                dropout_rate=self.config.dropout_rate,
                rng_key=rng_key,
            )
            coefficient_loss = jnp.mean((pred_coefficients - batch_coefficients) ** 2)
            moment_loss = jnp.mean((pred_moments - batch_moments) ** 2)
            return (
                self.config.coefficient_loss_weight * coefficient_loss
                + self.config.moment_loss_weight * moment_loss
            )

        @jax.jit
        def train_step(step_index, state, batch_profiles, batch_times, batch_coefficients, batch_moments, rng_key):
            params = get_params(state)
            gradients = jax.grad(loss_fn)(params, batch_profiles, batch_times, batch_coefficients, batch_moments, rng_key)
            return optimizer_update(step_index, gradients, state)

        def evaluate_loss(params, batch_profiles, batch_times, batch_coefficients, batch_moments):
            pred_coefficients, pred_moments = _jax_forward(
                params,
                batch_profiles,
                batch_times,
                training=False,
                dropout_rate=0.0,
            )
            coefficient_loss = np.mean((np.asarray(pred_coefficients) - np.asarray(batch_coefficients)) ** 2)
            moment_loss = np.mean((np.asarray(pred_moments) - np.asarray(batch_moments)) ** 2)
            return float(
                self.config.coefficient_loss_weight * coefficient_loss
                + self.config.moment_loss_weight * moment_loss
            )

        x_train = x_scaled[prepared.training_indices]
        t_train = t_scaled[prepared.training_indices]
        y_train = y_scaled[prepared.training_indices]
        m_train = m_scaled[prepared.training_indices]
        x_val = x_scaled[prepared.validation_indices]
        t_val = t_scaled[prepared.validation_indices]
        y_val = y_scaled[prepared.validation_indices]
        m_val = m_scaled[prepared.validation_indices]

        history: dict[str, list[float]] = {"loss": [], "val_loss": []}
        best_params = self.parameters
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        training_start = perf_counter()
        step_index = 0
        rng = jax.random.PRNGKey(self.config.random_seed)
        for epoch in range(1, self.config.epochs + 1):
            indices = np.arange(x_train.shape[0])
            np.random.default_rng(self.config.random_seed + epoch).shuffle(indices)
            epoch_loss_accumulator = 0.0
            seen = 0
            for start in range(0, x_train.shape[0], self.config.batch_size):
                batch_indices = indices[start:start + self.config.batch_size]
                batch_profiles = jnp.asarray(x_train[batch_indices], dtype=jnp.float32)
                batch_times = jnp.asarray(t_train[batch_indices], dtype=jnp.float32)
                batch_coefficients = jnp.asarray(y_train[batch_indices], dtype=jnp.float32)
                batch_moments = jnp.asarray(m_train[batch_indices], dtype=jnp.float32)
                rng, batch_key = jax.random.split(rng)
                opt_state = train_step(
                    step_index,
                    opt_state,
                    batch_profiles,
                    batch_times,
                    batch_coefficients,
                    batch_moments,
                    batch_key,
                )
                params = get_params(opt_state)
                batch_loss = evaluate_loss(params, batch_profiles, batch_times, batch_coefficients, batch_moments)
                epoch_loss_accumulator += batch_loss * len(batch_indices)
                seen += len(batch_indices)
                step_index += 1

            params = get_params(opt_state)
            epoch_loss = epoch_loss_accumulator / max(seen, 1)
            epoch_val_loss = evaluate_loss(
                params,
                jnp.asarray(x_val, dtype=jnp.float32),
                jnp.asarray(t_val, dtype=jnp.float32),
                jnp.asarray(y_val, dtype=jnp.float32),
                jnp.asarray(m_val, dtype=jnp.float32),
            )
            history["loss"].append(float(epoch_loss))
            history["val_loss"].append(float(epoch_val_loss))

            if epoch_val_loss < best_loss:
                best_loss = float(epoch_val_loss)
                best_epoch = epoch
                best_params = params
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break

        training_seconds = perf_counter() - training_start
        self.parameters = best_params
        epochs_ran = len(history["loss"])

        prediction_start = perf_counter()
        predicted_coefficients = self.predict_coefficients(
            prepared.profiles[prepared.validation_indices],
            sample_times=prepared.sample_times[prepared.validation_indices].reshape(-1),
        )
        predicted_moments = self.predict_moments(
            prepared.profiles[prepared.validation_indices],
            sample_times=prepared.sample_times[prepared.validation_indices].reshape(-1),
        )
        prediction_seconds = perf_counter() - prediction_start

        validation_profile_relative_l2, validation_coefficient_mse, validation_moment_mae = _validation_metrics(
            self.basis,
            self.runtime,
            prepared.profiles[prepared.validation_indices],
            predicted_coefficients,
            predicted_moments,
            prepared.coefficients[prepared.validation_indices],
            prepared.moments[prepared.validation_indices],
            prepared.grid,
        )

        parameter_count = int(
            sum(
                int(np.prod(array.shape))
                for array in jax.tree_util.tree_leaves(self.parameters)
            )
        )
        train_size = int(prepared.training_indices.shape[0])
        validation_size = int(prepared.validation_indices.shape[0])
        return ModalRegressionResult(
            backend="jax",
            runtime=self.runtime,
            train_size=train_size,
            validation_size=validation_size,
            history=history,
            epochs_ran=epochs_ran,
            best_epoch=best_epoch,
            best_validation_loss=float(best_loss),
            parameter_count=parameter_count,
            training_seconds=float(training_seconds),
            training_profiles_per_second=float((train_size * max(epochs_ran, 1)) / max(training_seconds, 1e-9)),
            validation_prediction_seconds=float(prediction_seconds),
            validation_inference_profiles_per_second=float(validation_size / max(prediction_seconds, 1e-9)),
            validation_profile_relative_l2=validation_profile_relative_l2,
            validation_coefficient_mse=validation_coefficient_mse,
            validation_moment_mae=validation_moment_mae,
        )

    def _check_ready(self) -> None:
        if self.parameters is None:
            raise RuntimeError("fit must be called before prediction")
        if any(
            value is None
            for value in (
                self.profile_mean,
                self.profile_scale,
                self.time_mean,
                self.time_scale,
                self.coefficient_mean,
                self.coefficient_scale,
                self.moment_mean,
                self.moment_scale,
            )
        ):
            raise RuntimeError("fit must be called before prediction")

    def _scaled_inputs(self, density_profiles, sample_times=None) -> tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        profile_array = np.asarray(density_profiles, dtype=np.float32)
        if profile_array.ndim == 1:
            profile_array = profile_array[None, :]
        if sample_times is None:
            time_array = np.zeros((profile_array.shape[0], 1), dtype=np.float32)
        else:
            time_array = np.asarray(sample_times, dtype=np.float32).reshape(-1, 1)
            if time_array.shape[0] != profile_array.shape[0]:
                raise ValueError("sample_times must match the number of profiles")
        return (
            (profile_array - self.profile_mean) / self.profile_scale,
            (time_array - self.time_mean) / self.time_scale,
        )

    def predict_coefficients(self, density_profiles, sample_times=None) -> np.ndarray:
        import jax.numpy as jnp

        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        coefficients, _ = _jax_forward(
            self.parameters,
            jnp.asarray(profile_inputs, dtype=jnp.float32),
            jnp.asarray(time_inputs, dtype=jnp.float32),
            training=False,
            dropout_rate=0.0,
        )
        output = np.asarray(coefficients, dtype=np.float32) * self.coefficient_scale + self.coefficient_mean
        return np.asarray(output, dtype=np.float32)

    def predict_moments(self, density_profiles, sample_times=None) -> np.ndarray:
        import jax.numpy as jnp

        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        _, moments = _jax_forward(
            self.parameters,
            jnp.asarray(profile_inputs, dtype=jnp.float32),
            jnp.asarray(time_inputs, dtype=jnp.float32),
            training=False,
            dropout_rate=0.0,
        )
        output = np.asarray(moments, dtype=np.float32) * self.moment_scale + self.moment_mean
        return np.asarray(output, dtype=np.float32)

    def reconstruct_profiles(self, density_profiles, position_grid, sample_times=None) -> torch.Tensor:
        predicted_coefficients = self.predict_coefficients(density_profiles, sample_times=sample_times)
        return reconstruct_profiles_from_basis(
            torch.as_tensor(predicted_coefficients, dtype=self.basis.domain.real_dtype),
            torch.as_tensor(position_grid, dtype=self.basis.domain.real_dtype),
            self.basis,
        )

    def export(self, path: str | Path) -> Path:
        self._check_ready()
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("wb") as handle:
            pickle.dump(
                {
                    "backend": "jax",
                    "config": asdict(self.config),
                    "parameters": self.parameters,
                    "profile_mean": self.profile_mean,
                    "profile_scale": self.profile_scale,
                    "time_mean": self.time_mean,
                    "time_scale": self.time_scale,
                    "coefficient_mean": self.coefficient_mean,
                    "coefficient_scale": self.coefficient_scale,
                    "moment_mean": self.moment_mean,
                    "moment_scale": self.moment_scale,
                },
                handle,
            )
        return export_path


def create_modal_regressor(
    backend: ModalBackend,
    basis: InfiniteWellBasis,
    *,
    config: ModalSurrogateConfig = ModalSurrogateConfig(),
) -> ModalRegressor:
    resolved = resolve_modal_backend(backend)
    if resolved == "torch":
        return PyTorchModalRegressor(basis, config=config)
    if resolved == "jax":
        return JAXModalRegressor(basis, config=config)
    if resolved == "tensorflow":
        return TensorFlowModalRegressorAdapter(basis, config=config)
    raise ValueError(f"unsupported ML backend: {backend}")


__all__ = [
    "JAXModalRegressor",
    "MLBackendReport",
    "MLBackendRuntime",
    "ModalBackendResolution",
    "ModalRegressionResult",
    "ModalSurrogateConfig",
    "PyTorchModalRegressor",
    "TensorFlowModalRegressorAdapter",
    "create_modal_regressor",
    "inspect_jax_backend",
    "inspect_ml_backends",
    "inspect_tensorflow_backend",
    "inspect_torch_backend",
    "jax_is_available",
    "resolve_modal_backend",
]
