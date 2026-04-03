from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

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


@dataclass(frozen=True, slots=True)
class TensorFlowHostPlatform:
    system: str
    machine: str
    python_version: str
    recommended_accelerator: str
    recommended_runtime: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TensorFlowRuntime:
    version: str
    host: TensorFlowHostPlatform
    gpu_devices: tuple[str, ...]
    visible_device_types: tuple[str, ...]
    mixed_precision_policy: str
    xla_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "host": self.host.to_dict(),
            "gpu_devices": list(self.gpu_devices),
            "visible_device_types": list(self.visible_device_types),
            "mixed_precision_policy": self.mixed_precision_policy,
            "xla_enabled": self.xla_enabled,
        }


@dataclass(frozen=True, slots=True)
class TensorFlowRegressorConfig:
    profile_hidden_units: tuple[int, ...] = (512, 256)
    time_hidden_units: tuple[int, ...] = (32,)
    residual_blocks: int = 3
    dropout_rate: float = 0.05
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 80
    validation_fraction: float = 0.2
    random_seed: int = 42
    early_stopping_patience: int = 10
    coefficient_loss_weight: float = 1.0
    moment_loss_weight: float = 0.25
    enable_xla: bool = True
    enable_mixed_precision: bool = True
    cache_datasets: bool = True
    tensorboard_log_dir: str | None = None
    tensorboard_histogram_frequency: int = 0
    tensorboard_profile_batch_start: int | None = None
    tensorboard_profile_batch_stop: int | None = None

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
        if self.tensorboard_log_dir is not None and not self.tensorboard_log_dir.strip():
            raise ValueError("tensorboard_log_dir must not be empty")
        if self.tensorboard_histogram_frequency < 0:
            raise ValueError("tensorboard_histogram_frequency must be non-negative")
        profile_batch_start = self.tensorboard_profile_batch_start
        profile_batch_stop = self.tensorboard_profile_batch_stop
        if (profile_batch_start is None) != (profile_batch_stop is None):
            raise ValueError(
                "tensorboard_profile_batch_start and tensorboard_profile_batch_stop must be set together"
            )
        if profile_batch_start is not None and profile_batch_start <= 0:
            raise ValueError("tensorboard_profile_batch_start must be positive")
        if profile_batch_stop is not None and profile_batch_stop < profile_batch_start:
            raise ValueError("tensorboard_profile_batch_stop must be >= tensorboard_profile_batch_start")


@dataclass(frozen=True, slots=True)
class TensorFlowModalRegressionResult:
    runtime: TensorFlowRuntime
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


def tensorflow_is_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is not installed. Install the 'ml' extra in a supported Python environment."
        ) from exc
    return tf


def _require_tensorboard() -> None:
    try:
        import tensorboard  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard is required when tensorboard_log_dir is set. Install the 'ml' extra in the active environment."
        ) from exc


def inspect_tensorflow_host() -> TensorFlowHostPlatform:
    system = platform.system()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return TensorFlowHostPlatform(
            system=system,
            machine=machine,
            python_version=python_version,
            recommended_accelerator="Apple Metal PluggableDevice when tensorflow-metal is installed",
            recommended_runtime=(
                "Use Python 3.11 or 3.12, install tensorflow plus tensorflow-metal, "
                "and verify that TensorFlow exposes a GPU device."
            ),
        )
    if system == "Windows":
        return TensorFlowHostPlatform(
            system=system,
            machine=machine,
            python_version=python_version,
            recommended_accelerator="WSL2 CUDA for TensorFlow GPU workloads",
            recommended_runtime="Use native Windows for CPU-only runs. For TensorFlow GPU execution, use WSL2 because native Windows CUDA support ended after TensorFlow 2.10.",
        )
    return TensorFlowHostPlatform(
        system=system,
        machine=machine,
        python_version=python_version,
        recommended_accelerator="CUDA GPU when available",
        recommended_runtime=(
            "Use Python 3.11 or 3.12 with the official TensorFlow pip packages "
            "and enable CUDA when a GPU is present."
        ),
    )


def configure_tensorflow_runtime(
    *,
    enable_memory_growth: bool = True,
    enable_xla: bool = True,
    enable_mixed_precision: bool = True,
) -> TensorFlowRuntime:
    tf = _require_tensorflow()
    host = inspect_tensorflow_host()
    gpu_devices = tf.config.list_physical_devices("GPU")

    if enable_memory_growth:
        for device in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception:
                pass

    xla_enabled = False
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
            xla_enabled = True
        except Exception:
            xla_enabled = False

    mixed_precision_policy = "float32"
    if enable_mixed_precision and gpu_devices:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            mixed_precision_policy = "mixed_float16"
        except Exception:
            tf.keras.mixed_precision.set_global_policy("float32")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    visible_device_types = tuple(
        sorted({device.device_type for device in tf.config.list_physical_devices()})
    )
    return TensorFlowRuntime(
        version=tf.__version__,
        host=host,
        gpu_devices=tuple(device.name for device in gpu_devices),
        visible_device_types=visible_device_types,
        mixed_precision_policy=mixed_precision_policy,
        xla_enabled=xla_enabled,
    )


class TensorFlowModalRegressor:
    def __init__(
        self,
        basis: InfiniteWellBasis,
        *,
        config: TensorFlowRegressorConfig = TensorFlowRegressorConfig(),
    ) -> None:
        self.basis = basis
        self.config = config
        self.runtime: TensorFlowRuntime | None = None
        self.model: Any | None = None
        self.profile_mean: np.ndarray | None = None
        self.profile_scale: np.ndarray | None = None
        self.time_mean: np.ndarray | None = None
        self.time_scale: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None
        self.moment_mean: np.ndarray | None = None
        self.moment_scale: np.ndarray | None = None
        self.validation_indices: np.ndarray | None = None

    def _split_indices(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if num_samples < 2:
            raise ValueError("at least two samples are required for a train/validation split")
        rng = np.random.default_rng(self.config.random_seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        validation_size = max(1, int(round(num_samples * self.config.validation_fraction)))
        if validation_size >= num_samples:
            validation_size = num_samples - 1
        validation_indices = np.sort(indices[:validation_size])
        training_indices = np.sort(indices[validation_size:])
        return training_indices, validation_indices

    def _standardize(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return mean.astype(np.float32), scale.astype(np.float32)

    def _build_model(self, profile_dim: int):
        tf = _require_tensorflow()

        profile_input = tf.keras.Input(shape=(profile_dim,), name="profile")
        time_input = tf.keras.Input(shape=(1,), name="time")

        x = profile_input
        for width in self.config.profile_hidden_units:
            x = tf.keras.layers.Dense(width, activation=None)(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Activation("gelu")(x)

        time_branch = time_input
        for width in self.config.time_hidden_units:
            time_branch = tf.keras.layers.Dense(width, activation="gelu")(time_branch)

        trunk_width = self.config.profile_hidden_units[-1]
        x = tf.keras.layers.Concatenate()([x, time_branch])
        x = tf.keras.layers.Dense(trunk_width, activation="gelu")(x)

        for _ in range(self.config.residual_blocks):
            residual = x
            block = tf.keras.layers.LayerNormalization()(x)
            block = tf.keras.layers.Dense(trunk_width, activation="gelu")(block)
            if self.config.dropout_rate > 0:
                block = tf.keras.layers.Dropout(self.config.dropout_rate)(block)
            block = tf.keras.layers.Dense(trunk_width, activation=None)(block)
            x = tf.keras.layers.Add()([residual, block])
            x = tf.keras.layers.Activation("gelu")(x)

        coefficient_output = tf.keras.layers.Dense(
            self.basis.num_modes,
            dtype="float32",
            name="coefficients",
        )(x)
        moment_output = tf.keras.layers.Dense(2, dtype="float32", name="moments")(x)

        model = tf.keras.Model(
            inputs={"profile": profile_input, "time": time_input},
            outputs={"coefficients": coefficient_output, "moments": moment_output},
            name="time_conditioned_modal_regressor",
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={"coefficients": "mse", "moments": "mse"},
            loss_weights={
                "coefficients": self.config.coefficient_loss_weight,
                "moments": self.config.moment_loss_weight,
            },
            metrics={
                "coefficients": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                "moments": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            },
            jit_compile=self.runtime.xla_enabled if self.runtime is not None else False,
        )
        return model

    def _tensorboard_profile_batch(self) -> int | tuple[int, int]:
        batch_start = self.config.tensorboard_profile_batch_start
        batch_stop = self.config.tensorboard_profile_batch_stop
        if batch_start is None or batch_stop is None:
            return 0
        return (batch_start, batch_stop)

    def _build_dataset(self, profile_inputs, time_inputs, coefficient_targets, moment_targets, *, training: bool):
        tf = _require_tensorflow()
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"profile": profile_inputs, "time": time_inputs},
                {"coefficients": coefficient_targets, "moments": moment_targets},
            )
        )
        if self.config.cache_datasets:
            dataset = dataset.cache()
        if training:
            dataset = dataset.shuffle(
                buffer_size=profile_inputs.shape[0],
                seed=self.config.random_seed,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, density_profiles, position_grid, sample_times=None) -> TensorFlowModalRegressionResult:
        tf = _require_tensorflow()
        self.runtime = configure_tensorflow_runtime(
            enable_xla=self.config.enable_xla,
            enable_mixed_precision=self.config.enable_mixed_precision,
        )
        tf.keras.utils.set_random_seed(self.config.random_seed)

        grid = torch.as_tensor(position_grid, dtype=self.basis.domain.real_dtype)
        profiles = torch.as_tensor(density_profiles, dtype=self.basis.domain.real_dtype)
        if profiles.ndim != 2:
            raise ValueError("density_profiles must be two-dimensional [sample, position]")
        if sample_times is None:
            sample_times_t = torch.zeros((profiles.shape[0], 1), dtype=self.basis.domain.real_dtype)
        else:
            sample_times_t = torch.as_tensor(sample_times, dtype=self.basis.domain.real_dtype).reshape(-1, 1)
            if sample_times_t.shape[0] != profiles.shape[0]:
                raise ValueError("sample_times must match the number of profiles")

        coefficients = project_profiles_onto_basis(profiles, grid, self.basis)
        mean_position = profile_mean(profiles, grid).reshape(-1, 1)
        width = torch.sqrt(profile_variance(profiles, grid)).reshape(-1, 1)
        moments = torch.cat([mean_position, width], dim=1)

        profile_features = profiles.detach().cpu().numpy().astype(np.float32, copy=False)
        time_features = sample_times_t.detach().cpu().numpy().astype(np.float32, copy=False)
        coefficient_targets = coefficients.detach().cpu().numpy().astype(np.float32, copy=False)
        moment_targets = moments.detach().cpu().numpy().astype(np.float32, copy=False)

        training_indices, validation_indices = self._split_indices(profile_features.shape[0])
        self.validation_indices = validation_indices

        x_train = profile_features[training_indices]
        x_val = profile_features[validation_indices]
        t_train = time_features[training_indices]
        t_val = time_features[validation_indices]
        y_train = coefficient_targets[training_indices]
        y_val = coefficient_targets[validation_indices]
        m_train = moment_targets[training_indices]
        m_val = moment_targets[validation_indices]

        self.profile_mean, self.profile_scale = self._standardize(x_train)
        self.time_mean, self.time_scale = self._standardize(t_train)
        self.coefficient_mean, self.coefficient_scale = self._standardize(y_train)
        self.moment_mean, self.moment_scale = self._standardize(m_train)

        x_train_scaled = (x_train - self.profile_mean) / self.profile_scale
        x_val_scaled = (x_val - self.profile_mean) / self.profile_scale
        t_train_scaled = (t_train - self.time_mean) / self.time_scale
        t_val_scaled = (t_val - self.time_mean) / self.time_scale
        y_train_scaled = (y_train - self.coefficient_mean) / self.coefficient_scale
        y_val_scaled = (y_val - self.coefficient_mean) / self.coefficient_scale
        m_train_scaled = (m_train - self.moment_mean) / self.moment_scale
        m_val_scaled = (m_val - self.moment_mean) / self.moment_scale

        train_dataset = self._build_dataset(
            x_train_scaled,
            t_train_scaled,
            y_train_scaled,
            m_train_scaled,
            training=True,
        )
        validation_dataset = self._build_dataset(
            x_val_scaled,
            t_val_scaled,
            y_val_scaled,
            m_val_scaled,
            training=False,
        )

        self.model = self._build_model(profile_features.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            )
        ]
        if self.config.tensorboard_log_dir is not None:
            _require_tensorboard()
            tensorboard_log_dir = Path(self.config.tensorboard_log_dir)
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(tensorboard_log_dir),
                    histogram_freq=self.config.tensorboard_histogram_frequency,
                    profile_batch=self._tensorboard_profile_batch(),
                    update_freq="epoch",
                    write_graph=True,
                )
            )

        training_start = perf_counter()
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.epochs,
            verbose=0,
            callbacks=callbacks,
        )
        training_seconds = perf_counter() - training_start

        prediction_start = perf_counter()
        predictions = self.model.predict(
            {"profile": x_val_scaled, "time": t_val_scaled},
            verbose=0,
        )
        prediction_seconds = perf_counter() - prediction_start
        predicted_coefficients = predictions["coefficients"] * self.coefficient_scale + self.coefficient_mean
        predicted_moments = predictions["moments"] * self.moment_scale + self.moment_mean

        predicted_profiles = reconstruct_profiles_from_basis(
            torch.as_tensor(predicted_coefficients, dtype=self.basis.domain.real_dtype),
            grid,
            self.basis,
        )
        validation_error = relative_l2_error(
            torch.as_tensor(x_val, dtype=self.basis.domain.real_dtype),
            predicted_profiles,
            grid,
        )
        epochs_ran = len(history.history.get("loss", ()))
        validation_loss_history = history.history.get("val_loss")
        best_epoch = epochs_ran
        best_validation_loss = float("nan")
        if validation_loss_history:
            best_epoch = int(np.argmin(validation_loss_history)) + 1
            best_validation_loss = float(np.min(validation_loss_history))
        coefficient_mse = float(np.mean((predicted_coefficients - y_val) ** 2))
        moment_mae = float(np.mean(np.abs(predicted_moments - m_val)))
        train_size = int(x_train.shape[0])
        validation_size = int(x_val.shape[0])
        training_profiles_per_second = float(
            (train_size * max(epochs_ran, 1)) / max(training_seconds, 1e-9)
        )
        validation_inference_profiles_per_second = float(
            validation_size / max(prediction_seconds, 1e-9)
        )

        return TensorFlowModalRegressionResult(
            runtime=self.runtime,
            train_size=train_size,
            validation_size=validation_size,
            history={name: [float(value) for value in values] for name, values in history.history.items()},
            epochs_ran=epochs_ran,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            parameter_count=int(self.model.count_params()),
            training_seconds=float(training_seconds),
            training_profiles_per_second=training_profiles_per_second,
            validation_prediction_seconds=float(prediction_seconds),
            validation_inference_profiles_per_second=validation_inference_profiles_per_second,
            validation_profile_relative_l2=float(torch.mean(validation_error)),
            validation_coefficient_mse=coefficient_mse,
            validation_moment_mae=moment_mae,
        )

    def _scaled_inputs(self, density_profiles, sample_times=None) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit must be called before prediction")
        if self.profile_mean is None or self.profile_scale is None or self.time_mean is None or self.time_scale is None:
            raise RuntimeError("fit must be called before prediction")

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
        if self.coefficient_mean is None or self.coefficient_scale is None:
            raise RuntimeError("fit must be called before predict_coefficients")
        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        predictions = self.model.predict({"profile": profile_inputs, "time": time_inputs}, verbose=0)
        return np.asarray(
            predictions["coefficients"] * self.coefficient_scale + self.coefficient_mean,
            dtype=np.float32,
        )

    def predict_moments(self, density_profiles, sample_times=None) -> np.ndarray:
        if self.moment_mean is None or self.moment_scale is None:
            raise RuntimeError("fit must be called before predict_moments")
        profile_inputs, time_inputs = self._scaled_inputs(density_profiles, sample_times)
        predictions = self.model.predict({"profile": profile_inputs, "time": time_inputs}, verbose=0)
        return np.asarray(
            predictions["moments"] * self.moment_scale + self.moment_mean,
            dtype=np.float32,
        )

    def reconstruct_profiles(self, density_profiles, position_grid, sample_times=None) -> torch.Tensor:
        predicted_coefficients = self.predict_coefficients(density_profiles, sample_times=sample_times)
        return reconstruct_profiles_from_basis(
            torch.as_tensor(predicted_coefficients, dtype=self.basis.domain.real_dtype),
            torch.as_tensor(position_grid, dtype=self.basis.domain.real_dtype),
            self.basis,
        )

    def export(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("fit must be called before export")
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.export(str(export_path))
        return export_path


__all__ = [
    "TensorFlowHostPlatform",
    "TensorFlowModalRegressionResult",
    "TensorFlowModalRegressor",
    "TensorFlowRegressorConfig",
    "TensorFlowRuntime",
    "configure_tensorflow_runtime",
    "inspect_tensorflow_host",
    "tensorflow_is_available",
]
