from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import inspect
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from spectral_packet_engine.tabular import TabularDataset


TreeLibrary = Literal["auto", "sklearn", "xgboost", "lightgbm", "catboost"]
TreeTask = Literal["regression", "classification"]
TreeSearchKind = Literal["grid", "random"]
TreeBackendProjectStatus = Literal["stable", "beta"]


@dataclass(frozen=True, slots=True)
class TreeBackendRuntime:
    library: str
    available: bool
    version: str | None
    python_version: str
    package_extra: str | None
    project_status: TreeBackendProjectStatus
    project_supported: bool
    recommended_runtime: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TreeBackendResolution:
    requested_library: str
    resolved_library: str | None
    available_libraries: tuple[str, ...]
    fallback_order: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TreeBackendReport:
    requested_library: str
    preferred_library: str | None
    runtime_available_libraries: tuple[str, ...]
    project_supported_libraries: tuple[str, ...]
    libraries: dict[str, TreeBackendRuntime]
    resolution: TreeBackendResolution

    @property
    def available_libraries(self) -> tuple[str, ...]:
        return self.runtime_available_libraries


@dataclass(frozen=True, slots=True)
class TreeModelConfig:
    library: str
    task: TreeTask
    model: str
    params: dict[str, Any]
    random_state: int = 0


@dataclass(frozen=True, slots=True)
class TreeDatasetSplit:
    train_size: int
    test_size: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    test_fraction: float


@dataclass(frozen=True, slots=True)
class FeatureImportanceEntry:
    feature_name: str
    importance: float


@dataclass(frozen=True, slots=True)
class TreeTrainingSummary:
    source_kind: str | None
    source_location: str | None
    library: str
    model: str
    task: TreeTask
    target_column: str
    feature_columns: tuple[str, ...]
    unused_columns: tuple[str, ...]
    num_rows: int
    num_features: int
    split: TreeDatasetSplit
    random_state: int
    library_runtime: TreeBackendRuntime
    params: dict[str, Any]
    metrics: dict[str, float]
    train_metrics: dict[str, float]
    fit_seconds: float
    prediction_seconds: float
    predictions: TabularDataset
    feature_importances: tuple[FeatureImportanceEntry, ...]
    export_path: str | None


@dataclass(frozen=True, slots=True)
class TreeTuningSummary:
    source_kind: str | None
    source_location: str | None
    library: str
    model: str
    task: TreeTask
    target_column: str
    feature_columns: tuple[str, ...]
    unused_columns: tuple[str, ...]
    num_rows: int
    num_features: int
    random_state: int
    search_kind: TreeSearchKind
    scoring: str
    cv: int
    n_iter: int | None
    library_runtime: TreeBackendRuntime
    best_score: float
    best_params: dict[str, Any]
    candidate_count: int
    results: TabularDataset
    training: TreeTrainingSummary


@dataclass(frozen=True, slots=True)
class _PreparedTreeDataset:
    dataset: TabularDataset
    feature_columns: tuple[str, ...]
    unused_columns: tuple[str, ...]
    target_column: str
    task: TreeTask
    features: np.ndarray
    target: np.ndarray


def _python_version_string() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _safe_package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _inspect_tree_backend(
    library: str,
    *,
    module_name: str,
    package_name: str,
    package_extra: str,
    project_status: TreeBackendProjectStatus,
    recommended_runtime: str,
    install_note: str,
) -> TreeBackendRuntime:
    available = _module_available(module_name)
    notes = [install_note]
    if available:
        notes.append(f"{library} tree-model support is available in this environment.")
    else:
        notes.append(f"{library} is not installed.")
    return TreeBackendRuntime(
        library=library,
        available=available,
        version=_safe_package_version(package_name) if available else None,
        python_version=_python_version_string(),
        package_extra=package_extra,
        project_status=project_status,
        project_supported=True,
        recommended_runtime=recommended_runtime,
        notes=tuple(notes),
    )


def inspect_sklearn_backend() -> TreeBackendRuntime:
    return _inspect_tree_backend(
        "sklearn",
        module_name="sklearn",
        package_name="scikit-learn",
        package_extra="ml-tree-core",
        project_status="stable",
        recommended_runtime="Install the 'ml-tree-core' extra for baseline tree-model workflows.",
        install_note="Scikit-learn is the baseline tree-model backend in the project.",
    )


def inspect_xgboost_backend() -> TreeBackendRuntime:
    return _inspect_tree_backend(
        "xgboost",
        module_name="xgboost",
        package_name="xgboost",
        package_extra="ml-xgboost",
        project_status="beta",
        recommended_runtime="Install the 'ml-xgboost' extra for XGBoost-backed tree workflows.",
        install_note="XGBoost is an optional boosted-tree backend in the project.",
    )


def inspect_lightgbm_backend() -> TreeBackendRuntime:
    return _inspect_tree_backend(
        "lightgbm",
        module_name="lightgbm",
        package_name="lightgbm",
        package_extra="ml-lightgbm",
        project_status="beta",
        recommended_runtime="Install the 'ml-lightgbm' extra for LightGBM-backed tree workflows.",
        install_note="LightGBM is an optional boosted-tree backend in the project.",
    )


def inspect_catboost_backend() -> TreeBackendRuntime:
    return _inspect_tree_backend(
        "catboost",
        module_name="catboost",
        package_name="catboost",
        package_extra="ml-catboost",
        project_status="beta",
        recommended_runtime="Install the 'ml-catboost' extra for CatBoost-backed tree workflows.",
        install_note="CatBoost is an optional boosted-tree backend in the project.",
    )


def _build_tree_backend_resolution(
    backends: Mapping[str, TreeBackendRuntime],
    *,
    requested_library: TreeLibrary = "auto",
) -> TreeBackendResolution:
    fallback_order = ("sklearn", "xgboost", "lightgbm", "catboost")
    available_libraries = tuple(
        library
        for library in fallback_order
        if backends[library].available
    )
    project_supported_libraries = tuple(
        library
        for library in available_libraries
        if backends[library].project_supported
    )

    if requested_library != "auto":
        runtime = backends[str(requested_library)]
        if runtime.available:
            return TreeBackendResolution(
                requested_library=str(requested_library),
                resolved_library=str(requested_library),
                available_libraries=available_libraries,
                fallback_order=fallback_order,
                reason=f"Using explicitly requested tree-model backend '{requested_library}'.",
                warnings=(),
            )
        return TreeBackendResolution(
            requested_library=str(requested_library),
            resolved_library=None,
            available_libraries=available_libraries,
            fallback_order=fallback_order,
            reason=f"Requested tree-model backend '{requested_library}' is unavailable in this environment.",
            warnings=runtime.notes,
        )

    if project_supported_libraries:
        resolved_library = project_supported_libraries[0]
        return TreeBackendResolution(
            requested_library="auto",
            resolved_library=resolved_library,
            available_libraries=available_libraries,
            fallback_order=fallback_order,
            reason=(
                f"Auto-selected '{resolved_library}' as the first runtime-available tree backend "
                "within the project's supported surface."
            ),
            warnings=(),
        )

    return TreeBackendResolution(
        requested_library="auto",
        resolved_library=None,
        available_libraries=available_libraries,
        fallback_order=fallback_order,
        reason="No tree-model backend is available in this environment.",
        warnings=(),
    )


def inspect_tree_backends(*, requested_library: TreeLibrary = "auto") -> TreeBackendReport:
    backends = {
        "sklearn": inspect_sklearn_backend(),
        "xgboost": inspect_xgboost_backend(),
        "lightgbm": inspect_lightgbm_backend(),
        "catboost": inspect_catboost_backend(),
    }
    runtime_available_libraries = tuple(
        library_name
        for library_name, runtime in backends.items()
        if runtime.available
    )
    project_supported_libraries = tuple(
        library_name
        for library_name, runtime in backends.items()
        if runtime.available and runtime.project_supported
    )
    preferred_library = _build_tree_backend_resolution(backends, requested_library="auto").resolved_library
    resolution = _build_tree_backend_resolution(backends, requested_library=requested_library)
    return TreeBackendReport(
        requested_library=str(requested_library),
        preferred_library=preferred_library,
        runtime_available_libraries=runtime_available_libraries,
        project_supported_libraries=project_supported_libraries,
        libraries=backends,
        resolution=resolution,
    )


def resolve_tree_library(preferred: TreeLibrary = "auto") -> str:
    report = inspect_tree_backends(requested_library=preferred)
    resolution = report.resolution
    if resolution.resolved_library is None:
        if preferred != "auto":
            runtime = report.libraries[str(preferred)]
            raise ModuleNotFoundError(
                f"The requested tree-model backend '{preferred}' is unavailable. {runtime.recommended_runtime}"
            )
        raise ModuleNotFoundError("No supported tree-model backend is available in this environment.")
    return resolution.resolved_library


def _require_sklearn():
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tree-model workflows require scikit-learn. Install the 'ml-tree-core' extra."
        ) from exc
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        root_mean_squared_error = None
    return {
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "precision_score": precision_score,
        "r2_score": r2_score,
        "recall_score": recall_score,
        "root_mean_squared_error": root_mean_squared_error,
        "GridSearchCV": GridSearchCV,
        "RandomizedSearchCV": RandomizedSearchCV,
        "train_test_split": train_test_split,
    }


def _normalize_model_token(token: str) -> str:
    return "".join(character.lower() for character in str(token) if character.isalnum())


def _inject_supported_defaults(constructor, params: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(constructor)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    merged = dict(params)
    for name, value in defaults.items():
        if name in accepted and name not in merged:
            merged[name] = value
    return merged


def _default_model_name(library: str, task: TreeTask) -> str:
    defaults = {
        ("sklearn", "regression"): "RandomForestRegressor",
        ("sklearn", "classification"): "RandomForestClassifier",
        ("xgboost", "regression"): "XGBRegressor",
        ("xgboost", "classification"): "XGBClassifier",
        ("lightgbm", "regression"): "LGBMRegressor",
        ("lightgbm", "classification"): "LGBMClassifier",
        ("catboost", "regression"): "CatBoostRegressor",
        ("catboost", "classification"): "CatBoostClassifier",
    }
    return defaults[(library, task)]


def create_tree_estimator(
    library: TreeLibrary,
    *,
    task: TreeTask,
    model: str | None = None,
    params: Mapping[str, Any] | None = None,
    random_state: int = 0,
) -> tuple[Any, TreeBackendRuntime, TreeModelConfig]:
    resolved_library = resolve_tree_library(library)
    backend_report = inspect_tree_backends(requested_library=resolved_library).libraries[resolved_library]
    requested_params = {} if params is None else dict(params)
    resolved_model = _default_model_name(resolved_library, task) if model is None else str(model)
    normalized_model = _normalize_model_token(resolved_model)

    if resolved_library == "sklearn":
        sklearn_modules = _require_sklearn()
        del sklearn_modules
        if task == "regression" and normalized_model in {"randomforestregressor", "randomforest", "rfregressor"}:
            from sklearn.ensemble import RandomForestRegressor

            effective_params = _inject_supported_defaults(
                RandomForestRegressor,
                requested_params,
                {"random_state": random_state},
            )
            estimator = RandomForestRegressor(**effective_params)
            resolved_model = "RandomForestRegressor"
        elif task == "classification" and normalized_model in {"randomforestclassifier", "randomforest", "rfclassifier"}:
            from sklearn.ensemble import RandomForestClassifier

            effective_params = _inject_supported_defaults(
                RandomForestClassifier,
                requested_params,
                {"random_state": random_state},
            )
            estimator = RandomForestClassifier(**effective_params)
            resolved_model = "RandomForestClassifier"
        elif task == "regression" and normalized_model in {"extratreesregressor", "extratrees", "etrregressor"}:
            from sklearn.ensemble import ExtraTreesRegressor

            effective_params = _inject_supported_defaults(
                ExtraTreesRegressor,
                requested_params,
                {"random_state": random_state},
            )
            estimator = ExtraTreesRegressor(**effective_params)
            resolved_model = "ExtraTreesRegressor"
        elif task == "classification" and normalized_model in {"extratreesclassifier", "extratrees", "etrclassifier"}:
            from sklearn.ensemble import ExtraTreesClassifier

            effective_params = _inject_supported_defaults(
                ExtraTreesClassifier,
                requested_params,
                {"random_state": random_state},
            )
            estimator = ExtraTreesClassifier(**effective_params)
            resolved_model = "ExtraTreesClassifier"
        else:
            raise ValueError(f"Unsupported sklearn model '{resolved_model}' for task '{task}'.")
    elif resolved_library == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("XGBoost support requires the 'ml-xgboost' extra.") from exc
        if task == "regression" and normalized_model in {"xgbregressor", "xgboostregressor", "xgb"}:
            effective_params = _inject_supported_defaults(
                XGBRegressor,
                requested_params,
                {"random_state": random_state, "verbosity": 0},
            )
            estimator = XGBRegressor(**effective_params)
            resolved_model = "XGBRegressor"
        elif task == "classification" and normalized_model in {"xgbclassifier", "xgboostclassifier", "xgb"}:
            effective_params = _inject_supported_defaults(
                XGBClassifier,
                requested_params,
                {"random_state": random_state, "verbosity": 0},
            )
            estimator = XGBClassifier(**effective_params)
            resolved_model = "XGBClassifier"
        else:
            raise ValueError(f"Unsupported xgboost model '{resolved_model}' for task '{task}'.")
    elif resolved_library == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("LightGBM support requires the 'ml-lightgbm' extra.") from exc
        if task == "regression" and normalized_model in {"lgbmregressor", "lightgbmregressor", "lgbm"}:
            effective_params = _inject_supported_defaults(
                LGBMRegressor,
                requested_params,
                {"random_state": random_state, "verbosity": -1},
            )
            estimator = LGBMRegressor(**effective_params)
            resolved_model = "LGBMRegressor"
        elif task == "classification" and normalized_model in {"lgbmclassifier", "lightgbmclassifier", "lgbm"}:
            effective_params = _inject_supported_defaults(
                LGBMClassifier,
                requested_params,
                {"random_state": random_state, "verbosity": -1},
            )
            estimator = LGBMClassifier(**effective_params)
            resolved_model = "LGBMClassifier"
        else:
            raise ValueError(f"Unsupported lightgbm model '{resolved_model}' for task '{task}'.")
    elif resolved_library == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("CatBoost support requires the 'ml-catboost' extra.") from exc
        if task == "regression" and normalized_model in {"catboostregressor", "catregressor", "catboost"}:
            effective_params = _inject_supported_defaults(
                CatBoostRegressor,
                requested_params,
                {"random_state": random_state, "verbose": False},
            )
            estimator = CatBoostRegressor(**effective_params)
            resolved_model = "CatBoostRegressor"
        elif task == "classification" and normalized_model in {"catboostclassifier", "catclassifier", "catboost"}:
            effective_params = _inject_supported_defaults(
                CatBoostClassifier,
                requested_params,
                {"random_state": random_state, "verbose": False},
            )
            estimator = CatBoostClassifier(**effective_params)
            resolved_model = "CatBoostClassifier"
        else:
            raise ValueError(f"Unsupported catboost model '{resolved_model}' for task '{task}'.")
    else:
        raise ValueError(f"Unsupported tree-model backend '{resolved_library}'.")

    return (
        estimator,
        backend_report,
        TreeModelConfig(
            library=resolved_library,
            task=task,
            model=resolved_model,
            params=effective_params,
            random_state=random_state,
        ),
    )


def _ensure_no_missing_or_non_finite(dataset: TabularDataset, columns: Sequence[str]) -> None:
    invalid_columns: list[str] = []
    for column in columns:
        values = dataset.columns[column]
        if values.dtype.kind in {"i", "u"}:
            continue
        if values.dtype.kind == "f":
            if not np.isfinite(values).all():
                invalid_columns.append(column)
            continue
        missing = [value for value in values.tolist() if value is None or value == ""]
        if missing:
            invalid_columns.append(column)
    if invalid_columns:
        raise ValueError(
            "Selected columns contain missing or non-finite values; explicit cleanup is required before tree-model training: "
            + ", ".join(invalid_columns)
        )


def prepare_tree_dataset(
    dataset: TabularDataset,
    *,
    target_column: str,
    task: TreeTask = "regression",
    feature_columns: Sequence[str] | None = None,
) -> _PreparedTreeDataset:
    if target_column not in dataset.columns:
        raise ValueError(f"unknown target column: {target_column}")

    if feature_columns is None:
        resolved_feature_columns = tuple(name for name in dataset.column_names if name != target_column)
    else:
        resolved_feature_columns = tuple(str(name) for name in feature_columns)

    if not resolved_feature_columns:
        raise ValueError("feature_columns must not be empty")
    if len(set(resolved_feature_columns)) != len(resolved_feature_columns):
        raise ValueError("feature_columns must not contain duplicates")
    if target_column in resolved_feature_columns:
        raise ValueError("target_column must not also be listed as a feature column")

    missing = [name for name in resolved_feature_columns if name not in dataset.columns]
    if missing:
        raise ValueError("unknown feature columns: " + ", ".join(missing))

    try:
        feature_matrix = dataset.numeric_matrix(resolved_feature_columns)
    except ValueError as exc:
        raise ValueError(
            "Tree-model feature columns must be numeric and explicitly inspectable. "
            + str(exc)
        ) from exc
    _ensure_no_missing_or_non_finite(dataset, resolved_feature_columns)

    target_values = dataset.columns[target_column]
    if task == "regression":
        if target_values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("Regression targets must be numeric.")
        numeric_target = target_values.astype(np.float64, copy=False)
        if not np.isfinite(numeric_target).all():
            raise ValueError("Regression targets must not contain missing or non-finite values.")
        target = numeric_target
    else:
        if target_values.dtype.kind == "f" and not np.isfinite(target_values).all():
            raise ValueError("Classification targets must not contain missing or non-finite values.")
        if target_values.dtype.kind in {"U", "S", "O", "b", "i", "u", "f"}:
            target = np.asarray(target_values)
        else:
            raise ValueError("Unsupported classification target dtype.")
        unique_labels = np.unique(target)
        if unique_labels.shape[0] < 2:
            raise ValueError("Classification targets must contain at least two classes.")

    unused_columns = tuple(
        name
        for name in dataset.column_names
        if name not in resolved_feature_columns and name != target_column
    )

    return _PreparedTreeDataset(
        dataset=dataset,
        feature_columns=resolved_feature_columns,
        unused_columns=unused_columns,
        target_column=target_column,
        task=task,
        features=feature_matrix,
        target=np.asarray(target),
    )


def _stratify_target_if_supported(target: np.ndarray, task: TreeTask) -> np.ndarray | None:
    if task != "classification":
        return None
    unique_labels, counts = np.unique(target, return_counts=True)
    del unique_labels
    if counts.min(initial=0) < 2:
        return None
    return target


def _metric_dict(task: TreeTask, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    sklearn_modules = _require_sklearn()
    if task == "regression":
        root_mean_squared_error = sklearn_modules["root_mean_squared_error"]
        if root_mean_squared_error is not None:
            rmse = root_mean_squared_error(y_true, y_pred)
        else:
            mean_squared_error = sklearn_modules["mean_squared_error"]
            if "squared" in inspect.signature(mean_squared_error).parameters:
                rmse = mean_squared_error(y_true, y_pred, squared=False)
            else:
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "rmse": float(rmse),
            "mae": float(sklearn_modules["mean_absolute_error"](y_true, y_pred)),
            "r2": float(sklearn_modules["r2_score"](y_true, y_pred)),
        }
    return {
        "accuracy": float(sklearn_modules["accuracy_score"](y_true, y_pred)),
        "precision_macro": float(
            sklearn_modules["precision_score"](y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            sklearn_modules["recall_score"](y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(sklearn_modules["f1_score"](y_true, y_pred, average="macro", zero_division=0)),
    }


def _feature_importances(
    estimator: Any,
    feature_columns: Sequence[str],
) -> tuple[FeatureImportanceEntry, ...]:
    if not hasattr(estimator, "feature_importances_"):
        return ()
    values = np.asarray(getattr(estimator, "feature_importances_"), dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(feature_columns):
        return ()
    ranked = sorted(
        zip(feature_columns, values.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(
        FeatureImportanceEntry(feature_name=str(name), importance=float(importance))
        for name, importance in ranked
    )


def _prediction_rows(
    test_indices: Sequence[int],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, actual, predicted in zip(test_indices, y_true, y_pred, strict=True):
        actual_value = actual.item() if isinstance(actual, np.generic) else actual
        predicted_value = predicted.item() if isinstance(predicted, np.generic) else predicted
        rows.append(
            {
                "row_index": int(row_index),
                "actual": actual_value,
                "predicted": predicted_value,
            }
        )
    return rows


def _resolved_export_path(export_dir: str | Path, *, library: str, model: str) -> Path:
    export_path = Path(export_dir)
    if library == "sklearn":
        suffix = ".pkl"
    elif library == "xgboost":
        suffix = ".json"
    elif library == "lightgbm":
        suffix = ".txt"
    elif library == "catboost":
        suffix = ".cbm"
    else:
        raise ValueError(f"unsupported tree-model backend for export: {library}")
    if export_path.suffix:
        return export_path
    return export_path / f"{library}_{model.lower()}{suffix}"


def export_tree_model(estimator: Any, *, library: str, model: str, export_dir: str | Path) -> Path:
    output_path = _resolved_export_path(export_dir, library=library, model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if library == "sklearn":
        with output_path.open("wb") as handle:
            pickle.dump(estimator, handle)
        return output_path
    if library == "xgboost":
        estimator.save_model(str(output_path))
        return output_path
    if library == "lightgbm":
        booster = getattr(estimator, "booster_", None)
        if booster is None:
            raise RuntimeError("LightGBM estimator does not expose a fitted booster_.")
        booster.save_model(str(output_path))
        return output_path
    if library == "catboost":
        estimator.save_model(str(output_path))
        return output_path
    raise ValueError(f"unsupported tree-model backend for export: {library}")


def _train_summary_from_fitted_estimator(
    estimator: Any,
    *,
    prepared: _PreparedTreeDataset,
    split: TreeDatasetSplit,
    library_runtime: TreeBackendRuntime,
    model_config: TreeModelConfig,
    fit_seconds: float,
    prediction_seconds: float,
    train_metrics: Mapping[str, float],
    metrics: Mapping[str, float],
    predictions: TabularDataset,
    export_path: str | None,
) -> TreeTrainingSummary:
    source = prepared.dataset.source
    return TreeTrainingSummary(
        source_kind=None if source is None else source.kind,
        source_location=None if source is None else source.location,
        library=model_config.library,
        model=model_config.model,
        task=model_config.task,
        target_column=prepared.target_column,
        feature_columns=prepared.feature_columns,
        unused_columns=prepared.unused_columns,
        num_rows=prepared.dataset.row_count,
        num_features=len(prepared.feature_columns),
        split=split,
        random_state=model_config.random_state,
        library_runtime=library_runtime,
        params=dict(model_config.params),
        metrics={str(key): float(value) for key, value in metrics.items()},
        train_metrics={str(key): float(value) for key, value in train_metrics.items()},
        fit_seconds=float(fit_seconds),
        prediction_seconds=float(prediction_seconds),
        predictions=predictions,
        feature_importances=_feature_importances(estimator, prepared.feature_columns),
        export_path=export_path,
    )


def train_tree_model_on_dataset(
    dataset: TabularDataset,
    *,
    target_column: str,
    task: TreeTask = "regression",
    library: TreeLibrary = "auto",
    model: str | None = None,
    params: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] | None = None,
    test_fraction: float = 0.2,
    random_state: int = 0,
    export_dir: str | Path | None = None,
) -> TreeTrainingSummary:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must lie in (0, 1)")

    sklearn_modules = _require_sklearn()
    prepared = prepare_tree_dataset(
        dataset,
        target_column=target_column,
        task=task,
        feature_columns=feature_columns,
    )
    estimator, library_runtime, model_config = create_tree_estimator(
        library,
        task=task,
        model=model,
        params=params,
        random_state=random_state,
    )

    indices = np.arange(prepared.dataset.row_count)
    x_train, x_test, y_train, y_test, train_indices, test_indices = sklearn_modules["train_test_split"](
        prepared.features,
        prepared.target,
        indices,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=_stratify_target_if_supported(prepared.target, task),
    )

    fit_started = perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    train_predictions = estimator.predict(x_train)
    test_predictions = estimator.predict(x_test)
    prediction_seconds = perf_counter() - predict_started

    predictions = TabularDataset.from_rows(_prediction_rows(test_indices, y_test, test_predictions))
    export_path = None if export_dir is None else str(
        export_tree_model(estimator, library=model_config.library, model=model_config.model, export_dir=export_dir)
    )
    split = TreeDatasetSplit(
        train_size=int(x_train.shape[0]),
        test_size=int(x_test.shape[0]),
        train_indices=tuple(int(index) for index in np.asarray(train_indices).tolist()),
        test_indices=tuple(int(index) for index in np.asarray(test_indices).tolist()),
        test_fraction=float(test_fraction),
    )
    return _train_summary_from_fitted_estimator(
        estimator,
        prepared=prepared,
        split=split,
        library_runtime=library_runtime,
        model_config=model_config,
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
        train_metrics=_metric_dict(task, y_train, train_predictions),
        metrics=_metric_dict(task, y_test, test_predictions),
        predictions=predictions,
        export_path=export_path,
    )


def _default_scoring(task: TreeTask) -> str:
    return "neg_root_mean_squared_error" if task == "regression" else "accuracy"


def _search_results_dataset(search) -> TabularDataset:
    parameter_names = sorted(
        {
            key[6:]
            for key in search.cv_results_.keys()
            if key.startswith("param_")
        }
    )
    rows: list[dict[str, Any]] = []
    params_list = search.cv_results_["params"]
    for index, params in enumerate(params_list):
        row = {
            "rank_test_score": int(search.cv_results_["rank_test_score"][index]),
            "mean_test_score": float(search.cv_results_["mean_test_score"][index]),
            "std_test_score": float(search.cv_results_["std_test_score"][index]),
            "mean_fit_time": float(search.cv_results_["mean_fit_time"][index]),
            "mean_score_time": float(search.cv_results_["mean_score_time"][index]),
        }
        for parameter_name in parameter_names:
            row[parameter_name] = params.get(parameter_name)
        rows.append(row)
    return TabularDataset.from_rows(rows)


def tune_tree_model_on_dataset(
    dataset: TabularDataset,
    *,
    target_column: str,
    task: TreeTask = "regression",
    library: TreeLibrary = "auto",
    model: str | None = None,
    feature_columns: Sequence[str] | None = None,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    search_kind: TreeSearchKind = "random",
    n_iter: int = 30,
    cv: int = 5,
    scoring: str | None = None,
    test_fraction: float = 0.2,
    random_state: int = 0,
    export_dir: str | Path | None = None,
) -> TreeTuningSummary:
    if search_space is None or not search_space:
        raise ValueError("search_space must not be empty")
    normalized_search_space = {str(key): list(values) for key, values in search_space.items()}
    for key, values in normalized_search_space.items():
        if not values:
            raise ValueError(f"search_space parameter '{key}' must contain at least one value")
    if search_kind not in {"grid", "random"}:
        raise ValueError("search_kind must be 'grid' or 'random'")
    if cv < 2:
        raise ValueError("cv must be at least 2")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must lie in (0, 1)")

    sklearn_modules = _require_sklearn()
    prepared = prepare_tree_dataset(
        dataset,
        target_column=target_column,
        task=task,
        feature_columns=feature_columns,
    )
    estimator, library_runtime, model_config = create_tree_estimator(
        library,
        task=task,
        model=model,
        random_state=random_state,
    )
    scoring_name = _default_scoring(task) if scoring is None else str(scoring)

    indices = np.arange(prepared.dataset.row_count)
    x_train, x_test, y_train, y_test, train_indices, test_indices = sklearn_modules["train_test_split"](
        prepared.features,
        prepared.target,
        indices,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=_stratify_target_if_supported(prepared.target, task),
    )

    if search_kind == "grid":
        search = sklearn_modules["GridSearchCV"](
            estimator,
            param_grid=normalized_search_space,
            scoring=scoring_name,
            cv=cv,
            refit=True,
        )
        resolved_n_iter: int | None = None
    else:
        search = sklearn_modules["RandomizedSearchCV"](
            estimator,
            param_distributions=normalized_search_space,
            n_iter=n_iter,
            scoring=scoring_name,
            cv=cv,
            random_state=random_state,
            refit=True,
        )
        resolved_n_iter = int(n_iter)

    fit_started = perf_counter()
    search.fit(x_train, y_train)
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    train_predictions = search.best_estimator_.predict(x_train)
    test_predictions = search.best_estimator_.predict(x_test)
    prediction_seconds = perf_counter() - predict_started

    best_model_config = TreeModelConfig(
        library=model_config.library,
        task=model_config.task,
        model=model_config.model,
        params={str(key): value for key, value in search.best_params_.items()},
        random_state=random_state,
    )
    predictions = TabularDataset.from_rows(_prediction_rows(test_indices, y_test, test_predictions))
    export_path = None if export_dir is None else str(
        export_tree_model(
            search.best_estimator_,
            library=best_model_config.library,
            model=best_model_config.model,
            export_dir=export_dir,
        )
    )
    split = TreeDatasetSplit(
        train_size=int(x_train.shape[0]),
        test_size=int(x_test.shape[0]),
        train_indices=tuple(int(index) for index in np.asarray(train_indices).tolist()),
        test_indices=tuple(int(index) for index in np.asarray(test_indices).tolist()),
        test_fraction=float(test_fraction),
    )
    training = _train_summary_from_fitted_estimator(
        search.best_estimator_,
        prepared=prepared,
        split=split,
        library_runtime=library_runtime,
        model_config=best_model_config,
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
        train_metrics=_metric_dict(task, y_train, train_predictions),
        metrics=_metric_dict(task, y_test, test_predictions),
        predictions=predictions,
        export_path=export_path,
    )

    source = prepared.dataset.source
    return TreeTuningSummary(
        source_kind=None if source is None else source.kind,
        source_location=None if source is None else source.location,
        library=best_model_config.library,
        model=best_model_config.model,
        task=best_model_config.task,
        target_column=prepared.target_column,
        feature_columns=prepared.feature_columns,
        unused_columns=prepared.unused_columns,
        num_rows=prepared.dataset.row_count,
        num_features=len(prepared.feature_columns),
        random_state=random_state,
        search_kind=search_kind,
        scoring=scoring_name,
        cv=cv,
        n_iter=resolved_n_iter,
        library_runtime=library_runtime,
        best_score=float(search.best_score_),
        best_params={str(key): value for key, value in search.best_params_.items()},
        candidate_count=int(len(search.cv_results_["params"])),
        results=_search_results_dataset(search),
        training=training,
    )


__all__ = [
    "FeatureImportanceEntry",
    "TreeBackendReport",
    "TreeBackendResolution",
    "TreeBackendRuntime",
    "TreeDatasetSplit",
    "TreeLibrary",
    "TreeModelConfig",
    "TreeSearchKind",
    "TreeTask",
    "TreeTrainingSummary",
    "TreeTuningSummary",
    "create_tree_estimator",
    "export_tree_model",
    "inspect_catboost_backend",
    "inspect_lightgbm_backend",
    "inspect_sklearn_backend",
    "inspect_tree_backends",
    "inspect_xgboost_backend",
    "prepare_tree_dataset",
    "resolve_tree_library",
    "train_tree_model_on_dataset",
    "tune_tree_model_on_dataset",
]
