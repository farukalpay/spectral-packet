from __future__ import annotations

import json
import socket

import numpy as np
import torch

from spectral_packet_engine import (
    GradientOptimizationConfig,
    ProfileTable,
    TabularDataset,
    build_profile_table_report,
    build_separable_2d_report,
    calibrate_potential_from_spectrum,
    design_potential_for_target_transition,
    export_feature_table_from_profile_table,
    fit_gaussian_packet_to_profile_table,
    harmonic_potential,
    infer_potential_family_from_spectrum,
    analyze_separable_tensor_product_spectrum,
    run_profile_inference_workflow,
    save_profile_table_csv,
    simulate_gaussian_packet,
    train_tree_model,
    tune_tree_model,
)
from spectral_packet_engine.artifacts import (
    inspect_artifact_directory,
    to_serializable,
    write_differentiable_artifacts,
    write_feature_table_artifacts,
    write_inverse_artifacts,
    write_potential_inference_artifacts,
    write_profile_table_report_artifacts,
    write_reduced_model_artifacts,
    write_tabular_artifacts,
    write_tree_training_artifacts,
    write_tree_tuning_artifacts,
    write_vertical_workflow_artifacts,
)
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import solve_eigenproblem


def test_to_serializable_sanitizes_non_finite_numbers() -> None:
    payload = {
        "tensor": torch.tensor([1.0, float("nan"), float("inf"), -float("inf")], dtype=torch.float64),
        "array": np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64),
        "scalar": float("nan"),
        "complex": complex(float("inf"), -float("inf")),
    }

    serializable = to_serializable(payload)

    assert serializable["tensor"] == [1.0, "nan", "inf", "-inf"]
    assert serializable["array"] == ["nan", "inf", "-inf"]
    assert serializable["scalar"] == "nan"
    assert serializable["complex"] == {"real": "inf", "imag": "-inf"}


def test_artifact_directory_report_marks_complete_and_cleans_stale_runtime_state(tmp_path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / ".spectral-packet-engine-tmp-stale.json").write_text("stale", encoding="utf-8")
    (output_dir / ".spectral-packet-engine.lock").write_text(
        json.dumps(
            {
                "lock_id": "stale",
                "hostname": socket.gethostname(),
                "pid": 999999,
                "started_at_utc": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    dataset = TabularDataset.from_rows([{"time": 0.0, "value": 1.0}])
    write_tabular_artifacts(output_dir, dataset, metadata={"workflow": "unit-test"})
    report = inspect_artifact_directory(output_dir)

    assert report.complete is True
    assert report.lock_active is False
    assert report.stale_temporary_files == ()
    assert "artifacts.json" in report.files
    assert report.metadata["workflow"] == "unit-test"


def test_profile_table_report_artifacts_bundle_has_expected_layout(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 16)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            np.exp(-((grid - 0.25) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.40) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.55) ** 2) / (2 * 0.06**2)),
        ],
        dtype=np.float64,
    )
    table = ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="artifact-test")
    report = build_profile_table_report(table, analyze_num_modes=8, compress_num_modes=4, device="cpu")

    output_dir = tmp_path / "profile_report"
    write_profile_table_report_artifacts(output_dir, report, metadata={"workflow": "profile-report-test"})
    directory = inspect_artifact_directory(output_dir)

    assert directory.complete is True
    assert "profile_table_report.json" in directory.files
    assert "profile_table_summary.json" in directory.files
    assert "analysis/artifacts.json" in directory.files
    assert "compression/artifacts.json" in directory.files
    assert directory.metadata["workflow"] == "profile-report-test"
    assert directory.metadata["analyze_num_modes"] == 8
    assert directory.metadata["compress_num_modes"] == 4


def test_profile_table_report_can_write_its_own_artifacts(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 8)
    times = np.asarray([0.0, 0.1], dtype=np.float64)
    profiles = np.asarray(
        [
            [0.1, 0.2, 0.5, 0.9, 0.5, 0.2, 0.1, 0.05],
            [0.05, 0.1, 0.35, 0.8, 0.55, 0.25, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    report = build_profile_table_report(
        ProfileTable(position_grid=grid, sample_times=times, profiles=profiles),
        analyze_num_modes=4,
        compress_num_modes=3,
        device="cpu",
    )

    directory = report.write_artifacts(
        tmp_path / "report_bundle",
        metadata={"workflow": "report-method-test"},
    )

    assert directory.complete is True
    assert directory.metadata["workflow"] == "report-method-test"


def test_feature_table_artifacts_bundle_has_expected_layout(tmp_path) -> None:
    grid = np.linspace(0.0, 1.0, 16)
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    profiles = np.asarray(
        [
            np.exp(-((grid - 0.25) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.40) ** 2) / (2 * 0.06**2)),
            np.exp(-((grid - 0.55) ** 2) / (2 * 0.06**2)),
        ],
        dtype=np.float64,
    )
    table_path = tmp_path / "feature_profiles.csv"
    save_profile_table_csv(
        ProfileTable(position_grid=grid, sample_times=times, profiles=profiles, source="feature-artifact-test"),
        table_path,
    )
    summary = export_feature_table_from_profile_table(table_path, num_modes=4, device="cpu")

    output_dir = tmp_path / "feature_bundle"
    write_feature_table_artifacts(output_dir, summary)
    directory = inspect_artifact_directory(output_dir)
    schema_payload = json.loads((output_dir / "features_schema.json").read_text(encoding="utf-8"))

    assert directory.complete is True
    assert "features.csv" in directory.files
    assert "feature_table_export.json" in directory.files
    assert "features_schema.json" in directory.files
    assert directory.metadata["workflow"] == "export-features"
    assert directory.metadata["input_kind"] == "file"
    assert directory.metadata["num_modes"] == 4
    assert directory.metadata["normalize_each_profile"] is False
    assert "numpy" in directory.metadata["library_versions"]
    assert "torch" in directory.metadata["library_versions"]
    assert schema_payload["column_order"] == [
        "time",
        "mode_1",
        "mode_2",
        "mode_3",
        "mode_4",
        "mean_position",
        "width",
        "mass",
    ]
    assert schema_payload["columns"][0]["dtype"] == "float64"
    assert schema_payload["columns"][0]["semantic_meaning"] == "Sample time for the profile row."
    assert schema_payload["columns"][1]["semantic_meaning"] == "Spectral modal coefficient for mode 1."
    assert schema_payload["columns"][-1]["semantic_meaning"] == "Integrated profile mass over the bounded 1D domain."
    assert schema_payload["ordering"]["time"]["policy"] == "preserve-profile-table-sample-order"
    assert "numpy" in schema_payload["library_versions"]


def test_inverse_artifacts_bundle_includes_uncertainty_outputs(tmp_path) -> None:
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3],
        num_modes=48,
        quadrature_points=1024,
        grid_points=48,
        device="cpu",
    )
    table = ProfileTable(
        position_grid=forward.grid.detach().cpu().numpy(),
        sample_times=forward.times.detach().cpu().numpy(),
        profiles=forward.densities.detach().cpu().numpy(),
    )
    summary = fit_gaussian_packet_to_profile_table(
        table,
        initial_guess={
            "center": 0.35,
            "width": 0.10,
            "wavenumber": 23.0,
            "phase": 0.0,
        },
        num_modes=48,
        quadrature_points=1024,
        steps=120,
        learning_rate=0.05,
        device="cpu",
    )

    output_dir = tmp_path / "inverse_bundle"
    write_inverse_artifacts(output_dir, summary)
    report = inspect_artifact_directory(output_dir)
    uncertainty_payload = json.loads((output_dir / "uncertainty_summary.json").read_text(encoding="utf-8"))
    sensitivity_payload = json.loads((output_dir / "sensitivity_map.json").read_text(encoding="utf-8"))

    assert report.complete is True
    assert "inverse_fit.json" in report.files
    assert "predicted_density.csv" in report.files
    assert "uncertainty_summary.json" in report.files
    assert "parameter_posterior.csv" in report.files
    assert "modal_posterior.csv" in report.files
    assert "sensitivity_map.json" in report.files
    assert "observation_posterior.json" in report.files
    assert "observation_information.json" in report.files
    assert report.metadata["workflow"] == "fit-table"
    assert report.metadata["has_physical_inference"] is True
    assert uncertainty_payload["parameter_posterior"]["parameter_names"] == ["center", "width", "wavenumber"]
    assert sensitivity_payload["parameter_names"] == ["center", "width", "wavenumber"]
    assert uncertainty_payload["observation_posterior"]["observation_shape"] == [3, 48]
    assert uncertainty_payload["observation_information"]["observation_shape"] == [3, 48]


def test_potential_inference_and_reduced_model_artifacts_follow_shared_bundle_contract(tmp_path) -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=8.0, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues
    inference_summary = infer_potential_family_from_spectrum(
        target_eigenvalues=target,
        families=("harmonic", "double-well"),
        initial_guesses={
            "harmonic": {"omega": 5.0},
            "double-well": {"a_param": 1.5, "b_param": 1.0},
        },
        num_points=128,
        optimization_config=GradientOptimizationConfig(steps=120, learning_rate=0.04),
        device="cpu",
    )
    inference_dir = tmp_path / "potential_inference"
    write_potential_inference_artifacts(inference_dir, inference_summary)
    inference_report = inspect_artifact_directory(inference_dir)

    assert inference_report.complete is True
    assert "potential_family_inference.json" in inference_report.files
    assert "candidate_ranking.csv" in inference_report.files
    assert "best_family_observation_posterior.json" in inference_report.files
    assert "best_family_observation_information.json" in inference_report.files
    assert inference_report.metadata["workflow"] == "infer-potential-spectrum"
    assert inference_report.metadata["best_family"] == "harmonic"

    reduced_summary = analyze_separable_tensor_product_spectrum(
        family_x="harmonic",
        parameters_x={"omega": 8.0},
        family_y="harmonic",
        parameters_y={"omega": 6.0},
        num_points_x=64,
        num_points_y=64,
        num_states_x=4,
        num_states_y=4,
        num_combined_states=6,
        device="cpu",
    )
    reduced_dir = tmp_path / "reduced_model"
    write_reduced_model_artifacts(reduced_dir, reduced_summary)
    reduced_report = inspect_artifact_directory(reduced_dir)

    assert reduced_report.complete is True
    assert "reduced_model_summary.json" in reduced_report.files
    assert "combined_spectrum.csv" in reduced_report.files
    assert "mode_budget.json" in reduced_report.files
    assert "structured_operator.json" in reduced_report.files
    assert reduced_report.metadata["workflow"] == "reduced-model"

    separable_report = build_separable_2d_report(
        num_modes_x=4,
        num_modes_y=4,
        num_combined_states=6,
        grid_points_x=32,
        grid_points_y=32,
        device="cpu",
    )
    separable_report_dir = tmp_path / "separable_2d_report"
    write_reduced_model_artifacts(separable_report_dir, separable_report)
    separable_directory = inspect_artifact_directory(separable_report_dir)
    mode_budget_payload = json.loads((separable_report_dir / "mode_budget.json").read_text(encoding="utf-8"))

    assert separable_directory.complete is True
    assert "separable_2d_report.json" in separable_directory.files
    assert "separable_2d_summary.json" in separable_directory.files
    assert "eigenvalues.csv" in separable_directory.files
    assert "mode_budget.json" in separable_directory.files
    assert "tensor_basis.json" in separable_directory.files
    assert separable_directory.metadata["workflow"] == "separable-2d-report"
    assert separable_directory.metadata["example_name"] == "box-plus-box"
    assert mode_budget_payload["total_tensor_mode_count"] == 16


def test_differentiable_and_vertical_artifacts_follow_shared_bundle_contract(tmp_path) -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64, device="cpu")
    target_spectrum = solve_eigenproblem(
        lambda x: harmonic_potential(x, omega=7.5, domain=domain),
        domain,
        num_points=128,
        num_states=3,
    ).eigenvalues
    calibration_summary = calibrate_potential_from_spectrum(
        family="harmonic",
        target_eigenvalues=target_spectrum,
        initial_guess={"omega": 4.0},
        num_points=128,
        optimization_config=GradientOptimizationConfig(steps=120, learning_rate=0.04),
        device="cpu",
    )
    calibration_dir = tmp_path / "calibration"
    write_differentiable_artifacts(calibration_dir, calibration_summary)
    calibration_report = inspect_artifact_directory(calibration_dir)

    assert calibration_report.complete is True
    assert "differentiable_summary.json" in calibration_report.files
    assert "predicted_eigenvalues.csv" in calibration_report.files
    assert "parameter_posterior.csv" in calibration_report.files
    assert "sensitivity_map.json" in calibration_report.files
    assert "observation_posterior.json" in calibration_report.files
    assert "observation_information.json" in calibration_report.files
    assert calibration_report.metadata["workflow"] == "calibrate-potential"

    target_transition = float((target_spectrum[1] - target_spectrum[0]).item())
    design_summary = design_potential_for_target_transition(
        family="harmonic",
        target_transition=target_transition,
        initial_guess={"omega": 4.0},
        num_points=128,
        num_states=3,
        optimization_config=GradientOptimizationConfig(steps=120, learning_rate=0.04),
        device="cpu",
    )
    differentiable_dir = tmp_path / "differentiable"
    write_differentiable_artifacts(differentiable_dir, design_summary)
    differentiable_report = inspect_artifact_directory(differentiable_dir)

    assert differentiable_report.complete is True
    assert "differentiable_summary.json" in differentiable_report.files
    assert "transition_design_spectrum.csv" in differentiable_report.files
    assert differentiable_report.metadata["workflow"] == "design-transition"

    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3],
        num_modes=48,
        quadrature_points=1024,
        grid_points=48,
        device="cpu",
    )
    table_path = tmp_path / "profiles.csv"
    save_profile_table_csv(
        ProfileTable(
            position_grid=forward.grid.detach().cpu().numpy(),
            sample_times=forward.times.detach().cpu().numpy(),
            profiles=forward.densities.detach().cpu().numpy(),
        ),
        table_path,
    )
    vertical_summary = run_profile_inference_workflow(
        table_path,
        initial_guess={
            "center": 0.35,
            "width": 0.10,
            "wavenumber": 23.0,
            "phase": 0.0,
        },
        analyze_num_modes=8,
        compress_num_modes=4,
        inverse_num_modes=48,
        feature_num_modes=6,
        quadrature_points=1024,
        device="cpu",
    )
    vertical_dir = tmp_path / "vertical"
    write_vertical_workflow_artifacts(vertical_dir, vertical_summary)
    vertical_report = inspect_artifact_directory(vertical_dir)

    assert vertical_report.complete is True
    assert "vertical_workflow_summary.json" in vertical_report.files
    assert "report/artifacts.json" in vertical_report.files
    assert "inverse/artifacts.json" in vertical_report.files
    assert "features/artifacts.json" in vertical_report.files
    assert vertical_report.metadata["workflow"] == "profile-inference-workflow"


def test_tree_training_and_tuning_artifacts_follow_shared_bundle_contract(tmp_path) -> None:
    dataset = TabularDataset.from_rows(
        [
            {
                "time": float(index) / 11.0,
                "mode_1": 0.2 + 0.03 * index,
                "mode_2": 1.2 - 0.04 * index,
                "mean_position": 0.25 + 0.01 * index,
                "target": 0.8 + 0.06 * index,
            }
            for index in range(12)
        ]
    )
    training = train_tree_model(
        dataset,
        target_column="target",
        library="sklearn",
        params={"n_estimators": 16, "max_depth": 4},
        export_dir=tmp_path / "tree_model_export",
    )
    training_dir = tmp_path / "tree_train_bundle"
    write_tree_training_artifacts(training_dir, training)
    training_report = inspect_artifact_directory(training_dir)

    assert training_report.complete is True
    assert "tree_training.json" in training_report.files
    assert "predictions.csv" in training_report.files
    assert "feature_importance.csv" in training_report.files
    assert any(path.endswith(".pkl") for path in training_report.files)
    assert training_report.metadata["workflow"] == "tree-train"
    assert training_report.metadata["library"] == "sklearn"
    assert training_report.metadata["library_versions"]["sklearn"] is not None

    tuning = tune_tree_model(
        dataset,
        target_column="target",
        library="sklearn",
        search_space={"n_estimators": [8, 16], "max_depth": [2, 4]},
        search_kind="grid",
        cv=2,
        export_dir=tmp_path / "tree_best_model_export",
    )
    tuning_dir = tmp_path / "tree_tune_bundle"
    write_tree_tuning_artifacts(tuning_dir, tuning)
    tuning_report = inspect_artifact_directory(tuning_dir)

    assert tuning_report.complete is True
    assert "tree_tuning.json" in tuning_report.files
    assert "tuning_results.csv" in tuning_report.files
    assert "best_model/artifacts.json" in tuning_report.files
    assert tuning_report.metadata["workflow"] == "tree-tune"
    assert tuning_report.metadata["library"] == "sklearn"
    assert tuning_report.metadata["library_versions"]["sklearn"] is not None
