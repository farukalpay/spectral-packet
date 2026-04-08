from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from spectral_packet_engine import (
    analyze_profile_table_spectra,
    build_profile_table_report,
    build_profile_table_report_from_database_query,
    compare_profile_tables,
    export_feature_table_from_database_query,
    export_feature_table_from_profile_table,
    load_tabular_dataset,
    make_plane_wave_packet,
    parquet_support_is_available,
    ProfileTable,
    project_packet_state,
    TabularDataset,
    compress_profile_table,
    evaluate_tensorflow_surrogate_on_profile_table,
    fit_gaussian_packet_to_profile_table,
    save_tabular_dataset,
    simulate_gaussian_packet,
    simulate_packet_state,
    simulate_packet_sweep,
    summarize_profile_table,
    train_tree_model,
    sweep_profile_table_compression,
    TensorFlowRegressorConfig,
    tune_tree_model,
    validate_installation,
    write_tabular_dataset_to_database,
    InfiniteWell1D,
)


def _synthetic_profile_table() -> ProfileTable:
    grid = np.linspace(0.0, 1.0, 64)
    times = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    profiles = []
    for center, width in zip(np.linspace(0.25, 0.55, len(times)), np.linspace(0.06, 0.09, len(times))):
        profile = np.exp(-((grid - center) ** 2) / (2 * width**2))
        profile = profile / np.trapezoid(profile, grid)
        profiles.append(profile)
    return ProfileTable(position_grid=grid, sample_times=times, profiles=np.asarray(profiles, dtype=np.float64))


def _tree_feature_dataset() -> TabularDataset:
    exported = export_feature_table_from_profile_table(
        _synthetic_profile_table(),
        num_modes=6,
        device="cpu",
        normalize_each_profile=True,
    )
    rows: list[dict[str, float]] = []
    for repeat in range(4):
        for row in exported.table.to_rows():
            item = dict(row)
            item["time"] = float(item["time"]) + float(repeat)
            item["target"] = (
                0.7 * float(item["mode_1"])
                - 0.2 * float(item["mode_2"])
                + 0.1 * float(item["mean_position"])
                + 0.05 * float(item["time"])
            )
            rows.append(item)
    return TabularDataset.from_rows(rows)


def test_compress_profile_table_returns_low_error() -> None:
    summary = compress_profile_table(_synthetic_profile_table(), num_modes=24, device="cpu")

    assert summary.num_modes == 24
    assert float(summary.error_summary.mean_relative_l2_error) < 0.05
    assert float(summary.error_summary.max_relative_l2_error) < 0.08


def test_fit_gaussian_packet_to_profile_table_recovers_simulated_parameters() -> None:
    forward = simulate_gaussian_packet(
        center=0.30,
        width=0.07,
        wavenumber=25.0,
        times=[0.0, 1e-3, 3e-3],
        num_modes=96,
        quadrature_points=2048,
        grid_points=96,
        device="cpu",
    )
    table = ProfileTable(
        position_grid=forward.grid.detach().cpu().numpy(),
        sample_times=forward.times.detach().cpu().numpy(),
        profiles=forward.densities.detach().cpu().numpy(),
    )

    result = fit_gaussian_packet_to_profile_table(
        table,
        initial_guess={
            "center": 0.36,
            "width": 0.11,
            "wavenumber": 22.0,
            "phase": 0.0,
        },
        num_modes=96,
        quadrature_points=1024,
        steps=180,
        learning_rate=0.05,
        device="cpu",
    )

    assert result.final_loss < 1e-3
    assert abs(result.estimated_parameters.center[0].item() - 0.30) < 0.02
    assert abs(result.estimated_parameters.width[0].item() - 0.07) < 0.02
    assert abs(result.estimated_parameters.wavenumber[0].item() - 25.0) < 0.8
    assert result.physical_inference is not None
    assert result.physical_inference.parameter_posterior.parameter_names == ("center", "width", "wavenumber")
    assert result.physical_inference.coefficient_posterior is not None
    assert result.physical_inference.sensitivity is not None


def test_generic_packet_workflows_support_plane_wave_family() -> None:
    domain = InfiniteWell1D.from_length(1.0)
    packet = make_plane_wave_packet(domain, wavenumber=9.0)

    projection = project_packet_state(
        packet,
        num_modes=96,
        quadrature_points=2048,
        grid_points=512,
        device="cpu",
    )
    forward = simulate_packet_state(
        packet,
        times=[0.0, 1e-3, 2e-3],
        num_modes=96,
        quadrature_points=2048,
        grid_points=512,
        device="cpu",
    )

    assert projection.coefficients.shape == (96,)
    assert float(projection.reconstruction_error) < 0.15
    assert torch.isclose(projection.density_matrix.normalized_purity, torch.tensor(1.0, dtype=torch.float64), atol=1e-10).item()
    assert projection.density_matrix.normalized_is_pure.item() is True
    assert projection.initial_support.outside_probability_mass[0].item() == pytest.approx(0.0)
    assert projection.initial_support.boundary_density_mismatch[0].item() > 0.0
    assert torch.allclose(forward.total_probability, torch.ones_like(forward.total_probability), atol=5e-3, rtol=5e-3)


def test_profile_table_summary_and_compression_sweep() -> None:
    table = _synthetic_profile_table()

    summary = summarize_profile_table(table, device="cpu")
    sweep = sweep_profile_table_compression(table, mode_counts=[4, 8, 16], device="cpu")

    assert summary.num_samples == table.num_samples
    assert summary.num_positions == table.num_positions
    assert tuple(sweep.mode_counts.detach().cpu().tolist()) == (4.0, 8.0, 16.0)
    assert float(sweep.mean_relative_l2_error[-1]) <= float(sweep.mean_relative_l2_error[0])


def test_profile_table_report_exposes_hero_workflow_summary() -> None:
    table = _synthetic_profile_table()
    report = build_profile_table_report(
        table,
        analyze_num_modes=12,
        compress_num_modes=6,
        device="cpu",
    )

    assert report.overview.num_samples == table.num_samples
    assert report.overview.num_positions == table.num_positions
    assert report.overview.analyze_num_modes == 12
    assert report.overview.compress_num_modes == 6
    assert report.analysis.num_modes == 12
    assert report.compression.num_modes == 6
    assert report.overview.dominant_modes[0] >= 1
    assert report.overview.capture_mode_budgets[1].threshold == pytest.approx(0.95)
    assert report.overview.mean_relative_l2_error < 0.2


def test_forward_simulation_summary_exposes_position_uncertainty_and_support() -> None:
    forward = simulate_gaussian_packet(
        center=0.22,
        width=0.08,
        wavenumber=14.0,
        times=[0.0, 1e-3],
        num_modes=64,
        quadrature_points=1024,
        grid_points=256,
        device="cpu",
    )

    assert forward.position_variance.shape == forward.times.shape
    assert forward.position_standard_deviation.shape == forward.times.shape
    assert forward.density_matrix.trace.shape == forward.times.shape
    assert torch.allclose(
        forward.density_matrix.normalized_purity,
        torch.ones_like(forward.density_matrix.normalized_purity),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.all(forward.density_matrix.normalized_is_pure).item()
    assert torch.all(forward.position_standard_deviation > 0).item()
    assert forward.initial_support.inside_probability_mass.shape == (1,)


def test_profile_table_report_from_database_query_matches_file_workflow(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    dataset = TabularDataset.from_rows(
        [
            {"label": "late", "time": 0.2, "x=1.0": 0.15, "x=0.0": 0.2, "x=0.5": 0.85},
            {"label": "early", "time": 0.0, "x=1.0": 0.1, "x=0.0": 0.1, "x=0.5": 1.0},
        ]
    )
    write_tabular_dataset_to_database(str(database_path), "profiles", dataset, if_exists="replace")

    report = build_profile_table_report_from_database_query(
        str(database_path),
        'SELECT label, time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"',
        time_column="time",
        position_columns=["x=1.0", "x=0.0", "x=0.5"],
        sort_by_time=True,
        analyze_num_modes=4,
        compress_num_modes=3,
        device="cpu",
    )

    assert report.overview.num_samples == 2
    assert report.overview.num_positions == 3
    assert report.analysis.num_modes == 4
    assert report.compression.num_modes == 3


def test_feature_table_export_from_profile_table_exposes_traceable_schema() -> None:
    summary = export_feature_table_from_profile_table(
        _synthetic_profile_table(),
        num_modes=6,
        device="cpu",
        normalize_each_profile=True,
    )

    assert summary.source_kind == "profile-table"
    assert summary.identifier_columns == ("time",)
    assert summary.includes == ("coefficients", "moments")
    assert summary.num_rows == 4
    assert summary.num_features == 9
    assert summary.feature_names[:3] == ("mode_1", "mode_2", "mode_3")
    assert summary.feature_names[-3:] == ("mean_position", "width", "mass")
    assert summary.table.row_count == 4
    assert summary.metadata["workflow"] == "export-features"
    assert summary.metadata["input"]["kind"] == "profile-table"
    assert summary.metadata["feature_generation"]["num_modes"] == 6
    assert summary.metadata["feature_generation"]["normalize_each_profile"] is True
    assert summary.ordering["time"]["was_reordered"] is False
    assert summary.ordering["positions"]["policy"] == "preserve-profile-table-grid-order"
    assert "numpy" in summary.library_versions
    assert "torch" in summary.library_versions


def test_feature_table_export_from_database_query_preserves_sql_provenance(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite"
    dataset = TabularDataset.from_rows(
        [
            {"label": "late", "time": 0.2, "x=1.0": 0.15, "x=0.0": 0.2, "x=0.5": 0.85},
            {"label": "early", "time": 0.0, "x=1.0": 0.1, "x=0.0": 0.1, "x=0.5": 1.0},
        ]
    )
    write_tabular_dataset_to_database(str(database_path), "profiles", dataset, if_exists="replace")

    summary = export_feature_table_from_database_query(
        str(database_path),
        'SELECT label, time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"',
        time_column="time",
        position_columns=["x=1.0", "x=0.0", "x=0.5"],
        sort_by_time=True,
        num_modes=3,
        device="cpu",
    )

    assert summary.source_kind == "database-query"
    assert summary.source_location is not None
    assert summary.source_location.endswith("profiles.sqlite")
    assert summary.num_rows == 2
    assert summary.num_features == 6
    assert summary.metadata["input"]["kind"] == "database-query"
    assert summary.metadata["input"]["profile_table"]["sort_by_time"] is True
    assert summary.ordering["time"]["requested_sort"] is True
    assert summary.ordering["time"]["was_reordered"] is True
    assert summary.ordering["positions"]["source_columns"] == ["x=1.0", "x=0.0", "x=0.5"]
    assert summary.ordering["positions"]["materialized_columns"] == ["x=0.0", "x=0.5", "x=1.0"]


def test_database_query_workflow_artifact_metadata_keeps_canonical_query_provenance(tmp_path) -> None:
    import spectral_packet_engine.workflows as workflows

    database_path = tmp_path / "profiles.sqlite"
    metadata = workflows.database_query_workflow_artifact_metadata(
        "db-query",
        database_path,
        "SELECT 1 AS value",
        parameters={"label": "test"},
    )

    assert metadata["workflow"] == "db-query"
    assert metadata["input"]["kind"] == "database-query"
    assert metadata["input"]["database"].endswith("profiles.sqlite")
    assert metadata["input"]["parameters"] == {"label": "test"}


def test_database_profile_query_workflow_artifact_metadata_keeps_profile_controls(tmp_path) -> None:
    import spectral_packet_engine.workflows as workflows

    database_path = tmp_path / "profiles.sqlite"
    metadata = workflows.database_profile_query_workflow_artifact_metadata(
        "sql-analyze-table",
        database_path,
        'SELECT time, "x=1.0", "x=0.0", "x=0.5" FROM "profiles"',
        parameters={"label": "test"},
        time_column="time",
        position_columns=["x=1.0", "x=0.0", "x=0.5"],
        sort_by_time=True,
    )

    assert metadata["workflow"] == "sql-analyze-table"
    assert metadata["input"]["kind"] == "database-query"
    assert metadata["input"]["profile_table"]["time_column"] == "time"
    assert metadata["input"]["profile_table"]["position_columns"] == ("x=1.0", "x=0.0", "x=0.5")
    assert metadata["input"]["profile_table"]["sort_by_time"] is True


def test_tree_training_and_tuning_workflows_accept_feature_table_paths(tmp_path) -> None:
    dataset = _tree_feature_dataset()
    features_path = tmp_path / "features.csv"
    save_tabular_dataset(dataset, features_path)

    training = train_tree_model(
        features_path,
        target_column="target",
        library="sklearn",
        params={"n_estimators": 24, "max_depth": 5},
        test_fraction=0.25,
        random_state=4,
        export_dir=tmp_path / "trained_model",
    )
    tuning = tune_tree_model(
        features_path,
        target_column="target",
        library="sklearn",
        search_space={"n_estimators": [8, 16], "max_depth": [2, 4]},
        search_kind="grid",
        cv=2,
        test_fraction=0.25,
        random_state=4,
        export_dir=tmp_path / "tuned_model",
    )

    assert training.library == "sklearn"
    assert training.metrics["rmse"] >= 0.0
    assert training.export_path is not None
    assert Path(training.export_path).exists()
    assert tuning.library == "sklearn"
    assert tuning.candidate_count == 4
    assert tuning.training.export_path is not None
    assert Path(tuning.training.export_path).exists()


def test_feature_table_export_parquet_roundtrip_when_available(tmp_path) -> None:
    if not parquet_support_is_available():
        pytest.xfail("pyarrow is not installed in this environment")

    summary = export_feature_table_from_profile_table(
        _synthetic_profile_table(),
        num_modes=4,
        device="cpu",
        format="parquet",
    )
    output_dir = tmp_path / "feature_parquet"
    from spectral_packet_engine.artifacts import write_feature_table_artifacts

    write_feature_table_artifacts(output_dir, summary)
    loaded = load_tabular_dataset(output_dir / "features.parquet")

    assert loaded.column_names == summary.table.column_names
    assert loaded.row_count == summary.table.row_count


def test_feature_table_export_parquet_reports_missing_pyarrow_clearly(monkeypatch) -> None:
    monkeypatch.setattr("spectral_packet_engine.workflows.parquet_support_is_available", lambda: False)

    with pytest.raises(ModuleNotFoundError, match="Install the 'files' extra"):
        export_feature_table_from_profile_table(
            _synthetic_profile_table(),
            num_modes=4,
            device="cpu",
            format="parquet",
        )


def test_compression_sweep_rejects_empty_mode_counts() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sweep_profile_table_compression(_synthetic_profile_table(), mode_counts=[], device="cpu")


def test_spectral_analysis_and_table_comparison() -> None:
    table = _synthetic_profile_table()

    analysis = analyze_profile_table_spectra(table, num_modes=24, device="cpu")
    comparison = compare_profile_tables(table, table, device="cpu")

    assert analysis.coefficients.shape == (table.num_samples, 24)
    assert analysis.spectral_summary.dominant_modes.shape[0] >= 1
    assert torch.all(
        analysis.spectral_summary.max_mode_counts_for_thresholds[1:]
        >= analysis.spectral_summary.max_mode_counts_for_thresholds[:-1]
    )
    assert float(comparison.comparison.max_relative_l2_error) == 0.0
    assert torch.allclose(
        comparison.comparison.mass_error,
        torch.zeros_like(comparison.comparison.mass_error),
    )


def test_validate_installation_and_packet_sweep() -> None:
    validation = validate_installation("cpu")
    sweep = simulate_packet_sweep(
        [
            {"center": 0.25, "width": 0.07, "wavenumber": 22.0},
            {"center": 0.35, "width": 0.08, "wavenumber": 24.0},
        ],
        times=[0.0, 1e-3],
        num_modes=64,
        quadrature_points=1024,
        grid_points=128,
        device="cpu",
    )

    assert validation.core_ready is True
    assert "python" in validation.stable_surfaces
    if validation.environment.tree_backends.preferred_library is not None:
        assert "tree-model-workflows" in validation.beta_surfaces
    assert len(sweep.items) == 2
    assert sweep.items[0].final_total_probability > 0.99


def test_packet_sweep_reuses_one_engine_context(monkeypatch) -> None:
    import spectral_packet_engine.workflows as workflows

    original_build_engine = workflows.build_engine
    build_calls: list[object] = []

    def _counted_build_engine(*args, **kwargs):
        build_calls.append((args, kwargs))
        return original_build_engine(*args, **kwargs)

    monkeypatch.setattr(workflows, "build_engine", _counted_build_engine)

    sweep = workflows.simulate_packet_sweep(
        [
            {"center": 0.25, "width": 0.07, "wavenumber": 22.0},
            {"center": 0.35, "width": 0.08, "wavenumber": 24.0},
        ],
        times=[0.0, 1e-3],
        num_modes=32,
        quadrature_points=512,
        grid_points=64,
        device="cpu",
    )

    assert len(build_calls) == 1
    assert len(sweep.items) == 2


def test_tensorflow_evaluation_returns_profile_comparison_when_available() -> None:
    pytest.importorskip("tensorflow")

    table = _synthetic_profile_table()
    evaluation = evaluate_tensorflow_surrogate_on_profile_table(
        table,
        num_modes=8,
        config=TensorFlowRegressorConfig(
            profile_hidden_units=(128, 64),
            time_hidden_units=(8,),
            residual_blocks=1,
            dropout_rate=0.0,
            epochs=8,
            batch_size=4,
            validation_fraction=0.25,
        ),
    )

    assert evaluation.predicted_coefficients.shape == (table.num_samples, 8)
    assert evaluation.predicted_moments.shape == (table.num_samples, 2)
    assert evaluation.reconstructed_profiles.shape == table.profiles.shape
    assert evaluation.comparison.mean_relative_l2_error >= 0.0
