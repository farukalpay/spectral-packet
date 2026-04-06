from __future__ import annotations

import spectral_packet_engine as spe


def test_grouped_public_surfaces_are_exposed() -> None:
    assert spe.core.InfiniteWellBasis is spe.InfiniteWellBasis
    assert spe.data.TabularDataset is spe.TabularDataset
    assert spe.data.export_feature_table_from_profile_table is spe.export_feature_table_from_profile_table
    assert (
        spe.data.resolve_profile_table_layout_from_tabular_dataset
        is spe.resolve_profile_table_layout_from_tabular_dataset
    )
    assert spe.product.inspect_product_identity is spe.inspect_product_identity
    assert spe.product.guide_workflow is spe.guide_workflow
    assert spe.surrogates.inspect_ml_backends is spe.inspect_ml_backends
    assert spe.tree_models.inspect_tree_backends is spe.inspect_tree_backends
    assert spe.interfaces.create_mcp_server is spe.create_mcp_server
    assert spe.interfaces.inspect_service_status is spe.inspect_service_status
    assert spe.physics_contracts.HamiltonianOperator is spe.HamiltonianOperator
    assert spe.benchmark_registry.BenchmarkRegistryReport is spe.BenchmarkRegistryReport
    assert spe.open_systems.LindbladOperator is spe.LindbladOperator
    assert spe.spectral_dataset.SpectralDataset is spe.SpectralDataset
    assert spe.data.SpectralDataset is spe.SpectralDataset
    assert spe.build_profile_table_report is not None
    assert spe.execute_database_script is not None
    assert spe.execute_database_statement is not None
    assert spe.write_feature_table_artifacts is not None
    assert spe.write_benchmark_registry_artifacts is not None
    assert spe.write_spectral_dataset_artifacts is not None
    assert spe.train_tree_model is not None
    assert spe.tune_tree_model is not None
    assert spe.build_separable_2d_report is not None
    assert spe.make_infinite_well_axis_modes is not None
    assert spe.write_profile_table_report_artifacts is not None
    assert spe.run_release_gate is not None


def test_product_identity_report_exposes_one_shared_workflow_map() -> None:
    report = spe.inspect_product_identity()

    assert report.product_name == "Spectral Packet Engine"
    assert report.hero_workflow.workflow_id == "profile-table-report"
    assert report.hero_workflow.surfaces.python == "load_profile_table_report(...)"
    assert report.hero_workflow.surfaces.cli == "profile-report"
    assert len(report.killer_workflows) == 3
    assert report.killer_workflows[0].killer_workflow_id == "spectral-evidence-loop"
    assert report.killer_workflows[1].killer_workflow_id == "inverse-reconstruction-loop"
    assert report.killer_workflows[2].killer_workflow_id == "spectral-feature-model-loop"
    assert report.opinionated_defaults["profile_report"]["analyze_num_modes"] == spe.DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES
    assert report.opinionated_defaults["workflow_routing"]["default_goal"] == "report"
    assert report.replaceability_risks
    assert report.decision_burdens
    assert report.glue_burdens
    assert any(workflow.workflow_id == "profile-table-report-from-sql" for workflow in report.workflows)
    assert any(workflow.workflow_id == "fit-profile-table-from-sql" for workflow in report.workflows)
    assert any(workflow.workflow_id == "infer-potential-spectrum" for workflow in report.workflows)
    assert any(workflow.workflow_id == "analyze-separable-spectrum" for workflow in report.workflows)
    assert any(workflow.workflow_id == "design-transition" for workflow in report.workflows)
    assert any(workflow.workflow_id == "profile-inference-workflow" for workflow in report.workflows)
    assert any(workflow.workflow_id == "official-benchmark-registry" for workflow in report.workflows)


def test_workflow_guide_prefers_report_first_for_file_and_sql_inputs() -> None:
    file_guidance = spe.guide_workflow(surface="python", input_kind="profile-table-file")
    sql_guidance = spe.guide_workflow(surface="api", input_kind="profile-table-sql")

    assert file_guidance.goal == "report"
    assert file_guidance.primary_workflow.workflow_id == "profile-table-report"
    assert file_guidance.plan_steps[1].action == "report.write_artifacts(...)"
    assert file_guidance.killer_workflow.killer_workflow_id == "spectral-evidence-loop"
    assert sql_guidance.primary_workflow.workflow_id == "profile-table-report-from-sql"
    assert sql_guidance.defaults["sort_by_time"] is True


def test_workflow_guide_is_goal_aware_for_inverse_and_feature_model_loops() -> None:
    inverse_guidance = spe.guide_workflow(surface="cli", input_kind="profile-table-sql", goal="inverse-fit")
    feature_guidance = spe.guide_workflow(surface="mcp", input_kind="profile-table-file", goal="feature-model")

    assert inverse_guidance.killer_workflow.killer_workflow_id == "inverse-reconstruction-loop"
    assert any(step.workflow_id == "fit-profile-table-from-sql" for step in inverse_guidance.plan_steps)
    assert inverse_guidance.defaults["goal"] == "inverse-fit"
    assert feature_guidance.killer_workflow.killer_workflow_id == "spectral-feature-model-loop"
    assert any(step.workflow_id == "export-feature-table" for step in feature_guidance.plan_steps)
    assert any(step.workflow_id == "tree-model-train" for step in feature_guidance.plan_steps)
    assert feature_guidance.defaults["goal"] == "feature-model"


def test_top_level_all_is_deduplicated() -> None:
    assert len(spe.__all__) == len(set(spe.__all__))
    assert "tree_models" in spe.__all__
    assert "physics_contracts" in spe.__all__
    assert "benchmark_registry" in spe.__all__
    assert "open_systems" in spe.__all__
    assert "spectral_dataset" in spe.__all__
    assert "HamiltonianOperator" in spe.__all__
    assert "MeasurementModel" in spe.__all__
    assert "LindbladOperator" in spe.__all__
    assert "SpectralDataset" in spe.__all__
    assert "analyze_structured_coupling" in spe.__all__
    assert "BenchmarkRegistryReport" in spe.__all__
    assert "run_benchmark_registry" in spe.__all__
    assert "write_benchmark_registry_artifacts" in spe.__all__
    assert "write_spectral_dataset_artifacts" in spe.__all__
    assert "FeatureTableExportSummary" in spe.__all__
    assert "train_tree_model" in spe.__all__
    assert "tune_tree_model" in spe.__all__
    assert "WorkflowGoal" in spe.__all__
    assert "DatabaseExecutionSummary" in spe.__all__
    assert "PotentialFamilyInferenceSummary" in spe.__all__
    assert "ObservationPosteriorSummary" in spe.__all__
    assert "ObservationInformationSummary" in spe.__all__
    assert "Separable2DReport" in spe.__all__
    assert "TensorProductBasis2D" in spe.__all__
    assert "build_separable_2d_report" in spe.__all__
    assert "run_transport_resonance_workflow" in spe.__all__
