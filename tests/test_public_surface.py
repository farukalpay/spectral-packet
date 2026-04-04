from __future__ import annotations

import spectral_packet_engine as spe


def test_grouped_public_surfaces_are_exposed() -> None:
    assert spe.core.InfiniteWellBasis is spe.InfiniteWellBasis
    assert spe.data.TabularDataset is spe.TabularDataset
    assert spe.product.inspect_product_identity is spe.inspect_product_identity
    assert spe.product.guide_workflow is spe.guide_workflow
    assert spe.surrogates.inspect_ml_backends is spe.inspect_ml_backends
    assert spe.interfaces.create_mcp_server is spe.create_mcp_server
    assert spe.interfaces.inspect_service_status is spe.inspect_service_status
    assert spe.build_profile_table_report is not None
    assert spe.write_profile_table_report_artifacts is not None
    assert spe.run_release_gate is not None


def test_product_identity_report_exposes_one_shared_workflow_map() -> None:
    report = spe.inspect_product_identity()

    assert report.product_name == "Spectral Packet Engine"
    assert report.hero_workflow.workflow_id == "profile-table-report"
    assert report.hero_workflow.surfaces.python == "load_profile_table_report(...)"
    assert report.hero_workflow.surfaces.cli == "profile-report"
    assert len(report.killer_workflows) == 3
    assert report.killer_workflows[0].killer_workflow_id == "file-profile-report-loop"
    assert report.opinionated_defaults["profile_report"]["analyze_num_modes"] == spe.DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES
    assert any(workflow.workflow_id == "profile-table-report-from-sql" for workflow in report.workflows)


def test_workflow_guide_prefers_report_first_for_file_and_sql_inputs() -> None:
    file_guidance = spe.guide_workflow(surface="python", input_kind="profile-table-file")
    sql_guidance = spe.guide_workflow(surface="api", input_kind="profile-table-sql")

    assert file_guidance.primary_workflow.workflow_id == "profile-table-report"
    assert file_guidance.plan_steps[1].action == "report.write_artifacts(...)"
    assert sql_guidance.primary_workflow.workflow_id == "profile-table-report-from-sql"
    assert sql_guidance.defaults["sort_by_time"] is True


def test_top_level_all_is_deduplicated() -> None:
    assert len(spe.__all__) == len(set(spe.__all__))
