from __future__ import annotations

"""Shared product identity, opinionated defaults, and workflow guidance."""

from dataclasses import dataclass
from typing import Any, Literal


PRODUCT_NAME = "Spectral Packet Engine"
PRODUCT_SPINE_STATEMENT = (
    "A Python-first spectral inverse physics engine for bounded-domain packet simulation, "
    "uncertainty-aware inverse reconstruction, controlled reduced models, differentiable design, "
    "and vertical scientific workflows, with file, SQL, CLI, MCP, and API surfaces over the same core."
)
RUNTIME_SPINE_STATEMENT = (
    "One shared Python engine validates environment assumptions explicitly, runs bounded "
    "in-process workflows, records task and artifact state, and leaves restart supervision "
    "to the host process manager."
)
DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES = 16
DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES = 8
DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS = (0.9, 0.95, 0.99)
DEFAULT_PROFILE_REPORT_NORMALIZE_EACH_PROFILE = False
DEFAULT_PROFILE_REPORT_DEVICE = "auto"
DEFAULT_PROFILE_REPORT_OUTPUT_DIR = "artifacts/profile_report"
DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR = "artifacts/profile_report_python"
DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR = "artifacts/sql_profile_report"
DEFAULT_SQL_PROFILE_TIME_COLUMN = "time"
DEFAULT_SQL_PROFILE_SORT_BY_TIME = True
DEFAULT_INVERSE_FIT_OUTPUT_DIR = "artifacts/inverse_fit"
DEFAULT_SQL_INVERSE_FIT_OUTPUT_DIR = "artifacts/sql_fit"
DEFAULT_FEATURE_EXPORT_NUM_MODES = 32
DEFAULT_FEATURE_EXPORT_FORMAT = "csv"
DEFAULT_FEATURE_EXPORT_OUTPUT_DIR = "artifacts/features"
DEFAULT_SQL_FEATURE_EXPORT_OUTPUT_DIR = "artifacts/sql_features"
DEFAULT_TREE_MODEL_LIBRARY = "auto"
DEFAULT_TREE_MODEL_TASK = "regression"
DEFAULT_TREE_MODEL_TEST_FRACTION = 0.2
DEFAULT_TREE_MODEL_RANDOM_STATE = 0
DEFAULT_TREE_MODEL_OUTPUT_DIR = "artifacts/tree_train"
DEFAULT_TREE_TUNE_OUTPUT_DIR = "artifacts/tree_tune"
DEFAULT_MCP_MAX_CONCURRENT_TASKS = 1
DEFAULT_MCP_LOG_LEVEL = "warning"
WorkflowGoal = Literal["report", "inverse-fit", "feature-model"]


@dataclass(frozen=True, slots=True)
class WorkflowSurfaceBindings:
    python: str | None = None
    cli: str | None = None
    mcp: str | None = None
    api: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.python is not None:
            payload["python"] = self.python
        if self.cli is not None:
            payload["cli"] = self.cli
        if self.mcp is not None:
            payload["mcp"] = self.mcp
        if self.api is not None:
            payload["api"] = self.api
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowIdentity:
    workflow_id: str
    label: str
    summary: str
    surfaces: WorkflowSurfaceBindings
    artifact_story: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "workflow_id": self.workflow_id,
            "label": self.label,
            "summary": self.summary,
            "surfaces": self.surfaces.to_dict(),
        }
        if self.artifact_story is not None:
            payload["artifact_story"] = self.artifact_story
        return payload


@dataclass(frozen=True, slots=True)
class KillerWorkflowDefinition:
    killer_workflow_id: str
    goal: WorkflowGoal
    label: str
    target_user: str
    input_types: tuple[str, ...]
    steps: tuple[str, ...]
    outputs: tuple[str, ...]
    entry_workflow_ids: tuple[str, ...]
    why_better_here: str

    def to_dict(self) -> dict[str, object]:
        return {
            "killer_workflow_id": self.killer_workflow_id,
            "goal": self.goal,
            "label": self.label,
            "target_user": self.target_user,
            "input_types": list(self.input_types),
            "steps": list(self.steps),
            "outputs": list(self.outputs),
            "entry_workflow_ids": list(self.entry_workflow_ids),
            "why_better_here": self.why_better_here,
        }


@dataclass(frozen=True, slots=True)
class WorkflowGuidanceStep:
    stage: str
    action: str
    detail: str
    workflow_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage": self.stage,
            "action": self.action,
            "detail": self.detail,
        }
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowGuidance:
    surface: str
    input_kind: str
    goal: WorkflowGoal
    killer_workflow: KillerWorkflowDefinition
    primary_workflow: WorkflowIdentity
    why_this_path: str
    defaults: dict[str, Any]
    plan_steps: tuple[WorkflowGuidanceStep, ...]
    follow_up_workflows: tuple[WorkflowIdentity, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "input_kind": self.input_kind,
            "goal": self.goal,
            "killer_workflow": self.killer_workflow.to_dict(),
            "primary_workflow": self.primary_workflow.to_dict(),
            "why_this_path": self.why_this_path,
            "defaults": dict(self.defaults),
            "plan_steps": [step.to_dict() for step in self.plan_steps],
            "follow_up_workflows": [workflow.to_dict() for workflow in self.follow_up_workflows],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ProductIdentityReport:
    product_name: str
    product_spine: str
    runtime_spine: str
    hero_workflow: WorkflowIdentity
    workflows: tuple[WorkflowIdentity, ...]
    killer_workflows: tuple[KillerWorkflowDefinition, ...]
    opinionated_defaults: dict[str, Any]
    replaceability_risks: tuple[str, ...]
    decision_burdens: tuple[str, ...]
    glue_burdens: tuple[str, ...]
    adoption_priorities: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "product_name": self.product_name,
            "product_spine": self.product_spine,
            "runtime_spine": self.runtime_spine,
            "hero_workflow": self.hero_workflow.to_dict(),
            "workflows": [workflow.to_dict() for workflow in self.workflows],
            "killer_workflows": [workflow.to_dict() for workflow in self.killer_workflows],
            "opinionated_defaults": dict(self.opinionated_defaults),
            "replaceability_risks": list(self.replaceability_risks),
            "decision_burdens": list(self.decision_burdens),
            "glue_burdens": list(self.glue_burdens),
            "adoption_priorities": list(self.adoption_priorities),
            "notes": list(self.notes),
        }


_WORKFLOW_CATALOG: tuple[WorkflowIdentity, ...] = (
    WorkflowIdentity(
        workflow_id="inspect-product",
        label="Inspect Product",
        summary="Inspect the shared product spine, runtime model, and canonical workflow map.",
        surfaces=WorkflowSurfaceBindings(
            python="inspect_product_identity()",
            cli="inspect-product",
            mcp="inspect_product",
            api="GET /product",
        ),
    ),
    WorkflowIdentity(
        workflow_id="inspect-environment",
        label="Inspect Environment",
        summary="Inspect machine capabilities, optional surfaces, and runtime backends before running the engine.",
        surfaces=WorkflowSurfaceBindings(
            python="inspect_environment(...)",
            cli="inspect-environment",
            mcp="inspect_environment",
            api="GET /capabilities",
        ),
    ),
    WorkflowIdentity(
        workflow_id="inspect-tree-backends",
        label="Inspect Tree Backends",
        summary="Inspect tree-model backend availability across the scikit-learn baseline and optional gradient-boosting integrations.",
        surfaces=WorkflowSurfaceBindings(
            python="inspect_tree_backend_support(...)",
            cli="tree-backends",
            mcp="inspect_tree_backends",
            api="GET /tree/backends",
        ),
    ),
    WorkflowIdentity(
        workflow_id="validate-installation",
        label="Validate Installation",
        summary="Validate the installed product surfaces, optional dependencies, and backend/runtime readiness.",
        surfaces=WorkflowSurfaceBindings(
            python="validate_installation(...)",
            cli="validate-install",
            mcp="validate_installation",
            api="GET /validate-install",
        ),
    ),
    WorkflowIdentity(
        workflow_id="official-benchmark-registry",
        label="Official Benchmark Registry",
        summary="Run the official deterministic benchmark suite and report error, timing, memory, mode budget, identifiability, and backend comparison evidence.",
        surfaces=WorkflowSurfaceBindings(
            python="run_benchmark_registry(...)",
        ),
        artifact_story="Writes benchmark_registry.json, benchmark_cases.csv, and artifacts.json through the shared artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="inspect-service-status",
        label="Inspect Service Status",
        summary="Inspect shared uptime, task counters, and recent execution history for machine-facing surfaces.",
        surfaces=WorkflowSurfaceBindings(
            python="inspect_service_status()",
            mcp="inspect_service_status",
            api="GET /status",
        ),
    ),
    WorkflowIdentity(
        workflow_id="analyze-profile-table",
        label="Analyze Profile Table",
        summary="Project a profile table into the bounded-domain modal basis and summarize its spectral structure.",
        surfaces=WorkflowSurfaceBindings(
            python="analyze_profile_table_spectra(...)",
            cli="analyze-profile-table",
            mcp="analyze_profile_table",
            api="POST /profiles/analyze",
        ),
    ),
    WorkflowIdentity(
        workflow_id="compress-profile-table",
        label="Compress Profile Table",
        summary="Compress a profile table into modal coefficients and reconstructions through the shared engine.",
        surfaces=WorkflowSurfaceBindings(
            python="compress_profile_table(...)",
            cli="compress-profile-table",
            mcp="compress_profile_table",
            api="POST /profiles/compress",
        ),
        artifact_story="Writes coefficient and reconstruction artifacts through the shared artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="export-feature-table",
        label="Export Feature Table",
        summary="Export modal coefficients and profile moments from a profile table into a traceable feature table.",
        surfaces=WorkflowSurfaceBindings(
            python="export_feature_table_from_profile_table(...)",
            cli="export-features",
            mcp="export_feature_table",
            api="POST /features/export",
        ),
        artifact_story="Writes a feature table, explicit feature schema, and an artifacts.json manifest through the shared artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="compare-profile-tables",
        label="Compare Profile Tables",
        summary="Compare two profile tables with shared bounded-domain error metrics and diagnostics.",
        surfaces=WorkflowSurfaceBindings(
            python="compare_profile_tables(...)",
            cli="compare-profile-tables",
            mcp="compare_profile_tables",
            api="POST /profiles/compare",
        ),
    ),
    WorkflowIdentity(
        workflow_id="fit-profile-table",
        label="Fit Profile Table",
        summary="Fit Gaussian packet parameters to an observed profile table through the shared inverse workflow.",
        surfaces=WorkflowSurfaceBindings(
            python="fit_gaussian_packet_to_profile_table(...)",
            cli="fit-profile-table",
            mcp="fit_packet_to_profile_table",
            api="POST /inverse/fit",
        ),
        artifact_story="Writes inverse-fit artifacts that capture fitted packet parameters, optimization state, posterior uncertainty, identifiability, and reconstruction diagnostics.",
    ),
    WorkflowIdentity(
        workflow_id="fit-profile-table-from-sql",
        label="SQL Fit Profile Table",
        summary="Materialize a profile-table-shaped SQL query and fit Gaussian packet parameters through the same inverse workflow.",
        surfaces=WorkflowSurfaceBindings(
            python="fit_gaussian_packet_to_profile_table_from_database_query(...)",
            cli="sql-fit-table",
            mcp="fit_packet_to_database_profile_query",
            api="POST /inverse/fit-from-sql",
        ),
        artifact_story="Writes the same uncertainty-aware inverse-fit bundle as file-backed runs and records SQL provenance in the artifact metadata.",
    ),
    WorkflowIdentity(
        workflow_id="infer-potential-spectrum",
        label="Infer Potential Spectrum",
        summary="Compare explicit bounded-domain potential families against an observed low-lying spectrum and report family ranking plus local uncertainty.",
        surfaces=WorkflowSurfaceBindings(
            python="infer_potential_family_from_spectrum(...)",
            cli="infer-potential-spectrum",
            mcp="infer_potential_spectrum",
        ),
        artifact_story="Writes family ranking, best-fit calibration, and posterior/sensitivity artifacts through the shared artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="analyze-separable-spectrum",
        label="Analyze Separable Spectrum",
        summary="Analyze a controlled separable tensor-product spectrum built from two explicit 1D bounded-domain components.",
        surfaces=WorkflowSurfaceBindings(
            python="analyze_separable_tensor_product_spectrum(...)",
            cli="analyze-separable-spectrum",
            mcp="analyze_separable_spectrum",
        ),
        artifact_story="Writes combined spectrum and reduced-model metadata without claiming unrestricted multidimensional support.",
    ),
    WorkflowIdentity(
        workflow_id="solve-radial-reduction",
        label="Solve Radial Reduction",
        summary="Solve a bounded radial effective-coordinate reduction with an explicit centrifugal term and base potential family.",
        surfaces=WorkflowSurfaceBindings(
            python="solve_radial_reduction(...)",
            cli="solve-radial-reduction",
            mcp="solve_radial_reduction",
        ),
        artifact_story="Writes radial effective-potential summaries and eigenvalue outputs through the shared artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="design-transition",
        label="Design Transition",
        summary="Perform differentiable inverse design so a selected spectral transition matches a target value for one explicit potential family.",
        surfaces=WorkflowSurfaceBindings(
            python="design_potential_for_target_transition(...)",
            cli="design-transition",
            mcp="design_transition",
        ),
        artifact_story="Writes transition-design spectra and parameter gradients through the shared differentiable artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="optimize-packet-control",
        label="Optimize Packet Control",
        summary="Optimize Gaussian packet preparation parameters to steer a target observable through differentiable spectral propagation.",
        surfaces=WorkflowSurfaceBindings(
            python="optimize_packet_control(...)",
            cli="optimize-packet-control",
            mcp="optimize_packet_control",
        ),
        artifact_story="Writes optimization history, final density, and objective-gradient artifacts through the shared differentiable artifact layer.",
    ),
    WorkflowIdentity(
        workflow_id="transport-workflow",
        label="Transport Workflow",
        summary="Run the barrier/resonance vertical workflow that combines scattering, WKB comparison, propagation, and Wigner diagnostics.",
        surfaces=WorkflowSurfaceBindings(
            python="run_transport_resonance_workflow(...)",
            cli="transport-workflow",
            mcp="transport_workflow",
        ),
        artifact_story="Writes one transport-focused vertical bundle with resonance tables and tunneling diagnostics.",
    ),
    WorkflowIdentity(
        workflow_id="profile-inference-workflow",
        label="Profile Inference Workflow",
        summary="Run the report-first tabular vertical that combines profile reporting, inverse fitting with uncertainty, and spectral feature export.",
        surfaces=WorkflowSurfaceBindings(
            python="run_profile_inference_workflow(...)",
            cli="profile-inference-workflow",
            mcp="profile_inference_workflow",
        ),
        artifact_story="Writes one vertical bundle that nests report, inverse, and feature artifacts under a shared provenance root.",
    ),
    WorkflowIdentity(
        workflow_id="profile-table-report",
        label="Profile Table Report",
        summary="Validate a profile table, inspect it, analyze modal structure, compress it, and write one artifact-backed report.",
        surfaces=WorkflowSurfaceBindings(
            python="load_profile_table_report(...)",
            cli="profile-report",
            mcp="profile_table_report",
            api="POST /profiles/report",
        ),
        artifact_story=(
            "Writes one root bundle with profile_table_report.json, profile_table_summary.json, "
            "analysis/, compression/, and a root artifacts.json manifest."
        ),
    ),
    WorkflowIdentity(
        workflow_id="profile-table-report-from-sql",
        label="SQL Profile Report",
        summary="Materialize a profile-table-shaped SQL query through explicit table controls and run the same report workflow.",
        surfaces=WorkflowSurfaceBindings(
            python="build_profile_table_report_from_database_query(...)",
            cli="sql-profile-report",
            mcp="report_database_profile_query",
            api="POST /profiles/report-from-sql",
        ),
        artifact_story=(
            "Writes the same report bundle as file-backed runs and records SQL provenance in the root artifact metadata."
        ),
    ),
    WorkflowIdentity(
        workflow_id="export-feature-table-from-sql",
        label="SQL Feature Export",
        summary="Materialize a profile-table-shaped SQL query and export the same traceable spectral feature table.",
        surfaces=WorkflowSurfaceBindings(
            python="export_feature_table_from_database_query(...)",
            cli="sql-export-features",
            mcp="export_feature_table_from_sql",
            api="POST /features/export-from-sql",
        ),
        artifact_story="Writes the same feature-table bundle as file-backed runs and records SQL provenance in the artifact metadata.",
    ),
    WorkflowIdentity(
        workflow_id="inspect-artifacts",
        label="Inspect Artifacts",
        summary="Inspect a managed artifact directory and report completeness, metadata, and files without re-running compute.",
        surfaces=WorkflowSurfaceBindings(
            python="inspect_artifact_directory(...)",
            cli="inspect-artifacts",
            mcp="list_artifacts",
            api="GET /artifacts",
        ),
    ),
    WorkflowIdentity(
        workflow_id="modal-surrogate-evaluate",
        label="Modal Surrogate Evaluate",
        summary="Run the backend-aware modal surrogate as an extension over the same profile-table workflows.",
        surfaces=WorkflowSurfaceBindings(
            python="evaluate_modal_surrogate_on_profile_table(...)",
            cli="ml-evaluate-table",
            mcp="evaluate_modal_surrogate",
            api="POST /ml/evaluate",
        ),
        artifact_story="Writes backend-tagged modal evaluation artifacts over the shared profile-table boundary.",
    ),
    WorkflowIdentity(
        workflow_id="tree-model-train",
        label="Tree Model Train",
        summary="Train a tree model on a feature table through the shared workflow layer with explicit predictors, target, and artifacts.",
        surfaces=WorkflowSurfaceBindings(
            python="train_tree_model(...)",
            cli="tree-train",
            mcp="train_tree_model",
            api="POST /tree/train",
        ),
        artifact_story="Writes model export, predictions, feature importance, and a root artifacts.json manifest.",
    ),
    WorkflowIdentity(
        workflow_id="tree-model-tune",
        label="Tree Model Tune",
        summary="Tune a tree model on a feature table with explicit search space and holdout evaluation.",
        surfaces=WorkflowSurfaceBindings(
            python="tune_tree_model(...)",
            cli="tree-tune",
            mcp="tune_tree_model",
            api="POST /tree/tune",
        ),
        artifact_story="Writes tuning results, the best-model bundle, and a root artifacts.json manifest.",
    ),
)

_KILLER_WORKFLOW_CATALOG: tuple[KillerWorkflowDefinition, ...] = (
    KillerWorkflowDefinition(
        killer_workflow_id="spectral-evidence-loop",
        goal="report",
        label="Spectral Evidence Loop",
        target_user=(
            "Scientific or engineering user who needs one defensible spectral answer from raw file-backed "
            "or SQL-backed profile data without writing custom orchestration glue."
        ),
        input_types=(
            "Profile table file in CSV, TSV, JSON, or optionally XLSX format",
            "Profile-table-shaped SQL query with explicit time and position controls",
        ),
        steps=(
            "Load and validate the profile table shape and numeric quality.",
            "Inspect bounded-domain moments, spectral budgets, and modal structure through the shared basis.",
            "Compress the table into modal coefficients and a reconstruction.",
            "Write one stable artifact bundle and inspect it for completeness before any follow-on work.",
        ),
        outputs=(
            "profile_table_report.json",
            "profile_table_summary.json",
            "analysis/spectral_analysis.json",
            "compression/compression_summary.json",
            "artifacts.json",
        ),
        entry_workflow_ids=("profile-table-report", "profile-table-report-from-sql", "inspect-artifacts"),
        why_better_here=(
            "One engine owns validation, modal analysis, compression, and artifact production, so the user "
            "does not need to wire together loaders, basis projection code, numerical diagnostics, "
            "reconstruction logic, and reporting by hand."
        ),
    ),
    KillerWorkflowDefinition(
        killer_workflow_id="inverse-reconstruction-loop",
        goal="inverse-fit",
        label="Inverse Reconstruction Loop",
        target_user=(
            "User with measured or simulated densities who needs interpretable Gaussian packet parameters "
            "instead of only coefficients or black-box predictions."
        ),
        input_types=(
            "Profile table file in CSV, TSV, JSON, or optionally XLSX format",
            "Profile-table-shaped SQL query with explicit time and position controls",
        ),
        steps=(
            "Run the report-first spectral evidence loop so inverse work starts from validated, inspectable data.",
            "Fit Gaussian packet parameters through the bounded-domain inverse workflow instead of generic curve fitting.",
            "Inspect fit quality and reconstruction diagnostics before re-running with different assumptions.",
            "Persist inverse artifacts with the same provenance chain as the spectral report.",
        ),
        outputs=(
            "profile_table_report.json",
            "inverse_summary.json",
            "inverse_reconstruction.csv",
            "artifacts.json",
        ),
        entry_workflow_ids=(
            "profile-table-report",
            "profile-table-report-from-sql",
            "fit-profile-table",
            "fit-profile-table-from-sql",
            "inspect-artifacts",
        ),
        why_better_here=(
            "The same bounded-domain engine that explains the data spectrally also fits the packet parameters, "
            "so the user gets interpretable inverse reconstruction instead of hand-rolled optimization over raw tables."
        ),
    ),
    KillerWorkflowDefinition(
        killer_workflow_id="spectral-feature-model-loop",
        goal="feature-model",
        label="Spectral Feature Model Loop",
        target_user=(
            "Data or applied-ML user who wants a predictive model over profile-table-shaped results "
            "without inventing feature engineering, provenance, or artifact conventions from scratch."
        ),
        input_types=(
            "Profile table file in CSV, TSV, JSON, or optionally XLSX format",
            "Profile-table-shaped SQL query with explicit time and position controls",
            "One supervised target column joined after feature export",
        ),
        steps=(
            "Run the report-first spectral evidence loop so model work begins from numerically healthy data.",
            "Export one traceable feature table with coefficients, moments, schema, and provenance.",
            "Join or append the supervised target explicitly instead of hiding labels inside wrappers.",
            "Train or tune a tree model through the shared workflow layer and inspect the resulting artifacts.",
        ),
        outputs=(
            "features.csv or features.parquet",
            "features_schema.json",
            "tree_training.json or tree_tuning.json",
            "predictions.csv",
            "artifacts.json",
        ),
        entry_workflow_ids=(
            "profile-table-report",
            "profile-table-report-from-sql",
            "export-feature-table",
            "export-feature-table-from-sql",
            "tree-model-train",
            "tree-model-tune",
            "inspect-artifacts",
        ),
        why_better_here=(
            "The product turns spectral compression into an explicit, reproducible feature contract with artifacts, "
            "so the user does not have to invent a one-off bridge from scientific profiles to predictive models."
        ),
    ),
)


def workflow_catalog() -> tuple[WorkflowIdentity, ...]:
    return _WORKFLOW_CATALOG


WorkflowSurfaceName = Literal["python", "cli", "mcp", "api"]
WorkflowInputKind = Literal["profile-table-file", "profile-table-sql"]


def killer_workflow_catalog() -> tuple[KillerWorkflowDefinition, ...]:
    return _KILLER_WORKFLOW_CATALOG


def _report_workflow_id(input_kind: WorkflowInputKind) -> str:
    return "profile-table-report-from-sql" if input_kind == "profile-table-sql" else "profile-table-report"


def _feature_export_workflow_id(input_kind: WorkflowInputKind) -> str:
    return "export-feature-table-from-sql" if input_kind == "profile-table-sql" else "export-feature-table"


def _inverse_fit_workflow_id(input_kind: WorkflowInputKind) -> str:
    return "fit-profile-table-from-sql" if input_kind == "profile-table-sql" else "fit-profile-table"


def _binding_for(surface: WorkflowSurfaceName, workflow_id: str) -> str:
    binding = getattr(_workflow_identity_by_id(workflow_id).surfaces, surface)
    return "" if binding is None else binding


def _workflow_identity_by_id(workflow_id: str) -> WorkflowIdentity:
    for workflow in _WORKFLOW_CATALOG:
        if workflow.workflow_id == workflow_id:
            return workflow
    raise KeyError(f"unknown workflow id: {workflow_id}")


def _killer_workflow_by_id(killer_workflow_id: str) -> KillerWorkflowDefinition:
    for workflow in _KILLER_WORKFLOW_CATALOG:
        if workflow.killer_workflow_id == killer_workflow_id:
            return workflow
    raise KeyError(f"unknown killer workflow id: {killer_workflow_id}")


def opinionated_defaults() -> dict[str, Any]:
    return {
        "workflow_routing": {
            "default_goal": "report",
            "report_before_inverse_fit": True,
            "report_before_feature_model": True,
            "inspect_artifacts_after_compute": True,
            "mcp_prefers_intent_level_workflows": True,
        },
        "profile_report": {
            "device": DEFAULT_PROFILE_REPORT_DEVICE,
            "analyze_num_modes": DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
            "compress_num_modes": DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
            "capture_thresholds": list(DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS),
            "normalize_each_profile": DEFAULT_PROFILE_REPORT_NORMALIZE_EACH_PROFILE,
            "python_output_dir": DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR,
            "operational_output_dir": DEFAULT_PROFILE_REPORT_OUTPUT_DIR,
        },
        "sql_profile_report": {
            "device": DEFAULT_PROFILE_REPORT_DEVICE,
            "analyze_num_modes": DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES,
            "compress_num_modes": DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES,
            "capture_thresholds": list(DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS),
            "normalize_each_profile": DEFAULT_PROFILE_REPORT_NORMALIZE_EACH_PROFILE,
            "time_column": DEFAULT_SQL_PROFILE_TIME_COLUMN,
            "sort_by_time": DEFAULT_SQL_PROFILE_SORT_BY_TIME,
            "operational_output_dir": DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR,
        },
        "inverse_fit": {
            "initial_center": 0.36,
            "initial_width": 0.11,
            "initial_wavenumber": 22.0,
            "initial_phase": 0.0,
            "num_modes": 128,
            "quadrature_points": 2048,
            "steps": 200,
            "learning_rate": 0.05,
            "inverse_output_dir": DEFAULT_INVERSE_FIT_OUTPUT_DIR,
            "sql_inverse_output_dir": DEFAULT_SQL_INVERSE_FIT_OUTPUT_DIR,
        },
        "feature_export": {
            "num_modes": DEFAULT_FEATURE_EXPORT_NUM_MODES,
            "format": DEFAULT_FEATURE_EXPORT_FORMAT,
            "include": ["coefficients", "moments"],
            "feature_output_dir": DEFAULT_FEATURE_EXPORT_OUTPUT_DIR,
            "sql_feature_output_dir": DEFAULT_SQL_FEATURE_EXPORT_OUTPUT_DIR,
        },
        "tree_model": {
            "library": DEFAULT_TREE_MODEL_LIBRARY,
            "task": DEFAULT_TREE_MODEL_TASK,
            "test_fraction": DEFAULT_TREE_MODEL_TEST_FRACTION,
            "random_state": DEFAULT_TREE_MODEL_RANDOM_STATE,
            "train_output_dir": DEFAULT_TREE_MODEL_OUTPUT_DIR,
            "tune_output_dir": DEFAULT_TREE_TUNE_OUTPUT_DIR,
        },
        "mcp_runtime": {
            "max_concurrent_tasks": DEFAULT_MCP_MAX_CONCURRENT_TASKS,
            "log_level": DEFAULT_MCP_LOG_LEVEL,
        },
    }


def resolve_workflow_identity(surface: WorkflowSurfaceName, binding: str) -> WorkflowIdentity | None:
    token = str(binding)
    for workflow in _WORKFLOW_CATALOG:
        candidate = getattr(workflow.surfaces, surface)
        if candidate == token:
            return workflow
    return None


def resolve_workflow_id(surface: WorkflowSurfaceName, binding: str) -> str | None:
    workflow = resolve_workflow_identity(surface, binding)
    return None if workflow is None else workflow.workflow_id


def guide_workflow(
    *,
    surface: WorkflowSurfaceName = "python",
    input_kind: WorkflowInputKind = "profile-table-file",
    goal: WorkflowGoal = "report",
) -> WorkflowGuidance:
    if surface not in {"python", "cli", "mcp", "api"}:
        raise ValueError(f"unsupported surface: {surface}")
    if input_kind not in {"profile-table-file", "profile-table-sql"}:
        raise ValueError(f"unsupported input_kind: {input_kind}")
    if goal not in {"report", "inverse-fit", "feature-model"}:
        raise ValueError(f"unsupported goal: {goal}")

    defaults = opinionated_defaults()
    report_workflow_id = _report_workflow_id(input_kind)
    feature_export_workflow_id = _feature_export_workflow_id(input_kind)
    inverse_fit_workflow_id = _inverse_fit_workflow_id(input_kind)
    primary_workflow = _workflow_identity_by_id(report_workflow_id)
    inspect_artifacts_workflow = _workflow_identity_by_id("inspect-artifacts")
    report_defaults = defaults["sql_profile_report"] if input_kind == "profile-table-sql" else defaults["profile_report"]
    report_output_dir = (
        DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR
        if input_kind == "profile-table-sql"
        else (DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR if surface == "python" else DEFAULT_PROFILE_REPORT_OUTPUT_DIR)
    )

    if goal == "report":
        killer_workflow = _killer_workflow_by_id("spectral-evidence-loop")
        follow_up_workflow_ids = (
            "compare-profile-tables",
            inverse_fit_workflow_id,
            "modal-surrogate-evaluate",
            feature_export_workflow_id,
            "tree-model-train",
        )
        why_this_path = (
            "Start with the spectral evidence loop because it removes the most glue for new data: one file or one SQL query "
            "enters the engine, the engine validates structure, analyzes modal content, compresses it, and writes an inspectable artifact chain."
        )
    elif goal == "inverse-fit":
        killer_workflow = _killer_workflow_by_id("inverse-reconstruction-loop")
        follow_up_workflow_ids = (
            inverse_fit_workflow_id,
            "compare-profile-tables",
            "analyze-profile-table",
        )
        why_this_path = (
            "Stay report-first even for inverse work so packet fitting starts from validated spectral evidence rather than directly from raw rows."
        )
    else:
        killer_workflow = _killer_workflow_by_id("spectral-feature-model-loop")
        follow_up_workflow_ids = (
            feature_export_workflow_id,
            "inspect-tree-backends",
            "tree-model-train",
            "tree-model-tune",
        )
        why_this_path = (
            "Use the report-first feature-model loop so predictive work inherits a stable spectral feature contract, provenance, and artifact history instead of ad hoc feature scripts."
        )

    follow_up_workflows = tuple(_workflow_identity_by_id(workflow_id) for workflow_id in follow_up_workflow_ids)

    if surface == "python":
        plan_steps: tuple[WorkflowGuidanceStep, ...] = (
            WorkflowGuidanceStep(
                stage="compute",
                action=_binding_for("python", report_workflow_id),
                detail="Build the shared profile report object first and let the engine validate structure before any narrower workflow.",
                workflow_id=report_workflow_id,
            ),
            WorkflowGuidanceStep(
                stage="artifacts",
                action="report.write_artifacts(...)",
                detail="Persist the stable artifact bundle and get an immediate completeness report back.",
            ),
        )
        if goal == "inverse-fit":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="inverse",
                    action=_binding_for("python", inverse_fit_workflow_id),
                    detail="Run the inverse reconstruction workflow only after the report confirms the observed table is healthy.",
                    workflow_id=inverse_fit_workflow_id,
                ),
            )
        elif goal == "feature-model":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="features",
                    action=_binding_for("python", feature_export_workflow_id),
                    detail="Export the explicit spectral feature table and schema before any model-specific work.",
                    workflow_id=feature_export_workflow_id,
                ),
                WorkflowGuidanceStep(
                    stage="labels",
                    action="join target column",
                    detail="Add the supervised target explicitly so the feature table remains traceable and inspectable.",
                ),
                WorkflowGuidanceStep(
                    stage="model",
                    action=_binding_for("python", "tree-model-train"),
                    detail="Train one explicit tree-model backend through the shared workflow layer.",
                    workflow_id="tree-model-train",
                ),
            )
        plan_steps += (
            WorkflowGuidanceStep(
                stage="inspect",
                action=_binding_for("python", "inspect-artifacts"),
                detail="Inspect a previously written artifact bundle later without recomputing.",
                workflow_id="inspect-artifacts",
            ),
        )
    elif surface == "cli":
        plan_steps = (
            WorkflowGuidanceStep(
                stage="validate",
                action=_binding_for("cli", "validate-installation"),
                detail="Confirm the environment once before running a new machine, backend combination, or feature stack.",
                workflow_id="validate-installation",
            ),
            WorkflowGuidanceStep(
                stage="compute",
                action=_binding_for("cli", report_workflow_id),
                detail="Run the opinionated report workflow and write one reference-grade bundle.",
                workflow_id=report_workflow_id,
            ),
        )
        if goal == "inverse-fit":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="inverse",
                    action=_binding_for("cli", inverse_fit_workflow_id),
                    detail="Use the bounded inverse workflow after the report instead of guessing packet parameters from raw rows directly.",
                    workflow_id=inverse_fit_workflow_id,
                ),
            )
        elif goal == "feature-model":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="features",
                    action=_binding_for("cli", feature_export_workflow_id),
                    detail="Export a traceable spectral feature table with schema and provenance.",
                    workflow_id=feature_export_workflow_id,
                ),
                WorkflowGuidanceStep(
                    stage="labels",
                    action="join target column",
                    detail="Add the supervised target column explicitly before model training.",
                ),
                WorkflowGuidanceStep(
                    stage="model",
                    action=_binding_for("cli", "tree-model-train"),
                    detail="Train one explicit tree-model backend through the shared workflow layer.",
                    workflow_id="tree-model-train",
                ),
            )
        plan_steps += (
            WorkflowGuidanceStep(
                stage="inspect",
                action=_binding_for("cli", "inspect-artifacts"),
                detail="Inspect the artifact bundle for completeness and provenance after the run.",
                workflow_id="inspect-artifacts",
            ),
        )
    elif surface == "api":
        plan_steps = (
            WorkflowGuidanceStep(
                stage="inspect",
                action="GET /product",
                detail="Inspect the shared product contract exposed by the API before composing a workflow request.",
                workflow_id="inspect-product",
            ),
            WorkflowGuidanceStep(
                stage="compute",
                action=_binding_for("api", report_workflow_id),
                detail="Post one report request instead of wiring separate inspection, analysis, and compression routes.",
                workflow_id=report_workflow_id,
            ),
        )
        if goal == "inverse-fit":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="inverse",
                    action=_binding_for("api", inverse_fit_workflow_id),
                    detail="Use the inverse route after the report if you need packet parameters, not only diagnostics.",
                    workflow_id=inverse_fit_workflow_id,
                ),
            )
        elif goal == "feature-model":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="features",
                    action=_binding_for("api", feature_export_workflow_id),
                    detail="Export the explicit feature contract before model work.",
                    workflow_id=feature_export_workflow_id,
                ),
                WorkflowGuidanceStep(
                    stage="labels",
                    action="join target column",
                    detail="Attach the supervised target outside the API only if labels live elsewhere; keep that step explicit.",
                ),
                WorkflowGuidanceStep(
                    stage="model",
                    action=_binding_for("api", "tree-model-train"),
                    detail="Run one explicit model-training request over the feature table.",
                    workflow_id="tree-model-train",
                ),
            )
        plan_steps += (
            WorkflowGuidanceStep(
                stage="inspect",
                action=_binding_for("api", "inspect-artifacts"),
                detail="Inspect the artifact bundle or rerun status without touching the compute path.",
                workflow_id="inspect-artifacts",
            ),
        )
    else:
        plan_steps = (
            WorkflowGuidanceStep(
                stage="inspect",
                action="inspect_product",
                detail="Load the shared product spine, killer workflows, and defaults before issuing compute requests.",
                workflow_id="inspect-product",
            ),
            WorkflowGuidanceStep(
                stage="validate",
                action="inspect_environment",
                detail="Inspect machine capabilities and bounded-execution policy through MCP.",
                workflow_id="inspect-environment",
            ),
            WorkflowGuidanceStep(
                stage="compute",
                action=_binding_for("mcp", report_workflow_id),
                detail="Run the report-first workflow as one bounded tool call instead of chaining low-level tools manually.",
                workflow_id=report_workflow_id,
            ),
        )
        if goal == "inverse-fit":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="inverse",
                    action=_binding_for("mcp", inverse_fit_workflow_id),
                    detail="Continue with inverse reconstruction only after the report succeeds and artifacts are inspectable.",
                    workflow_id=inverse_fit_workflow_id,
                ),
            )
        elif goal == "feature-model":
            plan_steps += (
                WorkflowGuidanceStep(
                    stage="features",
                    action=_binding_for("mcp", feature_export_workflow_id),
                    detail="Ask MCP to export the spectral feature table rather than building a prompt-side feature pipeline.",
                    workflow_id=feature_export_workflow_id,
                ),
                WorkflowGuidanceStep(
                    stage="labels",
                    action="join target column",
                    detail="Add or join the supervised target explicitly so the model input remains inspectable.",
                ),
                WorkflowGuidanceStep(
                    stage="model",
                    action=_binding_for("mcp", "tree-model-train"),
                    detail="Train the model through one bounded tool instead of ad hoc shell glue.",
                    workflow_id="tree-model-train",
                ),
            )
        plan_steps += (
            WorkflowGuidanceStep(
                stage="inspect",
                action="list_artifacts",
                detail="Verify artifact completeness and recorded metadata after the run.",
                workflow_id="inspect-artifacts",
            ),
            WorkflowGuidanceStep(
                stage="continue",
                action="inspect_service_status",
                detail="Check recent runs and runtime state before continuing or retrying.",
                workflow_id="inspect-service-status",
            ),
        )

    guidance_defaults: dict[str, Any] = {
        **defaults["workflow_routing"],
        **report_defaults,
        "goal": goal,
        "report_output_dir": report_output_dir,
        "output_dir": report_output_dir,
    }
    if surface == "mcp":
        guidance_defaults = {**defaults["mcp_runtime"], **guidance_defaults}
    if goal == "inverse-fit":
        guidance_defaults = {**guidance_defaults, **defaults["inverse_fit"]}
    elif goal == "feature-model":
        guidance_defaults = {**guidance_defaults, **defaults["feature_export"], **defaults["tree_model"]}

    notes = [
        "Use the report workflow as the default first answer; drop to lower-level analysis or compression only when you already know why.",
        "Inspect artifacts before continuing so follow-up workflows build on recorded evidence instead of memory or prompt state.",
    ]
    if goal == "inverse-fit":
        notes.append("Do not start by tuning inverse-fit hyperparameters blindly; let the report tell you whether the table is numerically healthy first.")
    elif goal == "feature-model":
        notes.append("Keep target joining explicit; the product owns feature generation and provenance, not hidden label fusion.")
    if surface == "mcp":
        notes.append("Prefer the intent-level tools in the recommended order instead of composing long prompt-side scripts around low-level tool fragments.")

    return WorkflowGuidance(
        surface=surface,
        input_kind=input_kind,
        goal=goal,
        killer_workflow=killer_workflow,
        primary_workflow=primary_workflow,
        why_this_path=why_this_path,
        defaults=guidance_defaults,
        plan_steps=plan_steps,
        follow_up_workflows=follow_up_workflows,
        notes=tuple(notes),
    )


def inspect_product_identity() -> ProductIdentityReport:
    hero_workflow = next(
        workflow for workflow in _WORKFLOW_CATALOG if workflow.workflow_id == "profile-table-report"
    )
    return ProductIdentityReport(
        product_name=PRODUCT_NAME,
        product_spine=PRODUCT_SPINE_STATEMENT,
        runtime_spine=RUNTIME_SPINE_STATEMENT,
        hero_workflow=hero_workflow,
        workflows=_WORKFLOW_CATALOG,
        killer_workflows=_KILLER_WORKFLOW_CATALOG,
        opinionated_defaults=opinionated_defaults(),
        replaceability_risks=(
            "Without stronger goal-aware routing, the product can still look like a set of technically strong commands organized by transport boundary instead of by user outcome.",
            "Feature modeling and inverse reconstruction still risk feeling like optional add-ons unless the product keeps routing users through report-first loops.",
            "Hidden or uncatalogued workflows reduce the credibility of the one-engine story and make custom glue feel safer than trusting the product defaults.",
        ),
        decision_burdens=(
            "Users still need help choosing between report, inverse-fit, and feature-model intents instead of only choosing between file and SQL inputs.",
            "Machine-side clients need clearer next-step guidance so they prefer high-value workflows over low-level tool chaining.",
            "Follow-up workflow order should come from the product, not from ad hoc prompt or shell decisions.",
        ),
        glue_burdens=(
            "Users still have to join model labels explicitly after feature export.",
            "Cross-step continuation still depends on artifact inspection and workflow guidance rather than a single end-to-end chained workflow.",
            "Some adoption-critical workflows existed in code before they were visible in the shared product catalog.",
        ),
        adoption_priorities=(
            "Make report-first, inverse reconstruction, and spectral feature-model loops the default product language across Python, CLI, MCP, and API.",
            "Push more intent-aware workflow guidance into MCP so clients ask for outcomes instead of inventing tool sequences.",
            "Keep artifacts as the continuity layer so every high-value loop ends with inspectable outputs and a clear next step.",
        ),
        notes=(
            "Python is the primary surface; CLI, MCP, and API wrap the same shared workflows.",
            "File-backed and SQL-backed inputs converge through explicit table materialization before entering spectral workflows.",
            "Modal-surrogate and tree-model workflows remain extensions over the spectral spine rather than a separate product identity.",
            "The default adoption path is report-first: validate structure, analyze modes, compress, write artifacts, then choose inverse or feature-model follow-up workflows from evidence.",
        ),
    )


__all__ = [
    "DEFAULT_FEATURE_EXPORT_FORMAT",
    "DEFAULT_FEATURE_EXPORT_NUM_MODES",
    "DEFAULT_FEATURE_EXPORT_OUTPUT_DIR",
    "DEFAULT_INVERSE_FIT_OUTPUT_DIR",
    "DEFAULT_MCP_LOG_LEVEL",
    "DEFAULT_MCP_MAX_CONCURRENT_TASKS",
    "DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES",
    "DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS",
    "DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES",
    "DEFAULT_PROFILE_REPORT_DEVICE",
    "DEFAULT_PROFILE_REPORT_NORMALIZE_EACH_PROFILE",
    "DEFAULT_PROFILE_REPORT_OUTPUT_DIR",
    "DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR",
    "DEFAULT_SQL_FEATURE_EXPORT_OUTPUT_DIR",
    "DEFAULT_SQL_INVERSE_FIT_OUTPUT_DIR",
    "DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR",
    "DEFAULT_SQL_PROFILE_SORT_BY_TIME",
    "DEFAULT_SQL_PROFILE_TIME_COLUMN",
    "DEFAULT_TREE_MODEL_LIBRARY",
    "DEFAULT_TREE_MODEL_OUTPUT_DIR",
    "DEFAULT_TREE_MODEL_RANDOM_STATE",
    "DEFAULT_TREE_MODEL_TASK",
    "DEFAULT_TREE_MODEL_TEST_FRACTION",
    "DEFAULT_TREE_TUNE_OUTPUT_DIR",
    "PRODUCT_NAME",
    "PRODUCT_SPINE_STATEMENT",
    "RUNTIME_SPINE_STATEMENT",
    "KillerWorkflowDefinition",
    "ProductIdentityReport",
    "WorkflowGoal",
    "WorkflowGuidance",
    "WorkflowGuidanceStep",
    "WorkflowIdentity",
    "WorkflowInputKind",
    "WorkflowSurfaceName",
    "WorkflowSurfaceBindings",
    "guide_workflow",
    "inspect_product_identity",
    "killer_workflow_catalog",
    "opinionated_defaults",
    "resolve_workflow_id",
    "resolve_workflow_identity",
    "workflow_catalog",
]
