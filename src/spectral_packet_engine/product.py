from __future__ import annotations

"""Shared product identity, opinionated defaults, and workflow guidance."""

from dataclasses import dataclass
from typing import Any, Literal


PRODUCT_NAME = "Spectral Packet Engine"
PRODUCT_SPINE_STATEMENT = (
    "A Python-first bounded-domain spectral computation library for packet simulation, "
    "modal analysis, profile compression, and inverse reconstruction, with file, SQL, "
    "CLI, MCP, and API surfaces over the same engine."
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
DEFAULT_MCP_MAX_CONCURRENT_TASKS = 1
DEFAULT_MCP_LOG_LEVEL = "warning"


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
)

_KILLER_WORKFLOW_CATALOG: tuple[KillerWorkflowDefinition, ...] = (
    KillerWorkflowDefinition(
        killer_workflow_id="file-profile-report-loop",
        label="File Profile Report Loop",
        target_user=(
            "Python or CLI user with one profile-table file who wants a validated, "
            "artifact-backed spectral answer without custom glue."
        ),
        input_types=(
            "Profile table file in CSV, TSV, JSON, or optionally XLSX format",
        ),
        steps=(
            "Load and validate the profile table shape and numeric quality.",
            "Inspect bounded-domain moments and modal structure through the shared basis.",
            "Compress the table into modal coefficients and a reconstruction.",
            "Write one stable artifact bundle and inspect it for completeness.",
        ),
        outputs=(
            "profile_table_report.json",
            "profile_table_summary.json",
            "analysis/spectral_analysis.json",
            "compression/compression_summary.json",
            "artifacts.json",
        ),
        entry_workflow_ids=("profile-table-report", "inspect-artifacts"),
        why_better_here=(
            "One engine owns validation, modal analysis, compression, and artifact production, "
            "so the user does not need to wire together loaders, basis projection code, "
            "reconstruction logic, and reporting by hand."
        ),
    ),
    KillerWorkflowDefinition(
        killer_workflow_id="sql-profile-report-loop",
        label="SQL Profile Report Loop",
        target_user=(
            "User with profile-table-shaped data in SQLite or SQLAlchemy-backed storage "
            "who wants the same spectral report without exporting intermediate files."
        ),
        input_types=(
            "Database URL or local SQLite path",
            "Profile-table-shaped SQL query with one time column and position columns",
        ),
        steps=(
            "Run a parameterized SQL query through explicit profile-table materialization controls.",
            "Sort or preserve row order through a declared policy instead of hidden wrapper logic.",
            "Run the same inspect-analyze-compress report workflow as file-backed inputs.",
            "Write one artifact bundle that records SQL provenance and report outputs together.",
        ),
        outputs=(
            "profile_table_report.json",
            "profile_table_summary.json",
            "analysis/spectral_analysis.json",
            "compression/compression_summary.json",
            "artifacts.json with SQL provenance",
        ),
        entry_workflow_ids=("profile-table-report-from-sql", "inspect-artifacts"),
        why_better_here=(
            "The SQL boundary stays explicit, validated, and reproducible, so a user can move "
            "from relational data to bounded-domain spectral results without exporting ad hoc CSVs "
            "or reimplementing query-to-grid conversion."
        ),
    ),
    KillerWorkflowDefinition(
        killer_workflow_id="mcp-operator-loop",
        label="MCP Operator Loop",
        target_user=(
            "Machine-side tool client that needs structured numerical work, clear next steps, "
            "and stable artifact handling instead of custom shell scripts."
        ),
        input_types=(
            "Profile table file path or profile-table-shaped SQL query",
            "MCP client connected over local stdio",
        ),
        steps=(
            "Inspect the shared product and runtime contract before compute starts.",
            "Inspect machine capabilities and bounded-execution policy through MCP.",
            "Run one high-value report workflow instead of chaining low-level tools manually.",
            "Inspect artifacts and service status to continue or debug the session cleanly.",
        ),
        outputs=(
            "Structured MCP responses",
            "Artifact bundle inspection",
            "Canonical service-status history with workflow ids and surface actions",
        ),
        entry_workflow_ids=(
            "inspect-product",
            "inspect-environment",
            "profile-table-report",
            "profile-table-report-from-sql",
            "inspect-artifacts",
            "inspect-service-status",
        ),
        why_better_here=(
            "The MCP layer knows the product’s intended workflow order, runtime constraints, "
            "and artifact semantics, so an AI client can operate the engine directly instead of "
            "inventing prompt glue around unrelated tools."
        ),
    ),
)


def workflow_catalog() -> tuple[WorkflowIdentity, ...]:
    return _WORKFLOW_CATALOG


WorkflowSurfaceName = Literal["python", "cli", "mcp", "api"]
WorkflowInputKind = Literal["profile-table-file", "profile-table-sql"]


def killer_workflow_catalog() -> tuple[KillerWorkflowDefinition, ...]:
    return _KILLER_WORKFLOW_CATALOG


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
) -> WorkflowGuidance:
    if surface not in {"python", "cli", "mcp", "api"}:
        raise ValueError(f"unsupported surface: {surface}")
    if input_kind not in {"profile-table-file", "profile-table-sql"}:
        raise ValueError(f"unsupported input_kind: {input_kind}")

    defaults = opinionated_defaults()
    follow_up_workflows = tuple(
        _workflow_identity_by_id(workflow_id)
        for workflow_id in (
            "compare-profile-tables",
            "fit-profile-table",
            "modal-surrogate-evaluate",
        )
    )

    if surface == "mcp":
        primary_workflow_id = "profile-table-report-from-sql" if input_kind == "profile-table-sql" else "profile-table-report"
        primary_workflow = _workflow_identity_by_id(primary_workflow_id)
        action = primary_workflow.surfaces.mcp
        assert action is not None
        return WorkflowGuidance(
            surface=surface,
            input_kind=input_kind,
            killer_workflow=_killer_workflow_by_id("mcp-operator-loop"),
            primary_workflow=primary_workflow,
            why_this_path=(
                "Use the MCP loop when a tool client needs the product to choose the high-value path "
                "instead of forcing the prompt to stitch together inspection, compute, artifacts, and status manually."
            ),
            defaults={
                **defaults["mcp_runtime"],
                **(defaults["sql_profile_report"] if input_kind == "profile-table-sql" else defaults["profile_report"]),
            },
            plan_steps=(
                WorkflowGuidanceStep(
                    stage="inspect",
                    action="inspect_product",
                    detail="Load the shared product spine, runtime model, and workflow map before issuing compute requests.",
                    workflow_id="inspect-product",
                ),
                WorkflowGuidanceStep(
                    stage="validate",
                    action="inspect_environment",
                    detail="Inspect machine capabilities and backend availability through the MCP surface.",
                    workflow_id="inspect-environment",
                ),
                WorkflowGuidanceStep(
                    stage="compute",
                    action=action,
                    detail="Run the highest-value report workflow as one bounded tool call.",
                    workflow_id=primary_workflow.workflow_id,
                ),
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
            ),
            follow_up_workflows=follow_up_workflows,
            notes=(
                "Prefer the report tools over chaining inspect/analyze/compress manually when the goal is a reference-grade output.",
                "Use list_artifacts and inspect_service_status as the default continuation path instead of re-running compute blindly.",
            ),
        )

    primary_workflow_id = "profile-table-report-from-sql" if input_kind == "profile-table-sql" else "profile-table-report"
    primary_workflow = _workflow_identity_by_id(primary_workflow_id)
    killer_workflow = _killer_workflow_by_id(
        "sql-profile-report-loop" if input_kind == "profile-table-sql" else "file-profile-report-loop"
    )
    inspect_artifacts_workflow = _workflow_identity_by_id("inspect-artifacts")
    output_defaults = defaults["sql_profile_report"] if input_kind == "profile-table-sql" else defaults["profile_report"]

    if surface == "python":
        primary_action = primary_workflow.surfaces.python
        inspect_action = inspect_artifacts_workflow.surfaces.python
        artifact_output_dir = (
            DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR if input_kind == "profile-table-sql" else DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR
        )
        plan_steps = (
            WorkflowGuidanceStep(
                stage="compute",
                action="" if primary_action is None else primary_action,
                detail="Build the shared profile report object with the product defaults.",
                workflow_id=primary_workflow.workflow_id,
            ),
            WorkflowGuidanceStep(
                stage="artifacts",
                action="report.write_artifacts(...)",
                detail="Persist the stable artifact bundle and get an immediate completeness report back.",
            ),
            WorkflowGuidanceStep(
                stage="inspect",
                action="" if inspect_action is None else inspect_action,
                detail="Inspect a previously written artifact bundle later without recomputing.",
                workflow_id="inspect-artifacts",
            ),
        )
    elif surface == "cli":
        primary_action = primary_workflow.surfaces.cli
        inspect_action = inspect_artifacts_workflow.surfaces.cli
        validate_action = _workflow_identity_by_id("validate-installation").surfaces.cli
        artifact_output_dir = (
            DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR if input_kind == "profile-table-sql" else DEFAULT_PROFILE_REPORT_OUTPUT_DIR
        )
        plan_steps = (
            WorkflowGuidanceStep(
                stage="validate",
                action="" if validate_action is None else validate_action,
                detail="Confirm the environment once before running a new machine or backend combination.",
                workflow_id="validate-installation",
            ),
            WorkflowGuidanceStep(
                stage="compute",
                action="" if primary_action is None else primary_action,
                detail="Run the opinionated report workflow and write one reference-grade bundle.",
                workflow_id=primary_workflow.workflow_id,
            ),
            WorkflowGuidanceStep(
                stage="inspect",
                action="" if inspect_action is None else inspect_action,
                detail="Inspect the artifact bundle for completeness and provenance after the run.",
                workflow_id="inspect-artifacts",
            ),
        )
    else:
        primary_action = primary_workflow.surfaces.api
        inspect_action = inspect_artifacts_workflow.surfaces.api
        artifact_output_dir = (
            DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR if input_kind == "profile-table-sql" else DEFAULT_PROFILE_REPORT_OUTPUT_DIR
        )
        plan_steps = (
            WorkflowGuidanceStep(
                stage="inspect",
                action="GET /product",
                detail="Inspect the shared product and runtime contract exposed by the API.",
                workflow_id="inspect-product",
            ),
            WorkflowGuidanceStep(
                stage="compute",
                action="" if primary_action is None else primary_action,
                detail="Post one report request instead of wiring separate inspection, analysis, and compression routes.",
                workflow_id=primary_workflow.workflow_id,
            ),
            WorkflowGuidanceStep(
                stage="inspect",
                action="" if inspect_action is None else inspect_action,
                detail="Inspect the artifact bundle or rerun status without touching the compute path.",
                workflow_id="inspect-artifacts",
            ),
        )

    return WorkflowGuidance(
        surface=surface,
        input_kind=input_kind,
        killer_workflow=killer_workflow,
        primary_workflow=primary_workflow,
        why_this_path=(
            "This path removes the most glue for the common case: one file or one SQL query enters the engine, "
            "the engine validates structure, runs modal analysis and compression, and emits an inspectable artifact chain."
        ),
        defaults={**output_defaults, "output_dir": artifact_output_dir},
        plan_steps=plan_steps,
        follow_up_workflows=follow_up_workflows,
        notes=(
            "Use the report workflow as the default first answer; drop to lower-level analysis or compression only when you already know why.",
            "Compare, inverse-fit, or surrogate-evaluate only after the report tells you the table is numerically and operationally healthy.",
        ),
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
        notes=(
            "Python is the primary surface; CLI, MCP, and API wrap the same shared workflows.",
            "File-backed and SQL-backed inputs converge through explicit table materialization before entering spectral workflows.",
            "Modal-surrogate workflows remain extensions over the spectral spine rather than a separate product identity.",
            "The default adoption path is report-first: validate structure, analyze modes, compress, write artifacts, then choose follow-up workflows from evidence.",
        ),
    )


__all__ = [
    "DEFAULT_MCP_LOG_LEVEL",
    "DEFAULT_MCP_MAX_CONCURRENT_TASKS",
    "DEFAULT_PROFILE_REPORT_ANALYZE_NUM_MODES",
    "DEFAULT_PROFILE_REPORT_CAPTURE_THRESHOLDS",
    "DEFAULT_PROFILE_REPORT_COMPRESS_NUM_MODES",
    "DEFAULT_PROFILE_REPORT_DEVICE",
    "DEFAULT_PROFILE_REPORT_NORMALIZE_EACH_PROFILE",
    "DEFAULT_PROFILE_REPORT_OUTPUT_DIR",
    "DEFAULT_PROFILE_REPORT_PYTHON_OUTPUT_DIR",
    "DEFAULT_SQL_PROFILE_REPORT_OUTPUT_DIR",
    "DEFAULT_SQL_PROFILE_SORT_BY_TIME",
    "DEFAULT_SQL_PROFILE_TIME_COLUMN",
    "PRODUCT_NAME",
    "PRODUCT_SPINE_STATEMENT",
    "RUNTIME_SPINE_STATEMENT",
    "KillerWorkflowDefinition",
    "ProductIdentityReport",
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
