from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import platform
from pathlib import Path

from spectral_packet_engine.config import PlatformConfig


_SUPPORTED_TRANSPORTS = {"stdio", "streamable-http"}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _normalize_log_level(level: str) -> str:
    normalized = str(level).strip().upper()
    if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("log_level must be one of DEBUG, INFO, WARNING, or ERROR")
    return normalized


def _normalize_transport(transport: str) -> str:
    normalized = str(transport).strip().lower()
    if normalized not in _SUPPORTED_TRANSPORTS:
        supported = ", ".join(sorted(_SUPPORTED_TRANSPORTS))
        raise ValueError(f"transport must be one of {supported}")
    return normalized


def _normalize_string_patterns(values: object, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = tuple(values)
    normalized: list[str] = []
    for value in raw_values:
        token = str(value).strip()
        if not token:
            continue
        normalized.append(token)
    return tuple(dict.fromkeys(normalized))


def default_mcp_scratch_dir() -> Path:
    return PlatformConfig.detect().cache_dir / "mcp-scratch"


def resolve_mcp_scratch_dir(config: "MCPServerConfig | None" = None) -> Path:
    if config is None or config.scratch_directory is None:
        return default_mcp_scratch_dir()
    return Path(config.scratch_directory).expanduser()


def ensure_mcp_scratch_dir(config: "MCPServerConfig | None" = None) -> Path:
    scratch_dir = resolve_mcp_scratch_dir(config)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir


def resolve_mcp_scratch_file(
    name: str,
    *,
    config: "MCPServerConfig | None" = None,
    suffix: str = "",
) -> Path:
    token = str(name).strip()
    if not token:
        raise ValueError("managed scratch names must not be empty")
    candidate = Path(token)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {".", ".."}:
        raise ValueError("managed scratch names must be plain filenames without path traversal")
    filename = candidate.name
    if suffix and not filename.endswith(suffix):
        filename = f"{filename}{suffix}"
    scratch_dir = ensure_mcp_scratch_dir(config)
    resolved = (scratch_dir / filename).resolve()
    try:
        resolved.relative_to(scratch_dir.resolve())
    except ValueError as exc:
        raise ValueError("managed scratch path escaped the configured scratch directory") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    streamable_http_path: str = "/mcp"
    max_concurrent_tasks: int = 1
    slot_acquire_timeout_seconds: float = 60.0
    log_level: str = "WARNING"
    log_file: str | None = None
    allow_unsafe_python: bool = False
    scratch_directory: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    max_generated_profiles: int = 512
    max_generated_grid_points: int = 4096
    max_execute_python_code_chars: int = 20000
    # ── Security limits (library-level, apply to all deployments) ──
    max_database_size_mb: float = 256.0
    max_scratch_databases: int = 20
    max_script_length_chars: int = 500_000
    max_script_statements: int = 5_000
    max_query_seconds: float = 30.0
    max_pivot_cardinality: int = 500
    max_interpolation_steps: int = 100_000
    max_unpivot_columns: int = 200
    max_result_rows_materialize: int = 500_000
    rate_limit_per_minute: int = 120

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport", _normalize_transport(self.transport))
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.slot_acquire_timeout_seconds <= 0:
            raise ValueError("slot_acquire_timeout_seconds must be positive")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_generated_profiles <= 0:
            raise ValueError("max_generated_profiles must be positive")
        if self.max_generated_grid_points <= 0:
            raise ValueError("max_generated_grid_points must be positive")
        if self.max_execute_python_code_chars <= 0:
            raise ValueError("max_execute_python_code_chars must be positive")
        if self.max_database_size_mb <= 0:
            raise ValueError("max_database_size_mb must be positive")
        if self.max_scratch_databases <= 0:
            raise ValueError("max_scratch_databases must be positive")
        if self.max_script_length_chars <= 0:
            raise ValueError("max_script_length_chars must be positive")
        if self.max_script_statements <= 0:
            raise ValueError("max_script_statements must be positive")
        if self.max_query_seconds <= 0:
            raise ValueError("max_query_seconds must be positive")
        if self.max_pivot_cardinality <= 0:
            raise ValueError("max_pivot_cardinality must be positive")
        if self.max_interpolation_steps <= 0:
            raise ValueError("max_interpolation_steps must be positive")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("rate_limit_per_minute must be positive")
        object.__setattr__(self, "log_level", _normalize_log_level(self.log_level))
        object.__setattr__(self, "log_file", None if self.log_file is None else str(self.log_file))
        object.__setattr__(self, "allow_unsafe_python", bool(self.allow_unsafe_python))
        normalized_http_path = "/" if self.streamable_http_path == "/" else "/" + str(self.streamable_http_path).strip("/")
        object.__setattr__(self, "streamable_http_path", normalized_http_path)
        object.__setattr__(
            self,
            "allowed_hosts",
            _normalize_string_patterns(self.allowed_hosts, field_name="allowed_hosts"),
        )
        object.__setattr__(
            self,
            "allowed_origins",
            _normalize_string_patterns(self.allowed_origins, field_name="allowed_origins"),
        )
        default_scratch = default_mcp_scratch_dir()
        resolved_scratch = default_scratch if self.scratch_directory is None else Path(self.scratch_directory).expanduser()
        object.__setattr__(self, "scratch_directory", str(resolved_scratch))

    @property
    def log_destination(self) -> str:
        return "stderr" if self.log_file is None else self.log_file

    @property
    def scratch_directory_path(self) -> Path:
        return resolve_mcp_scratch_dir(self)

    @property
    def endpoint_url(self) -> str | None:
        if self.transport != "streamable-http":
            return None
        return f"http://{self.host}:{self.port}{self.streamable_http_path}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MCPRuntimeReport:
    available: bool
    mcp_version: str | None
    transport: str
    system: str
    machine: str
    config: MCPServerConfig
    stderr_logging_safe: bool
    forced_cancellation_supported: bool
    package_entrypoint: tuple[str, ...]
    recommended_supervision: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "mcp_version": self.mcp_version,
            "transport": self.transport,
            "system": self.system,
            "machine": self.machine,
            "config": self.config.to_dict(),
            "stderr_logging_safe": self.stderr_logging_safe,
            "forced_cancellation_supported": self.forced_cancellation_supported,
            "package_entrypoint": list(self.package_entrypoint),
            "recommended_supervision": list(self.recommended_supervision),
            "notes": list(self.notes),
        }


def inspect_mcp_runtime(config: MCPServerConfig | None = None) -> MCPRuntimeReport:
    chosen_config = config or MCPServerConfig()
    available = importlib.util.find_spec("mcp.server.fastmcp") is not None
    mcp_version = _package_version("mcp")
    system = platform.system()
    machine = platform.machine().lower()

    notes: list[str] = []
    recommended_supervision: list[str] = []

    if not available:
        notes.append("Install the 'mcp' extra to enable the MCP server runtime.")
    else:
        if chosen_config.transport == "stdio":
            notes.append("MCP uses stdio transport and reserves stdout for protocol messages.")
        else:
            notes.append(
                f"MCP uses streamable HTTP transport at {chosen_config.endpoint_url} and can be supervised as a long-running network service."
            )
            if chosen_config.allowed_hosts:
                notes.append(
                    "Additional public Host header values are allowed for this MCP transport: "
                    + ", ".join(chosen_config.allowed_hosts)
                )
            if chosen_config.allowed_origins:
                notes.append(
                    "Additional public Origin values are allowed for this MCP transport: "
                    + ", ".join(chosen_config.allowed_origins)
                )
        notes.append("Repository-managed logs are safe on stderr or a dedicated log file.")
        notes.append("The runtime uses bounded in-process execution slots; it does not provide forced cancellation.")
        if chosen_config.allow_unsafe_python:
            notes.append("The execute_python tool is enabled. Treat this runtime as trusted-only because it can run arbitrary local Python.")
        else:
            notes.append("The execute_python tool is disabled by default; enable it explicitly only for trusted local sessions.")
        notes.append(f"Managed scratch files live under {chosen_config.scratch_directory_path}.")

    if system == "Linux":
        if chosen_config.transport == "streamable-http":
            recommended_supervision.extend(("systemd", "container restart policy"))
        notes.append("Linux is a first-class operational target for supervised MCP deployments.")
    elif system == "Darwin":
        if chosen_config.transport == "streamable-http":
            recommended_supervision.append("launchd")
        notes.append("macOS is a first-class operational target for local or launchd-supervised MCP deployments.")
    elif system == "Windows":
        notes.append("Windows is supported for local stdio MCP use, but supervised long-running deployment is best-effort.")
    else:
        notes.append(f"{system} is not a primary operational target; use the Python library directly if possible.")

    if chosen_config.transport == "stdio":
        notes.append("Automatic restarts for stdio MCP belong to the client launcher or an external wrapper, not to the repository process itself.")
        notes.append("Persistent always-on supervision is mainly meaningful for network transports; stdio servers are usually launched on demand by the client.")
    else:
        notes.append("Automatic restarts for streamable HTTP MCP belong to an external supervisor, not to the repository process itself.")

    return MCPRuntimeReport(
        available=available,
        mcp_version=mcp_version,
        transport=chosen_config.transport,
        system=system,
        machine=machine,
        config=chosen_config,
        stderr_logging_safe=True,
        forced_cancellation_supported=False,
        package_entrypoint=("python", "-m", "spectral_packet_engine", "serve-mcp"),
        recommended_supervision=tuple(recommended_supervision),
        notes=tuple(notes),
    )


__all__ = [
    "MCPRuntimeReport",
    "MCPServerConfig",
    "default_mcp_scratch_dir",
    "ensure_mcp_scratch_dir",
    "inspect_mcp_runtime",
    "resolve_mcp_scratch_dir",
    "resolve_mcp_scratch_file",
]
