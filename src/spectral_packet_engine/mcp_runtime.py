from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import platform


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


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    max_concurrent_tasks: int = 1
    slot_acquire_timeout_seconds: float = 60.0
    log_level: str = "WARNING"
    log_file: str | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.slot_acquire_timeout_seconds <= 0:
            raise ValueError("slot_acquire_timeout_seconds must be positive")
        object.__setattr__(self, "log_level", _normalize_log_level(self.log_level))
        object.__setattr__(self, "log_file", None if self.log_file is None else str(self.log_file))

    @property
    def log_destination(self) -> str:
        return "stderr" if self.log_file is None else self.log_file

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
        notes.append("Install the 'mcp' extra to enable the stdio MCP server runtime.")
    else:
        notes.append("MCP uses stdio transport and reserves stdout for protocol messages.")
        notes.append("Repository-managed logs are safe on stderr or a dedicated log file.")
        notes.append("The runtime uses bounded in-process execution slots; it does not provide forced cancellation.")

    if system == "Linux":
        recommended_supervision.extend(("systemd", "container restart policy"))
        notes.append("Linux is a first-class operational target for supervised MCP deployments.")
    elif system == "Darwin":
        recommended_supervision.append("launchd")
        notes.append("macOS is a first-class operational target for local or launchd-supervised MCP deployments.")
    elif system == "Windows":
        notes.append("Windows is supported for local stdio MCP use, but supervised long-running deployment is best-effort.")
    else:
        notes.append(f"{system} is not a primary operational target; use the Python library directly if possible.")

    notes.append("Automatic restarts belong to an external supervisor, not to the repository process itself.")

    return MCPRuntimeReport(
        available=available,
        mcp_version=mcp_version,
        transport="stdio",
        system=system,
        machine=machine,
        config=chosen_config,
        stderr_logging_safe=True,
        forced_cancellation_supported=False,
        recommended_supervision=tuple(recommended_supervision),
        notes=tuple(notes),
    )


__all__ = [
    "MCPRuntimeReport",
    "MCPServerConfig",
    "inspect_mcp_runtime",
]
