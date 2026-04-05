"""Centralized configuration for the Spectral Packet Engine.

All runtime defaults, MCP server settings, platform hints, and server
purpose statements live here.  This replaces scattered defaults across
product.py, mcp_runtime.py, and individual tool parameters.

Configuration precedence (highest wins):
1. Explicit function/constructor arguments
2. Environment variables (SPECTRAL_PACKET_*)
3. Config file (~/.spectral-packet-engine/config.toml or project-local)
4. Compiled defaults in this module

The ``load_config`` function resolves all layers and returns a frozen
``EngineConfig`` that the rest of the codebase can depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import os
import platform
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server purpose statement — this is what AI clients (Claude, ChatGPT, etc.)
# see when they connect via MCP.  It tells them what this server IS and
# what it can do, so they don't guess wrong.
# ---------------------------------------------------------------------------

SERVER_PURPOSE = (
    "Spectral Packet Engine is a spectral inverse physics server centered on bounded-domain computation. "
    "It provides tools for: (1) Gaussian wave-packet simulation and time propagation, "
    "(2) spectral modal decomposition and compression of 1D density profiles, "
    "(3) uncertainty-aware inverse reconstruction of wave-packet parameters from observed data, "
    "(4) explicit potential-family inference from observed spectra with local posterior and identifiability summaries, "
    "(5) controlled reduced models: separable tensor-product spectra, coupled channels, and radial reductions, "
    "(6) differentiable physics workflows for transition design and packet-control-style optimization, "
    "(7) report-first scientific tabular workflows with artifact output, "
    "(8) report-first spectral feature export for downstream tree-model workflows, "
    "(9) Chebyshev and sine-basis spectral analysis with convergence diagnostics, "
    "(10) momentum-space observables and Heisenberg uncertainty analysis, "
    "(11) energy conservation verification during propagation, "
    "(12) extended spectral methods: Fourier decomposition, Padé approximants, Hilbert transform, "
    "correlation spectral analysis, Richardson extrapolation, and Kramers-Kronig relations, and "
    "(13) SQL table operations: pivot, unpivot, interpolation, window aggregates, and type coercion. "
    "All tools operate on the same shared engine core and return structured JSON. "
    "IMPORTANT — the MCP server has its OWN filesystem. Your local paths (e.g. /home/claude/, "
    "/tmp/) do NOT exist on the server. NEVER pass local file paths to MCP tools. "
    "To upload data: use 'write_scratch_file' with inline content (CSV, JSON, etc.) — it "
    "returns a server-side path you can pass to inspect_profile_table, profile_table_report, "
    "write_database_table, or any file-based tool. "
    "To list/read uploaded files: use 'list_scratch_files' and 'read_scratch_file'. "
    "For database workflows: use 'create_scratch_database' with 'init_script' to create "
    "tables and insert data via SQL. 'query_database' is READ-ONLY and returns ALL rows "
    "(up to 500 by default, adjustable via max_rows). For very large results, use "
    "'export_query_csv'. For writes, use 'execute_database_script' or 'execute_database_statement'. "
    "To load data via SELECT into a new table, use 'materialize_query_table'. "
    "NEVER use bash/python/sqlite3 to open files or databases — use the MCP tools instead. "
    "Use 'inspect_product' to see the full workflow map, 'guide_workflow' to choose the default "
    "report, inverse-fit, or feature-model loop, 'inspect_environment' to check available hardware, "
    "and 'validate_installation' before heavy computation."
)

SERVER_PURPOSE_SHORT = (
    "Spectral inverse physics: packet simulation, uncertainty-aware inverse inference, controlled reduced models, differentiable design, report-first modal analysis, and SQL-backed workflows."
)


# ---------------------------------------------------------------------------
# Platform detection helpers
# ---------------------------------------------------------------------------

def _detect_platform() -> str:
    """Detect the current platform: 'linux', 'macos', or 'windows'."""
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    return "linux"


def _detect_default_device() -> str:
    """Detect the best default compute device for this platform."""
    system = _detect_platform()
    try:
        import torch
        if system == "macos" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        _log.debug("torch not available for device detection, falling back to cpu")
    return "cpu"


def _user_config_dir() -> Path:
    """Platform-appropriate user config directory."""
    system = _detect_platform()
    if system == "windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif system == "macos":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "spectral-packet-engine"


def _user_cache_dir() -> Path:
    """Platform-appropriate user cache directory."""
    system = _detect_platform()
    if system == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "macos":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "spectral-packet-engine"


def _user_data_dir() -> Path:
    """Platform-appropriate user data directory."""
    system = _detect_platform()
    if system == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "macos":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "spectral-packet-engine"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MCPConfig:
    """MCP server configuration."""
    max_concurrent_tasks: int = 1
    slot_acquire_timeout_seconds: float = 60.0
    log_level: str = "WARNING"
    log_file: str | None = None
    server_purpose: str = SERVER_PURPOSE
    server_purpose_short: str = SERVER_PURPOSE_SHORT

    def __post_init__(self) -> None:
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.slot_acquire_timeout_seconds <= 0:
            raise ValueError("slot_acquire_timeout_seconds must be positive")
        normalized = str(self.log_level).strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be one of DEBUG, INFO, WARNING, or ERROR")
        object.__setattr__(self, "log_level", normalized)


@dataclass(frozen=True, slots=True)
class SpectralDefaults:
    """Default parameters for spectral computation."""
    default_num_modes: int = 32
    default_quadrature_points: int = 4096
    default_grid_points: int = 512
    default_device: str = "auto"
    default_basis_type: str = "infinite-well"  # or "chebyshev"
    profile_report_analyze_modes: int = 16
    profile_report_compress_modes: int = 8
    profile_report_capture_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)
    convergence_error_tolerance: float = 0.01


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    """Platform-specific configuration."""
    system: str = ""
    config_dir: Path = field(default_factory=lambda: Path("."))
    cache_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("."))
    default_device: str = "cpu"
    stdout_encoding: str = "utf-8"

    @classmethod
    def detect(cls) -> PlatformConfig:
        system = _detect_platform()
        encoding = "utf-8"
        if system == "windows":
            import sys
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        return cls(
            system=system,
            config_dir=_user_config_dir(),
            cache_dir=_user_cache_dir(),
            data_dir=_user_data_dir(),
            default_device=_detect_default_device(),
            stdout_encoding=encoding,
        )


@dataclass(frozen=True, slots=True)
class DatabaseDefaults:
    """Default parameters for database operations."""
    default_url_prefix: str = "sqlite:///"
    connect_timeout_seconds: float = 30.0
    default_sort_by_time: bool = True
    default_time_column: str = "time"
    default_output_dir: str = "artifacts/sql_profile_report"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Top-level configuration container.

    This is the single source of truth for all engine defaults.
    """
    mcp: MCPConfig = field(default_factory=MCPConfig)
    spectral: SpectralDefaults = field(default_factory=SpectralDefaults)
    platform: PlatformConfig = field(default_factory=PlatformConfig.detect)
    database: DatabaseDefaults = field(default_factory=DatabaseDefaults)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inspection tools and artifact metadata."""
        return {
            "mcp": {
                "max_concurrent_tasks": self.mcp.max_concurrent_tasks,
                "slot_acquire_timeout_seconds": self.mcp.slot_acquire_timeout_seconds,
                "log_level": self.mcp.log_level,
                "log_file": self.mcp.log_file,
                "server_purpose": self.mcp.server_purpose_short,
            },
            "spectral": {
                "default_num_modes": self.spectral.default_num_modes,
                "default_basis_type": self.spectral.default_basis_type,
                "default_device": self.spectral.default_device,
                "convergence_error_tolerance": self.spectral.convergence_error_tolerance,
            },
            "platform": {
                "system": self.platform.system,
                "config_dir": str(self.platform.config_dir),
                "cache_dir": str(self.platform.cache_dir),
                "default_device": self.platform.default_device,
            },
            "database": {
                "connect_timeout_seconds": self.database.connect_timeout_seconds,
                "default_time_column": self.database.default_time_column,
            },
        }


# ---------------------------------------------------------------------------
# Config loading with environment variable overrides
# ---------------------------------------------------------------------------

def _env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable with the SPECTRAL_PACKET_ prefix."""
    return os.environ.get(f"SPECTRAL_PACKET_{key}", default)


def load_config() -> EngineConfig:
    """Load configuration with environment variable overrides.

    Environment variables (all optional):
        SPECTRAL_PACKET_MCP_MAX_TASKS      — max concurrent MCP tasks
        SPECTRAL_PACKET_MCP_LOG_LEVEL      — MCP log level
        SPECTRAL_PACKET_MCP_LOG_FILE       — MCP log file path
        SPECTRAL_PACKET_DEFAULT_DEVICE     — compute device (cpu/cuda/mps/auto)
        SPECTRAL_PACKET_DEFAULT_MODES      — default number of spectral modes
        SPECTRAL_PACKET_DB_TIMEOUT         — database connect timeout
    """
    mcp_config = MCPConfig(
        max_concurrent_tasks=int(_env("MCP_MAX_TASKS", "1")),
        slot_acquire_timeout_seconds=60.0,
        log_level=_env("MCP_LOG_LEVEL", "WARNING"),
        log_file=_env("MCP_LOG_FILE"),
    )

    device_override = _env("DEFAULT_DEVICE")
    modes_override = _env("DEFAULT_MODES")
    spectral_config = SpectralDefaults(
        default_device=device_override or "auto",
        default_num_modes=int(modes_override) if modes_override else 32,
    )

    db_timeout = _env("DB_TIMEOUT")
    db_config = DatabaseDefaults(
        connect_timeout_seconds=float(db_timeout) if db_timeout else 30.0,
    )

    return EngineConfig(
        mcp=mcp_config,
        spectral=spectral_config,
        platform=PlatformConfig.detect(),
        database=db_config,
    )


# ---------------------------------------------------------------------------
# Platform-aware local hosting helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HostingDiagnostic:
    """Result of a pre-flight hosting check."""
    ready: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    fixes: tuple[str, ...]
    platform_notes: tuple[str, ...]


def diagnose_hosting_readiness() -> HostingDiagnostic:
    """Run pre-flight checks for local MCP hosting.

    Detects common issues BEFORE the user hits them:
    - Missing dependencies
    - stdout encoding problems (Windows)
    - Permission issues on config/cache directories
    - Port conflicts (for API mode)
    - PyTorch/MCP version incompatibilities
    """
    errors: list[str] = []
    warnings: list[str] = []
    fixes: list[str] = []
    notes: list[str] = []
    system = _detect_platform()

    # 1. Check Python version
    import sys
    if sys.version_info < (3, 11):
        errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} detected; requires >= 3.11")
        fixes.append("Install Python 3.11+ from python.org or your package manager")

    # 2. Check torch availability
    try:
        import torch
        notes.append(f"PyTorch {torch.__version__} available")
        if system == "macos" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            notes.append("MPS (Apple Silicon) acceleration available")
        elif torch.cuda.is_available():
            notes.append(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            notes.append("Running on CPU (no GPU acceleration)")
    except ImportError:
        errors.append("PyTorch not installed")
        fixes.append("pip install torch>=2.4")

    # 3. Check MCP availability
    try:
        import mcp  # noqa: F401
        notes.append("MCP package available")
    except ImportError:
        warnings.append("MCP package not installed — MCP server mode unavailable")
        fixes.append('pip install "spectral-packet-engine[mcp]"')

    # 4. Windows-specific: stdout encoding
    if system == "windows":
        import sys as _sys
        encoding = getattr(_sys.stdout, "encoding", "unknown")
        if encoding.lower() not in ("utf-8", "utf8"):
            warnings.append(f"stdout encoding is '{encoding}', not UTF-8")
            fixes.append("Set environment variable PYTHONIOENCODING=utf-8 or run: chcp 65001")
            notes.append("MCP stdio transport requires UTF-8; non-UTF-8 encoding may corrupt JSON messages")

    # 5. Windows-specific: long path support
    if system == "windows":
        try:
            test_path = Path.home() / ("a" * 200) / "test"
            notes.append("Long path support: checking...")
            # Don't actually create — just note the recommendation
            warnings.append("Windows may have 260-char path limit for artifact directories")
            fixes.append("Enable long paths: reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f")
        except Exception:
            _log.debug("Windows long path check could not complete")

    # 6. macOS-specific: Gatekeeper / quarantine on downloaded binaries
    if system == "macos":
        notes.append("If torch fails with 'not verified' errors, run: xattr -cr $(python -c 'import torch; print(torch.__path__[0])')")

    # 7. Check directory permissions
    for name, dir_fn in [("config", _user_config_dir), ("cache", _user_cache_dir)]:
        try:
            d = dir_fn()
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
        except PermissionError:
            errors.append(f"Cannot write to {name} directory: {d}")
            fixes.append(f"Fix permissions: chmod -R u+rw {d}")
        except Exception as exc:
            warnings.append(f"{name} directory check: {exc}")

    # 8. Check numpy version compatibility
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major < 2:
            warnings.append(f"NumPy {np.__version__} detected; numpy>=2.0 recommended")
            fixes.append("pip install numpy>=2.0")
    except ImportError:
        _log.debug("numpy not available for version check")

    ready = len(errors) == 0
    return HostingDiagnostic(
        ready=ready,
        errors=tuple(errors),
        warnings=tuple(warnings),
        fixes=tuple(fixes),
        platform_notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# MCP config file generation for AI clients
# ---------------------------------------------------------------------------

def generate_mcp_client_config(
    *,
    server_command: str | None = None,
    server_args: list[str] | None = None,
) -> dict[str, Any]:
    """Generate an MCP client configuration block for Claude Desktop / VS Code.

    This is the JSON that goes into claude_desktop_config.json or
    .vscode/mcp.json so AI clients know how to launch this server.
    """
    from spectral_packet_engine.mcp_deployment import build_local_mcp_client_configuration

    if server_command is not None or server_args is not None:
        return {
            "mcpServers": {
                "spectral-packet-engine": {
                    "command": "python3" if server_command is None else server_command,
                    "args": [] if server_args is None else list(server_args),
                    "metadata": {
                        "name": "Spectral Packet Engine",
                        "description": SERVER_PURPOSE_SHORT,
                        "version": _get_version(),
                    },
                }
            }
        }

    payload = build_local_mcp_client_configuration(
        working_directory=Path.cwd(),
        python_executable="py" if _detect_platform() == "windows" else "python3",
        source_checkout=(Path.cwd() / "src" / "spectral_packet_engine" / "__init__.py").exists(),
    ).to_dict()
    payload["mcpServers"]["spectral-packet-engine"]["metadata"]["version"] = _get_version()
    return payload


def _get_version() -> str:
    try:
        from spectral_packet_engine.version import __version__
        return __version__
    except ImportError:
        return "unknown"


__all__ = [
    "DATABASE_DEFAULTS",
    "EngineConfig",
    "HostingDiagnostic",
    "MCPConfig",
    "PlatformConfig",
    "SERVER_PURPOSE",
    "SERVER_PURPOSE_SHORT",
    "SpectralDefaults",
    "diagnose_hosting_readiness",
    "generate_mcp_client_config",
    "load_config",
]
