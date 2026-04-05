from __future__ import annotations

"""Optional machine-facing interfaces over the shared engine workflows."""

from spectral_packet_engine.api import api_is_available, create_api_app
from spectral_packet_engine.mcp import create_mcp_server, mcp_is_available
from spectral_packet_engine.mcp_runtime import MCPRuntimeReport, MCPServerConfig, inspect_mcp_runtime
from spectral_packet_engine.service_runtime import APIStackRuntime, inspect_api_stack
from spectral_packet_engine.service_status import (
    ServiceStatusReport,
    ServiceTaskCounters,
    ServiceTaskStatusRecord,
    inspect_service_status,
)

__all__ = [
    "APIStackRuntime",
    "MCPRuntimeReport",
    "MCPServerConfig",
    "ServiceStatusReport",
    "ServiceTaskCounters",
    "ServiceTaskStatusRecord",
    "api_is_available",
    "create_api_app",
    "create_mcp_server",
    "inspect_api_stack",
    "inspect_mcp_runtime",
    "inspect_service_status",
    "mcp_is_available",
]
