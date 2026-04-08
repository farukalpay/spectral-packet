from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

from spectral_packet_engine.config import PlatformConfig


DEFAULT_MCP_CLIENT_SERVER_NAME = "spectral-packet-engine"
DEFAULT_MCP_SERVICE_LABEL = "dev.spectral-packet-engine.mcp"


def _normalize_env(env: Mapping[str, str] | None) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in ({} if env is None else env).items()
        if str(value)
    }


def _normalize_mount_path(path: str) -> str:
    return "/" if str(path) == "/" else "/" + str(path).strip("/")


def _normalize_public_scheme(scheme: str) -> str:
    normalized = str(scheme).strip().lower()
    if normalized not in {"http", "https"}:
        raise ValueError("public_scheme must be either 'http' or 'https'")
    return normalized


def _normalize_public_hostnames(hosts: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(hosts, str):
        raw_hosts = tuple(str(item).strip() for item in hosts.split(","))
    else:
        raw_hosts = tuple(str(item).strip() for item in hosts)
    normalized: list[str] = []
    for host in raw_hosts:
        if not host:
            continue
        if "://" in host or "/" in host or " " in host:
            raise ValueError(f"public hostnames must be plain host tokens, got {host!r}")
        normalized.append(host)
    deduplicated = tuple(dict.fromkeys(normalized))
    if not deduplicated:
        raise ValueError("at least one public hostname is required")
    return deduplicated


def _origin_for_host(host: str, *, scheme: str, port: int | None) -> str:
    default_port = 443 if scheme == "https" else 80
    if port is None or int(port) == default_port:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{int(port)}"


def derive_public_mcp_network_identity(
    *,
    public_hosts: Sequence[str] | str,
    public_scheme: str = "https",
    public_port: int | None = None,
    streamable_http_path: str = "/mcp",
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    resolved_hosts = _normalize_public_hostnames(public_hosts)
    resolved_scheme = _normalize_public_scheme(public_scheme)
    resolved_path = _normalize_mount_path(streamable_http_path)
    allowed_hosts: list[str] = []
    for host in resolved_hosts:
        allowed_hosts.append(host)
        if ":" not in host:
            allowed_hosts.append(f"{host}:*")
    allowed_origins = tuple(
        _origin_for_host(host, scheme=resolved_scheme, port=public_port)
        for host in resolved_hosts
    )
    public_endpoint = f"{allowed_origins[0]}{resolved_path}"
    return tuple(dict.fromkeys(allowed_hosts)), allowed_origins, public_endpoint


def detect_source_checkout(path: str | Path) -> bool:
    root = Path(path)
    return (root / "src" / "spectral_packet_engine" / "__init__.py").exists()


def default_mcp_server_command(
    *,
    python_executable: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    max_concurrent_tasks: int = 1,
    slot_timeout_seconds: float = 60.0,
    log_level: str = "warning",
    log_file: str | Path | None = None,
    allow_unsafe_python: bool = False,
    scratch_directory: str | Path | None = None,
    allowed_hosts: Sequence[str] = (),
    allowed_origins: Sequence[str] = (),
) -> tuple[str, ...]:
    command = [
        python_executable or sys.executable,
        "-m",
        "spectral_packet_engine",
        "serve-mcp",
        "--transport",
        str(transport),
        "--max-concurrent-tasks",
        str(max_concurrent_tasks),
        "--slot-timeout-seconds",
        str(slot_timeout_seconds),
        "--log-level",
        str(log_level).lower(),
    ]
    if str(transport) == "streamable-http":
        command.extend(
            (
                "--host",
                str(host),
                "--port",
                str(port),
                "--streamable-http-path",
                str(streamable_http_path),
            )
        )
    if log_file is not None:
        command.extend(("--log-file", str(log_file)))
    if allow_unsafe_python:
        command.append("--allow-unsafe-python")
    if scratch_directory is not None:
        command.extend(("--scratch-dir", str(scratch_directory)))
    for host_pattern in allowed_hosts:
        command.extend(("--allowed-host", str(host_pattern)))
    for origin_pattern in allowed_origins:
        command.extend(("--allowed-origin", str(origin_pattern)))
    return tuple(command)


def render_nginx_mcp_site(
    *,
    public_hosts: Sequence[str] | str,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8765,
    streamable_http_path: str = "/mcp",
    root_redirect_path: str | None = "/mcp/server_info",
    server_tokens_off: bool = True,
) -> str:
    resolved_hosts = _normalize_public_hostnames(public_hosts)
    resolved_path = _normalize_mount_path(streamable_http_path)
    resolved_root_redirect = None if root_redirect_path is None else _normalize_mount_path(root_redirect_path)
    server_tokens_line = "    server_tokens off;\n" if server_tokens_off else ""
    root_block = (
        f"    location = / {{\n        return 308 {resolved_root_redirect};\n    }}\n\n"
        if resolved_root_redirect is not None
        else ""
    )
    return (
        "server {\n"
        "    listen 80;\n"
        "    listen [::]:80;\n"
        f"    server_name {' '.join(resolved_hosts)};\n"
        f"{server_tokens_line}"
        f"{root_block}"
        f"    location = {resolved_path} {{\n"
        f"        proxy_pass http://{upstream_host}:{int(upstream_port)};\n"
        "        proxy_http_version 1.1;\n"
        "        proxy_set_header Connection \"\";\n"
        "        proxy_set_header Host $host;\n"
        "        proxy_set_header X-Real-IP $remote_addr;\n"
        "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        "        proxy_set_header X-Forwarded-Proto $scheme;\n"
        "        proxy_buffering off;\n"
        "        proxy_read_timeout 300;\n"
        "    }\n\n"
        f"    location {resolved_path}/ {{\n"
        f"        proxy_pass http://{upstream_host}:{int(upstream_port)};\n"
        "        proxy_http_version 1.1;\n"
        "        proxy_set_header Connection \"\";\n"
        "        proxy_set_header Host $host;\n"
        "        proxy_set_header X-Real-IP $remote_addr;\n"
        "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        "        proxy_set_header X-Forwarded-Proto $scheme;\n"
        "        proxy_buffering off;\n"
        "        proxy_read_timeout 300;\n"
        "    }\n"
        "}\n"
    )


def render_nginx_mcp_location_snippet(
    *,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8765,
    streamable_http_path: str = "/mcp",
) -> str:
    resolved_path = _normalize_mount_path(streamable_http_path)
    return (
        f"location = {resolved_path} {{\n"
        f"    proxy_pass http://{upstream_host}:{int(upstream_port)};\n"
        "    proxy_http_version 1.1;\n"
        "    proxy_set_header Connection \"\";\n"
        "    proxy_set_header Host $host;\n"
        "    proxy_set_header X-Real-IP $remote_addr;\n"
        "    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        "    proxy_set_header X-Forwarded-Proto $scheme;\n"
        "    proxy_buffering off;\n"
        "    proxy_read_timeout 300;\n"
        "}\n\n"
        f"location {resolved_path}/ {{\n"
        f"    proxy_pass http://{upstream_host}:{int(upstream_port)};\n"
        "    proxy_http_version 1.1;\n"
        "    proxy_set_header Connection \"\";\n"
        "    proxy_set_header Host $host;\n"
        "    proxy_set_header X-Real-IP $remote_addr;\n"
        "    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        "    proxy_set_header X-Forwarded-Proto $scheme;\n"
        "    proxy_buffering off;\n"
        "    proxy_read_timeout 300;\n"
        "}\n"
    )


@dataclass(frozen=True, slots=True)
class MCPClientConfiguration:
    server_name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": self.command,
            "args": list(self.args),
        }
        if self.env:
            payload["env"] = dict(self.env)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return {"mcpServers": {self.server_name: payload}}


@dataclass(frozen=True, slots=True)
class MCPPublicIngressPlan:
    public_hosts: tuple[str, ...]
    public_scheme: str
    public_port: int | None
    public_endpoint_url: str
    streamable_http_path: str
    upstream_host: str
    upstream_port: int
    root_redirect_path: str | None
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    docker_environment: dict[str, str]
    nginx_config: str
    nginx_location_snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_hosts": list(self.public_hosts),
            "public_scheme": self.public_scheme,
            "public_port": self.public_port,
            "public_endpoint_url": self.public_endpoint_url,
            "streamable_http_path": self.streamable_http_path,
            "upstream_host": self.upstream_host,
            "upstream_port": self.upstream_port,
            "root_redirect_path": self.root_redirect_path,
            "allowed_hosts": list(self.allowed_hosts),
            "allowed_origins": list(self.allowed_origins),
            "docker_environment": dict(self.docker_environment),
            "nginx_config": self.nginx_config,
            "nginx_location_snippet": self.nginx_location_snippet,
        }


def build_mcp_public_ingress_plan(
    *,
    public_hosts: Sequence[str] | str,
    public_scheme: str = "https",
    public_port: int | None = None,
    upstream_host: str = "127.0.0.1",
    upstream_port: int = 8765,
    streamable_http_path: str = "/mcp",
    root_redirect_path: str | None = "/mcp/server_info",
) -> MCPPublicIngressPlan:
    resolved_hosts = _normalize_public_hostnames(public_hosts)
    resolved_scheme = _normalize_public_scheme(public_scheme)
    resolved_path = _normalize_mount_path(streamable_http_path)
    allowed_hosts, allowed_origins, public_endpoint = derive_public_mcp_network_identity(
        public_hosts=resolved_hosts,
        public_scheme=resolved_scheme,
        public_port=public_port,
        streamable_http_path=resolved_path,
    )
    docker_environment = {
        "SPE_PUBLIC_HOSTS": ",".join(resolved_hosts),
        "SPE_PUBLIC_SCHEME": resolved_scheme,
        "SPE_STREAMABLE_HTTP_PATH": resolved_path,
        "SPE_ALLOWED_HOSTS": ",".join(allowed_hosts),
        "SPE_ALLOWED_ORIGINS": ",".join(allowed_origins),
    }
    if public_port is not None:
        docker_environment["SPE_PUBLIC_PORT"] = str(int(public_port))
    if root_redirect_path is not None:
        docker_environment["SPE_ROOT_REDIRECT_PATH"] = _normalize_mount_path(root_redirect_path)
    return MCPPublicIngressPlan(
        public_hosts=resolved_hosts,
        public_scheme=resolved_scheme,
        public_port=None if public_port is None else int(public_port),
        public_endpoint_url=public_endpoint,
        streamable_http_path=resolved_path,
        upstream_host=str(upstream_host),
        upstream_port=int(upstream_port),
        root_redirect_path=None if root_redirect_path is None else _normalize_mount_path(root_redirect_path),
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        docker_environment=docker_environment,
        nginx_config=render_nginx_mcp_site(
            public_hosts=resolved_hosts,
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            streamable_http_path=resolved_path,
            root_redirect_path=root_redirect_path,
        ),
        nginx_location_snippet=render_nginx_mcp_location_snippet(
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            streamable_http_path=resolved_path,
        ),
    )


def build_local_mcp_client_configuration(
    *,
    working_directory: str | Path,
    python_executable: str = "python3",
    server_name: str = DEFAULT_MCP_CLIENT_SERVER_NAME,
    max_concurrent_tasks: int = 1,
    slot_timeout_seconds: float = 60.0,
    log_level: str = "warning",
    allow_unsafe_python: bool = False,
    source_checkout: bool | None = None,
) -> MCPClientConfiguration:
    root = Path(working_directory)
    use_source_checkout = detect_source_checkout(root) if source_checkout is None else bool(source_checkout)
    env = {"PYTHONPATH": "src"} if use_source_checkout else {}
    command = default_mcp_server_command(
        python_executable=python_executable,
        max_concurrent_tasks=max_concurrent_tasks,
        slot_timeout_seconds=slot_timeout_seconds,
        log_level=log_level,
        allow_unsafe_python=allow_unsafe_python,
    )
    return MCPClientConfiguration(
        server_name=server_name,
        command=command[0],
        args=command[1:],
        env=env,
        metadata={
            "name": "Spectral Packet Engine",
            "description": (
                "Bounded-domain spectral computation: packet simulation, modal analysis, "
                "profile compression, inverse reconstruction, spectral feature export, "
                "load modeling, and SQL-backed workflows."
            ),
        },
    )


def build_ssh_mcp_client_configuration(
    *,
    host: str,
    remote_working_directory: str | Path,
    remote_python_executable: str = "python3",
    server_name: str = DEFAULT_MCP_CLIENT_SERVER_NAME,
    max_concurrent_tasks: int = 1,
    slot_timeout_seconds: float = 60.0,
    log_level: str = "warning",
    allow_unsafe_python: bool = False,
    source_checkout: bool = False,
) -> MCPClientConfiguration:
    remote_env = "PYTHONPATH=src " if source_checkout else ""
    remote_command = shlex.join(
        list(
            default_mcp_server_command(
                python_executable=remote_python_executable,
                max_concurrent_tasks=max_concurrent_tasks,
                slot_timeout_seconds=slot_timeout_seconds,
                log_level=log_level,
                allow_unsafe_python=allow_unsafe_python,
            )
        )
    )
    ssh_command = f"cd {shlex.quote(str(remote_working_directory))} && {remote_env}{remote_command}"
    return MCPClientConfiguration(
        server_name=server_name,
        command="ssh",
        args=(str(host), ssh_command),
        metadata={
            "name": "Spectral Packet Engine",
            "description": "Remote stdio bridge over SSH for the Spectral Packet Engine MCP server.",
        },
    )


@dataclass(frozen=True, slots=True)
class MCPServiceInstallPlan:
    platform: str
    label: str
    manifest_path: str
    command: tuple[str, ...]
    working_directory: str
    log_file: str
    endpoint_url: str
    scratch_directory: str | None
    environment: dict[str, str]
    restart_policy: str
    enable_commands: tuple[tuple[str, ...], ...]
    manifest: str
    source_checkout: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "label": self.label,
            "manifest_path": self.manifest_path,
            "command": list(self.command),
            "working_directory": self.working_directory,
            "log_file": self.log_file,
            "endpoint_url": self.endpoint_url,
            "scratch_directory": self.scratch_directory,
            "environment": dict(self.environment),
            "restart_policy": self.restart_policy,
            "enable_commands": [list(command) for command in self.enable_commands],
            "source_checkout": self.source_checkout,
        }


@dataclass(frozen=True, slots=True)
class MCPTunnelPlan:
    host: str
    local_port: int
    remote_port: int
    remote_host: str
    streamable_http_path: str
    tunnel_command: tuple[str, ...]
    local_endpoint_url: str
    remote_endpoint_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "local_port": self.local_port,
            "remote_port": self.remote_port,
            "remote_host": self.remote_host,
            "streamable_http_path": self.streamable_http_path,
            "tunnel_command": list(self.tunnel_command),
            "local_endpoint_url": self.local_endpoint_url,
            "remote_endpoint_url": self.remote_endpoint_url,
        }


def build_ssh_tunnel_command(
    *,
    host: str,
    local_port: int,
    remote_port: int,
    remote_host: str = "127.0.0.1",
) -> tuple[str, ...]:
    return (
        "ssh",
        "-N",
        "-L",
        f"{int(local_port)}:{remote_host}:{int(remote_port)}",
        str(host),
    )


def build_mcp_tunnel_plan(
    *,
    host: str,
    local_port: int = 8765,
    remote_port: int = 8765,
    remote_host: str = "127.0.0.1",
    streamable_http_path: str = "/mcp",
) -> MCPTunnelPlan:
    normalized_path = "/" if streamable_http_path == "/" else "/" + str(streamable_http_path).strip("/")
    tunnel_command = build_ssh_tunnel_command(
        host=host,
        local_port=local_port,
        remote_port=remote_port,
        remote_host=remote_host,
    )
    return MCPTunnelPlan(
        host=str(host),
        local_port=int(local_port),
        remote_port=int(remote_port),
        remote_host=str(remote_host),
        streamable_http_path=normalized_path,
        tunnel_command=tunnel_command,
        local_endpoint_url=f"http://127.0.0.1:{int(local_port)}{normalized_path}",
        remote_endpoint_url=f"http://{remote_host}:{int(remote_port)}{normalized_path}",
    )


def _quote_systemd_exec(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _render_systemd_unit(plan: MCPServiceInstallPlan) -> str:
    environment_lines = "".join(
        f'Environment="{key}={value}"\n'
        for key, value in sorted(plan.environment.items())
    )
    return (
        "[Unit]\n"
        "Description=Spectral Packet Engine MCP server\n"
        "After=default.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={plan.working_directory}\n"
        f"ExecStart={_quote_systemd_exec(plan.command)}\n"
        f"Restart={plan.restart_policy}\n"
        "RestartSec=3\n"
        f"{environment_lines}"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _plist_array(values: Sequence[str]) -> str:
    return "\n".join(f"      <string>{value}</string>" for value in values)


def _render_launchd_plist(plan: MCPServiceInstallPlan) -> str:
    environment_payload = ""
    if plan.environment:
        lines = ["    <key>EnvironmentVariables</key>", "    <dict>"]
        for key, value in sorted(plan.environment.items()):
            lines.extend(
                (
                    f"      <key>{key}</key>",
                    f"      <string>{value}</string>",
                )
            )
        lines.append("    </dict>")
        environment_payload = "\n" + "\n".join(lines)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "    <key>Label</key>\n"
        f"    <string>{plan.label}</string>\n"
        "    <key>ProgramArguments</key>\n"
        "    <array>\n"
        f"{_plist_array(plan.command)}\n"
        "    </array>\n"
        "    <key>RunAtLoad</key>\n"
        "    <true/>\n"
        "    <key>KeepAlive</key>\n"
        "    <true/>\n"
        "    <key>WorkingDirectory</key>\n"
        f"    <string>{plan.working_directory}</string>\n"
        "    <key>StandardErrorPath</key>\n"
        f"    <string>{plan.log_file}</string>\n"
        f"{environment_payload}\n"
        "</dict>\n"
        "</plist>\n"
    )


def build_mcp_service_install_plan(
    *,
    working_directory: str | Path,
    label: str = DEFAULT_MCP_SERVICE_LABEL,
    python_executable: str | None = None,
    transport: str = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8765,
    streamable_http_path: str = "/mcp",
    max_concurrent_tasks: int = 1,
    slot_timeout_seconds: float = 60.0,
    log_level: str = "warning",
    log_file: str | Path | None = None,
    allow_unsafe_python: bool = False,
    scratch_directory: str | Path | None = None,
    allowed_hosts: Sequence[str] = (),
    allowed_origins: Sequence[str] = (),
    source_checkout: bool | None = None,
    platform_name: str | None = None,
) -> MCPServiceInstallPlan:
    platform_config = PlatformConfig.detect()
    detected_platform = platform_config.system if platform_name is None else str(platform_name)
    if detected_platform not in {"linux", "macos"}:
        raise ValueError("Automatic MCP supervision manifests are supported on Linux and macOS only.")
    if str(transport) == "stdio":
        raise ValueError("Automatic supervision requires a network transport such as streamable-http.")

    root = Path(working_directory).resolve()
    use_source_checkout = detect_source_checkout(root) if source_checkout is None else bool(source_checkout)
    environment = {"PYTHONPATH": "src"} if use_source_checkout else {}
    resolved_log_file = Path(log_file) if log_file is not None else platform_config.cache_dir / "logs" / "mcp.log"
    resolved_scratch_directory = None if scratch_directory is None else Path(scratch_directory).expanduser()
    command = default_mcp_server_command(
        python_executable=python_executable,
        transport=transport,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        max_concurrent_tasks=max_concurrent_tasks,
        slot_timeout_seconds=slot_timeout_seconds,
        log_level=log_level,
        log_file=resolved_log_file,
        allow_unsafe_python=allow_unsafe_python,
        scratch_directory=resolved_scratch_directory,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    endpoint_url = f"http://{host}:{port}{streamable_http_path}"

    if detected_platform == "linux":
        manifest_path = Path.home() / ".config" / "systemd" / "user" / f"{label}.service"
        enable_commands = (
            ("systemctl", "--user", "daemon-reload"),
            ("systemctl", "--user", "enable", "--now", f"{label}.service"),
        )
        manifest = _render_systemd_unit(
            MCPServiceInstallPlan(
                platform=detected_platform,
                label=label,
                manifest_path=str(manifest_path),
                command=command,
                working_directory=str(root),
                log_file=str(resolved_log_file),
                endpoint_url=endpoint_url,
                scratch_directory=None if resolved_scratch_directory is None else str(resolved_scratch_directory),
                environment=environment,
                restart_policy="always",
                enable_commands=enable_commands,
                manifest="",
                source_checkout=use_source_checkout,
            )
        )
    else:
        manifest_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        uid = str(os.getuid())
        enable_commands = (
            ("launchctl", "bootout", f"gui/{uid}", str(manifest_path)),
            ("launchctl", "bootstrap", f"gui/{uid}", str(manifest_path)),
        )
        manifest = _render_launchd_plist(
            MCPServiceInstallPlan(
                platform=detected_platform,
                label=label,
                manifest_path=str(manifest_path),
                command=command,
                working_directory=str(root),
                log_file=str(resolved_log_file),
                endpoint_url=endpoint_url,
                scratch_directory=None if resolved_scratch_directory is None else str(resolved_scratch_directory),
                environment=environment,
                restart_policy="always",
                enable_commands=enable_commands,
                manifest="",
                source_checkout=use_source_checkout,
            )
        )

    return MCPServiceInstallPlan(
        platform=detected_platform,
        label=label,
        manifest_path=str(manifest_path),
        command=command,
        working_directory=str(root),
        log_file=str(resolved_log_file),
        endpoint_url=endpoint_url,
        scratch_directory=None if resolved_scratch_directory is None else str(resolved_scratch_directory),
        environment=environment,
        restart_policy="always",
        enable_commands=enable_commands,
        manifest=manifest,
        source_checkout=use_source_checkout,
    )


def install_mcp_service(plan: MCPServiceInstallPlan, *, enable: bool = False) -> dict[str, Any]:
    manifest_path = Path(plan.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(plan.manifest, encoding="utf-8")
    executed: list[dict[str, Any]] = []
    if enable:
        for index, command in enumerate(plan.enable_commands):
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                check=False,
            )
            is_bootout_miss = (
                plan.platform == "macos"
                and index == 0
                and completed.returncode != 0
                and ("No such process" in completed.stderr or "Could not find service" in completed.stderr)
            )
            executed.append(
                {
                    "command": list(command),
                    "returncode": 0 if is_bootout_miss else completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
            if completed.returncode != 0 and not is_bootout_miss:
                raise RuntimeError(
                    f"failed to enable MCP service with command {' '.join(command)}: {completed.stderr.strip() or completed.stdout.strip()}"
                )
    return {
        "manifest_path": str(manifest_path),
        "enabled": enable,
        "enable_results": executed,
        "plan": plan.to_dict(),
    }


__all__ = [
    "DEFAULT_MCP_CLIENT_SERVER_NAME",
    "DEFAULT_MCP_SERVICE_LABEL",
    "MCPClientConfiguration",
    "MCPPublicIngressPlan",
    "MCPServiceInstallPlan",
    "MCPTunnelPlan",
    "build_local_mcp_client_configuration",
    "build_mcp_public_ingress_plan",
    "build_mcp_service_install_plan",
    "build_mcp_tunnel_plan",
    "build_ssh_tunnel_command",
    "build_ssh_mcp_client_configuration",
    "derive_public_mcp_network_identity",
    "default_mcp_server_command",
    "detect_source_checkout",
    "install_mcp_service",
    "render_nginx_mcp_location_snippet",
    "render_nginx_mcp_site",
]
