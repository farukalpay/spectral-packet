#!/bin/sh
set -eu

scratch_dir="${SPE_SCRATCH_DIR:-/data/scratch}"
log_file="${SPE_LOG_FILE:-/data/logs/mcp.log}"
log_dir="$(dirname "$log_file")"

mkdir -p "$scratch_dir" "$log_dir"

if [ "$(id -u)" = "0" ]; then
  touch "$log_file" 2>/dev/null || true
  chown appuser:appuser "$scratch_dir" "$log_dir" 2>/dev/null || true
  chown appuser:appuser "$log_file" 2>/dev/null || true
fi

if [ -n "${SPE_PUBLIC_HOSTS:-}" ]; then
  eval "$(
    python - <<'PY'
import os
import shlex

from spectral_packet_engine.mcp_deployment import build_mcp_public_ingress_plan

hosts = [item.strip() for item in os.getenv("SPE_PUBLIC_HOSTS", "").split(",") if item.strip()]
if not hosts:
    raise SystemExit(0)
public_port = os.getenv("SPE_PUBLIC_PORT", "").strip()
plan = build_mcp_public_ingress_plan(
    public_hosts=hosts,
    public_scheme=os.getenv("SPE_PUBLIC_SCHEME", "https"),
    public_port=None if not public_port else int(public_port),
    streamable_http_path=os.getenv("SPE_STREAMABLE_HTTP_PATH", "/mcp"),
    upstream_host="127.0.0.1",
    upstream_port=int(os.getenv("SPE_PORT", "8000")),
    root_redirect_path=os.getenv("SPE_ROOT_REDIRECT_PATH", "/mcp/server_info"),
)
if not os.getenv("SPE_ALLOWED_HOSTS", "").strip():
    print(f'SPE_ALLOWED_HOSTS={shlex.quote(",".join(plan.allowed_hosts))}')
if not os.getenv("SPE_ALLOWED_ORIGINS", "").strip():
    print(f'SPE_ALLOWED_ORIGINS={shlex.quote(",".join(plan.allowed_origins))}')
PY
  )"
fi

set -- \
  spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port "${SPE_PORT:-8000}" \
  --streamable-http-path "${SPE_STREAMABLE_HTTP_PATH:-/mcp}" \
  --max-concurrent-tasks "${SPE_MCP_MAX_CONCURRENT:-2}" \
  --slot-timeout-seconds "${SPE_MCP_SLOT_TIMEOUT_SECONDS:-60}" \
  --log-level "${SPE_MCP_LOG_LEVEL:-warning}" \
  --storage-protection-window-seconds "${SPE_STORAGE_PROTECTION_WINDOW_SECONDS:-86400}" \
  --storage-seed-bytes "${SPE_STORAGE_SEED_BYTES:-8388608}" \
  --storage-minimum-mutation-cost-bytes "${SPE_STORAGE_MINIMUM_MUTATION_COST_BYTES:-4096}" \
  --storage-snapshot-retention "${SPE_STORAGE_SNAPSHOT_RETENTION:-8}" \
  --scratch-dir "$scratch_dir" \
  --log-file "$log_file"

if [ "${SPE_ALLOW_UNSAFE_PYTHON:-false}" = "true" ]; then
  set -- "$@" --allow-unsafe-python
fi

if [ "${SPE_STORAGE_RESTORE_ON_STARTUP:-true}" != "true" ]; then
  set -- "$@" --disable-managed-db-restore
fi

old_ifs="${IFS}"
IFS=','
for host in ${SPE_ALLOWED_HOSTS:-}; do
  if [ -n "$host" ]; then
    set -- "$@" --allowed-host "$host"
  fi
done
for origin in ${SPE_ALLOWED_ORIGINS:-}; do
  if [ -n "$origin" ]; then
    set -- "$@" --allowed-origin "$origin"
  fi
done
IFS="${old_ifs}"

if [ "$(id -u)" = "0" ]; then
  exec python - "$@" <<'PY'
import os
import pwd
import sys

user = "appuser"
account = pwd.getpwnam(user)
os.initgroups(user, account.pw_gid)
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
os.execvp(sys.argv[1], sys.argv[1:])
PY
fi

exec "$@"
