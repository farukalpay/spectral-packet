#!/bin/sh
set -eu

repo_root="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
export COPYFILE_DISABLE=1

if [ ! -f .env ]; then
  echo ".env not found. Copy .env.example and fill in deployment variables first." >&2
  exit 1
fi

set -a
. ./.env
set +a

: "${SPE_SERVER_HOST:?SPE_SERVER_HOST must be set in .env}"
: "${SPE_SERVER_USER:?SPE_SERVER_USER must be set in .env}"
: "${SPE_DEPLOY_ROOT:?SPE_DEPLOY_ROOT must be set in .env}"

remote="${SPE_SERVER_USER}@${SPE_SERVER_HOST}"
deploy_root="${SPE_DEPLOY_ROOT}"
published_host="${SPE_PUBLISHED_HOST:-127.0.0.1}"
published_port="${SPE_PUBLISHED_PORT:-8765}"
streamable_http_path="${SPE_STREAMABLE_HTTP_PATH:-/mcp}"
compose_project_name="${SPE_COMPOSE_PROJECT_NAME:-spectral-packet-engine}"
manage_legacy_systemd="${SPE_MANAGE_LEGACY_SYSTEMD:-auto}"
legacy_systemd_units="${SPE_LEGACY_SYSTEMD_UNITS:-spectral-packet-engine.service lightcap-mcp.service}"
public_hosts="${SPE_PUBLIC_HOSTS:-}"
public_scheme="${SPE_PUBLIC_SCHEME:-https}"
public_port="${SPE_PUBLIC_PORT:-}"
root_redirect_path="${SPE_ROOT_REDIRECT_PATH:-/mcp/server_info}"
install_nginx_site="${SPE_INSTALL_NGINX_SITE:-false}"
nginx_site_name="${SPE_NGINX_SITE_NAME:-spectral-packet-engine-mcp.conf}"
nginx_site_mode="${SPE_NGINX_SITE_MODE:-detect}"

ssh -p "${SPE_SERVER_PORT:-22}" "$remote" "mkdir -p '$deploy_root'"

tar cz \
  --exclude .git \
  --exclude .venv \
  --exclude .pytest_cache \
  --exclude .mypy_cache \
  --exclude .ruff_cache \
  --exclude artifacts \
  --exclude build \
  --exclude dist \
  --exclude logs \
  --exclude docker-data \
  --exclude .env \
  . | ssh -p "${SPE_SERVER_PORT:-22}" "$remote" "tar xz -C '$deploy_root'"

ssh -p "${SPE_SERVER_PORT:-22}" "$remote" "
  set -eu
  cd '$deploy_root'
  compose() {
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
      docker compose \"\$@\"
    elif command -v docker-compose >/dev/null 2>&1; then
      docker-compose \"\$@\"
    else
      echo 'Docker Compose is not installed on the remote host.' >&2
      exit 1
    fi
  }
  port_in_use() {
    if command -v ss >/dev/null 2>&1; then
      ss -ltnH \"sport = :\${SPE_PUBLISHED_PORT}\" 2>/dev/null | grep -q .
    else
      netstat -ltn 2>/dev/null | awk '{print \$4}' | grep -Eq '(^|:)'\"\${SPE_PUBLISHED_PORT}\"'$'
    fi
  }
  print_port_owner() {
    if command -v ss >/dev/null 2>&1; then
      ss -ltnp | grep \":\${SPE_PUBLISHED_PORT} \" || true
    fi
    docker ps --filter \"publish=\${SPE_PUBLISHED_PORT}\" --format 'container={{.ID}} image={{.Image}} ports={{.Ports}} label={{.Label \"com.lightcap.service\"}}' || true
  }
  remove_managed_container_on_port() {
    existing_containers=\$(docker ps --filter \"publish=\${SPE_PUBLISHED_PORT}\" --format '{{.ID}}|{{.Image}}|{{.Label \"com.lightcap.service\"}}')
    if [ -z \"\$existing_containers\" ]; then
      return 0
    fi
    while IFS='|' read -r container_id container_image container_label; do
      [ -n \"\$container_id\" ] || continue
      if [ \"\$container_image\" = 'spectral-packet-engine:mcp' ] || [ \"\$container_label\" = 'spectral-packet-engine-mcp' ]; then
        echo \"Removing existing managed container \$container_id from port \${SPE_PUBLISHED_PORT}\"
        docker rm -f \"\$container_id\" >/dev/null
      else
        echo \"Port \${SPE_PUBLISHED_PORT} is already published by unrelated container \$container_id (\$container_image).\" >&2
        return 1
      fi
    done <<EOF
\$existing_containers
EOF
  }
  stop_matching_legacy_units() {
    if [ \"\${SPE_MANAGE_LEGACY_SYSTEMD}\" = 'off' ] || ! command -v systemctl >/dev/null 2>&1; then
      return 0
    fi
    for unit in \${SPE_LEGACY_SYSTEMD_UNITS}; do
      if ! systemctl is-active --quiet \"\$unit\"; then
        continue
      fi
      if systemctl cat \"\$unit\" 2>/dev/null | grep -F -- \"--port \${SPE_PUBLISHED_PORT}\" >/dev/null 2>&1; then
        echo \"Stopping legacy systemd unit \$unit on port \${SPE_PUBLISHED_PORT}\"
        systemctl stop \"\$unit\"
        systemctl disable \"\$unit\" >/dev/null 2>&1 || true
      fi
    done
  }
  install_nginx_site() {
    if [ \"\${SPE_INSTALL_NGINX_SITE}\" != 'true' ]; then
      return 0
    fi
    if [ -z \"\${SPE_PUBLIC_HOSTS}\" ]; then
      echo 'SPE_INSTALL_NGINX_SITE=true requires SPE_PUBLIC_HOSTS.' >&2
      exit 1
    fi
    if ! command -v nginx >/dev/null 2>&1; then
      echo 'nginx is not installed on the remote host.' >&2
      exit 1
    fi
    mkdir -p .deploy
    PYTHONPATH=src python3 - <<'PY' > .deploy/nginx-mcp-site.conf
import os

from spectral_packet_engine.mcp_deployment import build_mcp_public_ingress_plan

public_port = os.getenv("SPE_PUBLIC_PORT", "").strip()
plan = build_mcp_public_ingress_plan(
    public_hosts=os.environ["SPE_PUBLIC_HOSTS"],
    public_scheme=os.getenv("SPE_PUBLIC_SCHEME", "https"),
    public_port=None if not public_port else int(public_port),
    streamable_http_path=os.getenv("SPE_STREAMABLE_HTTP_PATH", "/mcp"),
    upstream_host="127.0.0.1",
    upstream_port=int(os.environ["SPE_PUBLISHED_PORT"]),
    root_redirect_path=os.getenv("SPE_ROOT_REDIRECT_PATH", "/mcp/server_info"),
)
print(plan.nginx_config, end="")
PY
    PYTHONPATH=src python3 - <<'PY' > .deploy/nginx-mcp-location.conf
import os

from spectral_packet_engine.mcp_deployment import build_mcp_public_ingress_plan

public_port = os.getenv("SPE_PUBLIC_PORT", "").strip()
plan = build_mcp_public_ingress_plan(
    public_hosts=os.environ["SPE_PUBLIC_HOSTS"],
    public_scheme=os.getenv("SPE_PUBLIC_SCHEME", "https"),
    public_port=None if not public_port else int(public_port),
    streamable_http_path=os.getenv("SPE_STREAMABLE_HTTP_PATH", "/mcp"),
    upstream_host="127.0.0.1",
    upstream_port=int(os.environ["SPE_PUBLISHED_PORT"]),
    root_redirect_path=os.getenv("SPE_ROOT_REDIRECT_PATH", "/mcp/server_info"),
)
print(plan.nginx_location_snippet, end="")
PY
    conflict_sites=""
    old_ifs="\${IFS}"
    IFS=','
    for host in \${SPE_PUBLIC_HOSTS}; do
      [ -n \"\$host\" ] || continue
      matches=\$(grep -R -l -E \"server_name[^;]*([[:space:]]|^)\${host}([[:space:]]|;)\" /etc/nginx/sites-enabled 2>/dev/null || true)
      if [ -n \"\$matches\" ]; then
        conflict_sites=\$(printf '%s\n%s' \"\$conflict_sites\" \"\$matches\")
      fi
    done
    IFS=\"\${old_ifs}\"
    conflict_sites=\$(printf '%s\n' \"\$conflict_sites\" | sed '/^$/d' | sort -u)
    target_enabled_path=\"/etc/nginx/sites-enabled/\${SPE_NGINX_SITE_NAME}\"
    filtered_conflicts=\$(printf '%s\n' \"\$conflict_sites\" | sed \"\|^\${target_enabled_path}\$|d\")
    if [ \"\${SPE_NGINX_SITE_MODE}\" = 'detect' ] && [ -n \"\$filtered_conflicts\" ]; then
      snippet_path=\"/etc/nginx/snippets/\${SPE_NGINX_SITE_NAME%.conf}-location.conf\"
      mkdir -p /etc/nginx/snippets
      install -m 0644 .deploy/nginx-mcp-location.conf \"\$snippet_path\"
      echo \"Detected existing nginx site ownership for one or more MCP public hosts:\" >&2
      printf '%s\n' \"\$filtered_conflicts\" >&2
      echo \"Wrote the MCP location snippet to \$snippet_path. Apply that snippet inside the owning server block instead of installing a competing dedicated site.\" >&2
      exit 1
    fi
    install -m 0644 .deploy/nginx-mcp-site.conf \"/etc/nginx/sites-available/\${SPE_NGINX_SITE_NAME}\"
    ln -sfn \"/etc/nginx/sites-available/\${SPE_NGINX_SITE_NAME}\" \"/etc/nginx/sites-enabled/\${SPE_NGINX_SITE_NAME}\"
    nginx -t
    if command -v systemctl >/dev/null 2>&1; then
      systemctl reload nginx
    else
      nginx -s reload
    fi
  }
  export COMPOSE_PROJECT_NAME='${compose_project_name}'
  export SPE_PUBLISHED_HOST='${published_host}'
  export SPE_PUBLISHED_PORT='${published_port}'
  export SPE_STREAMABLE_HTTP_PATH='${streamable_http_path}'
  export SPE_MCP_MAX_CONCURRENT='${SPE_MCP_MAX_CONCURRENT:-2}'
  export SPE_MCP_SLOT_TIMEOUT_SECONDS='${SPE_MCP_SLOT_TIMEOUT_SECONDS:-60}'
  export SPE_MCP_LOG_LEVEL='${SPE_MCP_LOG_LEVEL:-warning}'
  export SPE_STORAGE_PROTECTION_WINDOW_SECONDS='${SPE_STORAGE_PROTECTION_WINDOW_SECONDS:-86400}'
  export SPE_STORAGE_SEED_BYTES='${SPE_STORAGE_SEED_BYTES:-8388608}'
  export SPE_STORAGE_MINIMUM_MUTATION_COST_BYTES='${SPE_STORAGE_MINIMUM_MUTATION_COST_BYTES:-4096}'
  export SPE_STORAGE_SNAPSHOT_RETENTION='${SPE_STORAGE_SNAPSHOT_RETENTION:-8}'
  export SPE_STORAGE_RESTORE_ON_STARTUP='${SPE_STORAGE_RESTORE_ON_STARTUP:-true}'
  export SPE_PUBLIC_HOSTS='${public_hosts}'
  export SPE_PUBLIC_SCHEME='${public_scheme}'
  export SPE_PUBLIC_PORT='${public_port}'
  export SPE_ROOT_REDIRECT_PATH='${root_redirect_path}'
  export SPE_ALLOWED_HOSTS='${SPE_ALLOWED_HOSTS:-}'
  export SPE_ALLOWED_ORIGINS='${SPE_ALLOWED_ORIGINS:-}'
  export SPE_ALLOW_UNSAFE_PYTHON='${SPE_ALLOW_UNSAFE_PYTHON:-false}'
  export SPE_MANAGE_LEGACY_SYSTEMD='${manage_legacy_systemd}'
  export SPE_LEGACY_SYSTEMD_UNITS='${legacy_systemd_units}'
  export SPE_INSTALL_NGINX_SITE='${install_nginx_site}'
  export SPE_NGINX_SITE_NAME='${nginx_site_name}'
  export SPE_NGINX_SITE_MODE='${nginx_site_mode}'
  compose down --remove-orphans || true
  remove_managed_container_on_port
  stop_matching_legacy_units
  if port_in_use; then
    echo \"Port \${SPE_PUBLISHED_PORT} is still in use after Docker teardown and legacy handoff.\" >&2
    print_port_owner >&2
    exit 1
  fi
  compose up -d --build --remove-orphans
  install_nginx_site
  if command -v curl >/dev/null 2>&1; then
    ready=0
    attempt=0
    while [ \"\$attempt\" -lt 20 ]; do
      if curl -fsS 'http://127.0.0.1:${published_port}${streamable_http_path}/server_info' >/dev/null \
        && curl -fsS 'http://127.0.0.1:${published_port}${streamable_http_path}/tool_registry' >/dev/null; then
        ready=1
        break
      fi
      attempt=\$((attempt + 1))
      sleep 2
    done
    if [ \"\$ready\" -ne 1 ]; then
      echo 'MCP container did not become ready in time.' >&2
      exit 1
    fi
    if [ \"\${SPE_INSTALL_NGINX_SITE}\" = 'true' ]; then
      first_host=\$(printf '%s' \"\${SPE_PUBLIC_HOSTS}\" | awk -F',' '{print \$1}')
      curl -fsSI -H \"Host: \$first_host\" http://127.0.0.1/ >/dev/null
      curl -fsS -H \"Host: \$first_host\" \"http://127.0.0.1\${SPE_STREAMABLE_HTTP_PATH}/server_info\" >/dev/null
    fi
  fi
"

echo "Remote Docker deployment completed for ${remote}:${published_host}:${published_port}${streamable_http_path}"
