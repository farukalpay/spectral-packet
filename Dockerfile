FROM python:3.11-slim

ARG TORCH_CPU_VERSION=2.11.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPE_PORT=8000 \
    SPE_STREAMABLE_HTTP_PATH=/mcp \
    SPE_SCRATCH_DIR=/data/scratch \
    SPE_LOG_FILE=/data/logs/mcp.log \
    SPE_MCP_MAX_CONCURRENT=2 \
    SPE_MCP_SLOT_TIMEOUT_SECONDS=60 \
    SPE_MCP_LOG_LEVEL=warning

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY docker/entrypoint.sh /usr/local/bin/spectral-packet-entrypoint

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_CPU_VERSION}+cpu" \
    && python -m pip install ".[mcp,files,sql]" \
    && chmod +x /usr/local/bin/spectral-packet-entrypoint \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/scratch /data/logs \
    && chown -R appuser:appuser /app /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import json, urllib.request; payload=json.load(urllib.request.urlopen('http://127.0.0.1:8000/mcp/server_info', timeout=5)); assert payload['transport']=='streamable-http'"

ENTRYPOINT ["/usr/local/bin/spectral-packet-entrypoint"]
