ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends git g++ && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-install-project --no-editable; \
    else uv sync --no-install-project --no-editable; fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-editable; \
    else uv sync --no-editable; fi

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# WNTR imports pkg_resources which was removed in setuptools>=82.
# Fix both venvs — /app/.venv (copied standalone) and /app/env/.venv (copied with env dir).
RUN for venv in /app/.venv /app/env/.venv; do \
      if [ -d "$venv" ]; then \
        "$venv/bin/python" -m ensurepip 2>/dev/null || true; \
        "$venv/bin/python" -m pip install --no-cache-dir "setuptools<82" 2>/dev/null || true; \
      fi; \
    done

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV WORKERS="1"
ENV ENABLE_WEB_INTERFACE="1"
ENV MAX_CONCURRENT_ENVS="1"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "cd /app/env && exec uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
