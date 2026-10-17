# syntax=docker/dockerfile:1

# --- Build stage -------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer, independent of source changes)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project itself
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- Runtime stage -------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Create an unprivileged user
RUN groupadd --system rubick && useradd --system --gid rubick --home /app rubick

COPY --from=builder --chown=rubick:rubick /app/.venv /app/.venv
COPY --from=builder --chown=rubick:rubick /app/src /app/src
COPY --chown=rubick:rubick README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------
# Runtime configuration (override at `docker run` time with -e / --env-file)
# --------------------------------------------------------------------------
# LLAMA_SERVER_URL     : URL of the llama-server backend to proxy (REQUIRED)
#                        e.g. http://192.168.0.9:8080 or http://host.docker.internal:8080
# RUBICK_HOST          : Bind host inside the container (default: 0.0.0.0)
# RUBICK_PORT          : Bind port inside the container (default: 8000)
# RUBICK_AUTO_GUARDIAN : Enable automatic safety pre-flight screening (default: false)
# RUBICK_INTENT_PARSER : Enable transparent intent auto-routing (default: true)
# RUBICK_EXPLAIN       : Attach natural language explanations to hijacked decisions (default: true)
# RUBICK_EMBED_PROMPT  : PromptEOL embedding template (default: "Text: {text}\nSummary in one word:")
# RUBICK_TIMEOUT       : Backend request timeout in seconds (default: 60.0)
ENV LLAMA_SERVER_URL="http://host.docker.internal:8080" \
    RUBICK_HOST="0.0.0.0" \
    RUBICK_PORT="8000" \
    RUBICK_AUTO_GUARDIAN="false" \
    RUBICK_INTENT_PARSER="true" \
    RUBICK_EXPLAIN="true" \
    RUBICK_TIMEOUT="60.0"

USER rubick

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"RUBICK_PORT\",\"8000\")}/health', timeout=3)" || exit 1

# Entrypoint script reads env vars and forwards them as CLI flags so that
# both `docker run -e LLAMA_SERVER_URL=...` and `docker run rubick --backend ...`
# work interchangeably.
COPY --chown=rubick:rubick docker-entrypoint.sh /app/docker-entrypoint.sh
USER root
RUN chmod +x /app/docker-entrypoint.sh
USER rubick

ENTRYPOINT ["/app/docker-entrypoint.sh"]
