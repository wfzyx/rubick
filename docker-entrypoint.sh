#!/bin/sh
# Entrypoint for the Rubick Gateway container.
#
# Reads the standard RUBICK_* / LLAMA_SERVER_URL environment variables
# (see Dockerfile and README.md for the full list) and forwards them as CLI
# flags to `rubick`. Any arguments passed to `docker run` are appended last,
# so they take precedence over the environment-derived defaults, e.g.:
#
#   docker run -e LLAMA_SERVER_URL=http://192.168.0.9:8080 rubick
#   docker run ghcr.io/wfzyx/rubick --backend http://192.168.0.9:8080 --no-explain
#
set -eu

ARGS=""

if [ -n "${LLAMA_SERVER_URL:-}" ]; then
    ARGS="$ARGS --backend $LLAMA_SERVER_URL"
fi

if [ -n "${RUBICK_HOST:-}" ]; then
    ARGS="$ARGS --host $RUBICK_HOST"
fi

if [ -n "${RUBICK_PORT:-}" ]; then
    ARGS="$ARGS --port $RUBICK_PORT"
fi

case "$(printf '%s' "${RUBICK_AUTO_GUARDIAN:-false}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) ARGS="$ARGS --auto-guardian" ;;
esac

case "$(printf '%s' "${RUBICK_INTENT_PARSER:-true}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) ARGS="$ARGS --intent-parser" ;;
    *) ARGS="$ARGS --no-intent-parser" ;;
esac

case "$(printf '%s' "${RUBICK_EXPLAIN:-true}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) ARGS="$ARGS --explain" ;;
    *) ARGS="$ARGS --no-explain" ;;
esac

# shellcheck disable=SC2086
exec rubick $ARGS "$@"
