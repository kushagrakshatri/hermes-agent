#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HERMES_HOME_DIR="${HERMES_HOME:-$ROOT_DIR/.hermes}"
DEBUG_PORT="${HERMES_CHROME_DEBUG_PORT:-9222}"
PROFILE_DIR="${HERMES_HOME_DIR}/chrome-debug"
CDP_URL="http://127.0.0.1:${DEBUG_PORT}"
CHROME_APP="Google Chrome"

mkdir -p "$PROFILE_DIR"

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$DEBUG_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Chrome debug port is already listening at ${CDP_URL}"
  echo "Profile dir: ${PROFILE_DIR}"
  exit 0
fi

open -na "$CHROME_APP" --args \
  --remote-debugging-port="$DEBUG_PORT" \
  --user-data-dir="$PROFILE_DIR" \
  --no-first-run \
  --no-default-browser-check

if command -v nc >/dev/null 2>&1; then
  for _ in {1..15}; do
    if nc -z 127.0.0.1 "$DEBUG_PORT" >/dev/null 2>&1; then
      echo "Chrome debug port is ready at ${CDP_URL}"
      echo "Profile dir: ${PROFILE_DIR}"
      exit 0
    fi
    sleep 1
  done
fi

echo "Started Chrome debug instance"
echo "CDP URL: ${CDP_URL}"
echo "Profile dir: ${PROFILE_DIR}"
echo "Chrome may still be starting. If Hermes cannot attach yet, rerun this script and try again in a few seconds."
