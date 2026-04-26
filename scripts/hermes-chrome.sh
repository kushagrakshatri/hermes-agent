#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export HERMES_HOME="${HERMES_HOME:-$ROOT_DIR/.hermes}"
export BROWSER_CDP_URL="${BROWSER_CDP_URL:-http://127.0.0.1:9222}"

"$ROOT_DIR/scripts/start-chrome-debug.sh"

source "$ROOT_DIR/venv/bin/activate"
exec hermes "$@"
