#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

cd "${REPO_ROOT}"
venv/bin/python -m pytest p3/tests "$@"
