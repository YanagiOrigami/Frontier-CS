#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXEC_ROOT=$(cd "$SCRIPT_DIR/../../../execution_env" && pwd 2>/dev/null || true)
if [[ -z "$EXEC_ROOT" ]]; then
  echo "Error: execution_env directory not found." >&2
  exit 1
fi

PYBIN="$EXEC_ROOT/.venv/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "Error: venv python not found at $PYBIN. Did you run prepare_env.sh?" >&2
  exit 1
fi

SOLUTION_PATH="$EXEC_ROOT/solution_env/solution.py"
SPEC_PATH="$SCRIPT_DIR/resources/submission_spec.json"
OUTPUT_JSON=$(CBL_LOG_LEVEL=WARNING "$PYBIN" "$SCRIPT_DIR/evaluator.py" --solution "$SOLUTION_PATH" --spec "$SPEC_PATH")
SCORE=$(python3 - <<'PY' "$OUTPUT_JSON"
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("combined_score", payload.get("score", 0)))
PY
)

echo "$OUTPUT_JSON" > "$EXEC_ROOT/evaluator_output.json"
echo "$SCORE"
