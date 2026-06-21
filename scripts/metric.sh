
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

EVAL_SAVE_PATH=""

EVAL_TEST_PATH="./test"

EVAL_UNSCENE_PATH="./unscene"

EVAL_UNOBJECT_PATH="./unobject"

OBJECT_INFO_PATH="./map_spawnarea_info.json"

METRIC_PY="$PROJECT_DIR/utils/metric.py"

if [[ ! -f "$METRIC_PY" ]]; then
    echo "[ERROR] metric.py does not exist: $METRIC_PY" >&2
    exit 1
fi

python "$METRIC_PY" \
    --eval_save_path "$EVAL_SAVE_PATH" \
    --eval_test_path "$EVAL_TEST_PATH" \
    --eval_unscene_path "$EVAL_UNSCENE_PATH" \
    --eval_unobject_path "$EVAL_UNOBJECT_PATH" \
    --object_info_path "$OBJECT_INFO_PATH"