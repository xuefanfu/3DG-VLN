#!/bin/bash

set -e

# 获取项目根目录：scripts 的上一级目录
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Running utils/metric.py ..."

python "$PROJECT_DIR/utils/metric.py"
