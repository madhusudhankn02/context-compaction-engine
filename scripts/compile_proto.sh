#!/usr/bin/env bash
# scripts/compile_proto.sh
#
# Generates compaction_pb2.py and compaction_pb2_grpc.py from the .proto file.
# Run this once after cloning, and again whenever compaction.proto changes.
#
# Prerequisites:
#   pip install grpcio grpcio-tools   (already in pyproject.toml [grpc] extra)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/src/compaction_engine/bridge/grpc"
PROTO_FILE="$PROTO_DIR/compaction.proto"
OUT_DIR="$PROTO_DIR"

echo "Compiling $PROTO_FILE → $OUT_DIR"

python -m grpc_tools.protoc \
    --proto_path="$REPO_ROOT/src" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_FILE"

# grpc_tools generates absolute import paths; fix to relative for package use.
sed -i 's/^import compaction_pb2/from compaction_engine.bridge.grpc import compaction_pb2/' \
    "$OUT_DIR/compaction_pb2_grpc.py" 2>/dev/null || \
  sed -i '' 's/^import compaction_pb2/from compaction_engine.bridge.grpc import compaction_pb2/' \
    "$OUT_DIR/compaction_pb2_grpc.py"

echo "Done. Generated files:"
ls -lh "$OUT_DIR"/compaction_pb2*.py
