"""
gRPC server entry point.

Run with:
    python -m compaction_engine.bridge.grpc.server

Or via the Makefile target `make serve-grpc`.
Requires proto stubs — run `bash scripts/compile_proto.sh` first.
"""

from __future__ import annotations

import signal
import sys
from concurrent import futures

from compaction_engine.bridge.container import build_container
from compaction_engine.bridge.grpc.servicer import CompactionServicer, _require_stubs, compaction_pb2_grpc
from compaction_engine.config import EngineSettings
from compaction_engine.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_DEFAULT_PORT = 50051
_DEFAULT_MAX_WORKERS = 4


def serve(
    port: int = _DEFAULT_PORT,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    settings: EngineSettings | None = None,
) -> None:
    _require_stubs()

    try:
        import grpc
    except ImportError as exc:
        raise ImportError(
            "grpcio is not installed. Run: pip install grpcio"
        ) from exc

    resolved = settings or EngineSettings()
    configure_logging(resolved.log_level, resolved.enable_json_logging)
    container = build_container(resolved)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    compaction_pb2_grpc.add_CompactionServiceServicer_to_server(
        CompactionServicer(container), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info("gRPC server started", extra={"port": port, "max_workers": max_workers})

    def _handle_sigterm(*_: object) -> None:
        logger.info("SIGTERM received — stopping gRPC server")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping gRPC server")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
