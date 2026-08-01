# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only — these don't go into the runtime image.
RUN pip install --no-cache-dir hatchling

COPY pyproject.toml .
COPY src/ src/

# Build the wheel.
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Compaction Engine — REST Bridge"
LABEL org.opencontainers.image.version="0.1.0"

# Non-root user for production security posture.
RUN useradd --create-home --shell /bin/bash compaction
WORKDIR /app
USER compaction

# Copy the pre-built wheel and install with all runtime extras.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --find-links /wheels \
    "compaction-engine[anthropic]" \
    fastapi \
    "uvicorn[standard]"

# Copy source so uvicorn can locate compaction_engine.bridge.rest.app:app.
COPY --chown=compaction:compaction src/ src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Liveness probe: GET /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "compaction_engine.bridge.rest.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
