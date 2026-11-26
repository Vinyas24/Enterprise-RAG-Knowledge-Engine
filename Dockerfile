# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# base: shared runtime environment (dependencies are installed per target)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    PYTHONHASHSEED=random

WORKDIR /app

# Run as an unprivileged user; uid/gid are pinned so bind-mounted files have
# predictable ownership on the host.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

# Upload mount point, owned by the runtime user. Docker seeds a *new named
# volume* from the image directory, including ownership, so the unprivileged
# process can write to it without a root entrypoint or a chown sidecar.
RUN mkdir -p /data/uploads && chown -R app:app /data

# ---------------------------------------------------------------------------
# dev: editable install + dev tooling; source and tests are bind-mounted by
# docker compose, so the image only needs to carry the dependencies.
# ---------------------------------------------------------------------------
FROM base AS dev

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY eval ./eval

RUN pip install -e ".[dev]"

COPY tests ./tests

# The container runs as uid 1000; without this, tools that write into the
# working tree (pytest cache, coverage) fail with permission errors.
RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

CMD ["uvicorn", "rag_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# ---------------------------------------------------------------------------
# runtime: production-shaped image (no dev dependencies, no source tree)
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY eval ./eval

RUN pip install . && rm -rf /app/src

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

CMD ["uvicorn", "rag_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
