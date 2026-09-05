# AEGIS reproduction image.
#
# Pinned to a specific Python patch release and a specific numpy so a run in
# 2027 produces the same numbers as a run today: every result in the paper is
# a function of floating-point arithmetic over a fixed corpus, and an
# unpinned BLAS or numpy minor bump is enough to move a fourth decimal place.
#
#   docker build -f docker/Dockerfile -t aegis-redaction:1.0.0 .
#   docker run --rm -v "$PWD/results:/app/results" aegis-redaction:1.0.0 make smoke
#   docker run --rm -v "$PWD/results:/app/results" aegis-redaction:1.0.0 make reproduce
#
# The image bundles the corpora, so no network access is needed at run time
# and the container can be run with --network=none.

FROM python:3.11.9-slim-bookworm

LABEL org.opencontainers.image.title="AEGIS" \
      org.opencontainers.image.description="Distribution-shift-certified risk control for PII redaction" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/aegis-redaction/aegis-finance"

# Determinism: no bytecode caching, no hash randomisation, single-threaded BLAS
# (thread count changes reduction order, which changes low-order bits).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AEGIS_DATA_ROOT=/app/data \
    AEGIS_RESULTS=/app/results \
    PYTHONPATH=/app/src

RUN apt-get update \
 && apt-get install --no-install-recommends -y make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer cache survives source edits.
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade "pip==24.0" \
 && python -m pip install "numpy==1.26.4" "pytest==8.2.0"

COPY pyproject.toml Makefile README.md LICENSE ./
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/ ./data/
COPY results/paper/ ./results/paper/

RUN python -m pip install --no-deps -e . \
 && mkdir -p /app/results \
 && python scripts/verify_data.py

# A non-root user, because an image that redacts other people's documents
# should not run as root.
RUN useradd --create-home --uid 10001 aegis \
 && chown -R aegis:aegis /app
USER aegis

# Fail the build if the package cannot even be imported.
RUN python -c "import aegis; print('aegis', aegis.__version__)"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import aegis" || exit 1

ENTRYPOINT ["/usr/bin/env"]
CMD ["make", "smoke"]
