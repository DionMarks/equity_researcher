# Container image for bankresearch, built in two stages so no compiler or pip cache ends up
# in the final image. `make docker-build`, `make docker-ingest` and `make docker-run` use it.

ARG PYTHON_VERSION=3.12

# --- build -----------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /src
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
# A normal install rather than an editable one, since an editable install would point
# back at /src, which doesn't exist in the runtime stage. Pinned to the same versions the
# tests ran against.
RUN pip install -c constraints.txt ".[pdf,llm]"

# --- runtime ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TERM=xterm-256color

# Runs as an unprivileged user. Nothing here needs root, and this process parses PDFs
# downloaded from the internet.
RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# Read at run time: configuration, the list of filings to download and the eval cases.
# The prompts ship inside the package.
COPY --chown=app:app config ./config
COPY --chown=app:app data/manifest.yaml ./data/manifest.yaml
COPY --chown=app:app evals/cases ./evals/cases
COPY --chown=app:app scripts ./scripts

# The filings, the database and every run's output all live on one volume, so they
# survive the container and a second run doesn't download everything again.
RUN mkdir -p /app/data/runs && chown -R app:app /app
VOLUME ["/app/data"]

USER app

# Settings normally find the repo root by walking up from the package, which here would
# be site-packages, so the paths are set explicitly. They need the BANKRESEARCH_ prefix:
# without it they're ignored.
ENV BANKRESEARCH_CONFIG_DIR=/app/config \
    BANKRESEARCH_DATA_DIR=/app/data \
    BANKRESEARCH_RUNS_DIR=/app/data/runs \
    BANKRESEARCH_EVALS_DIR=/app/evals

# No API key is baked in. Pass one per run with `docker run -e ANTHROPIC_API_KEY ...`.
# Without one a question stops and says why: every answer is written and checked by the model.

HEALTHCHECK --interval=1m --timeout=10s --start-period=5s --retries=1 \
    CMD bankresearch config-check >/dev/null || exit 1

ENTRYPOINT ["bankresearch"]
CMD ["--help"]
