# Multi-stage build keeps the runtime image small.
# Builder uses uv to resolve deps; runtime is pure python + venv.

FROM python:3.13-slim AS builder

# uv is the project's chosen package manager — fast resolves + lockfile honoring.
COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /usr/local/bin/uv

# Node.js for the game-agent's `node --check` JS validator. Tiny — only the
# parser is invoked, no npm install.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lockfile + project metadata first (cache-friendly).
COPY pyproject.toml uv.lock README.md ./

# Sync deps into a project-local venv (no dev deps in production).
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project itself.
COPY backend ./backend
COPY evals ./evals
RUN uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

# `node` again at runtime — used by the game-agent's JS validator.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring the resolved venv + source from the builder stage.
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Default to in-memory SQLite; override via ADAPTIVE_LEARNING_DB_URL.
# Required env at runtime: ANTHROPIC_API_KEY, TAVILY_API_KEY.
EXPOSE 8000

# `uvicorn --host 0.0.0.0` binds to the container's network namespace; PaaS
# platforms (Railway / Fly / Render) route their assigned $PORT to it.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
