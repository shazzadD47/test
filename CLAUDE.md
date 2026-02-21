# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Delineate AI App — a FastAPI + Celery platform for AI-powered data extraction from scientific papers. Uses multiple LLM providers (Gemini, GPT, Claude) with LangChain/LangGraph for document processing, table extraction, plot digitization, and RAG-based chat.

## Common Commands

```bash
# Run locally with Docker
docker compose -f docker-compose-dev.yml up --build

# Run app directly
uvicorn app.main:app --reload

# Run a Celery worker (example: io-celery)
celery -A app.core.celery.app worker -n io-celery@%h -Q delineate-ai-queue --loglevel=INFO -P prefork --concurrency=4 --events

# Tests
pytest                                    # all tests
pytest --cov-report term --cov=app ./tests  # with coverage
pytest tests/core/                        # specific directory
pytest tests/core/test_something.py -v    # single file, verbose
make pytest                               # via Makefile

# Linting & formatting (pre-commit: black, isort, flake8, bandit, pycln, pyupgrade)
make pre-commit
pre-commit run --all-files

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
# In Docker: docker compose -f docker-compose-dev.yml exec app alembic upgrade head

# Clean caches
make clean
```

## Architecture

### Entry Point
`app/main.py` — FastAPI app with lifespan management. Routes are registered from `app/v2/` (legacy) and `app/v3/` (current).

### API Versioning
- `/v2/endpoints/` — Legacy (equation_extract, plot_digitization, csv_embedding)
- `/v3/endpoints/` — Current (25+ modules). Each module follows: `routers.py` → `services.py` → `tasks.py` (Celery) + `schemas.py`

### Celery Task System
- **Config:** `app/core/celery/app.py` — worker settings, autodiscovery
- **Queues:** `app/core/celery/queues.py` — `delineate-ai-queue` (default), `cpu-tasks`, `dosing-table`, `general-extraction`
- **Routing:** `app/core/celery/task_routes.py` — maps task names to queues
- Tasks not in `TASK_ROUTES` go to the default queue. Worker concurrency is set per-worker via K8s CLI `--concurrency` flag, overriding the Python config.

### Event Bus
`app/core/event_bus/` — RabbitMQ-based. `send_to_backend()` publishes results back to the backend service. Events defined in `BackendEventEnumType`.

### Database
- PostgreSQL + SQLAlchemy ORM (`app/core/database/`)
- Migrations: Alembic (`alembic/`)
- Models: `FileDetails`, `FigureDetails`, `PageDetails`, `TableDetails`

### LLM Integration
- Factory pattern: `app/core/auto/factory.py` — instantiates models by name
- Cost tracking: `app/core/callbacks/cost_handler.py` and `app/core/utils/decorators/cost_tracker.py`
- Supports OpenAI, Anthropic, Google Gemini via LangChain
- LangGraph used for stateful agents (`agent_chat`, `report_generator`) with Postgres checkpointing

### Vector Store
Qdrant (`app/core/vector_store.py`) for embeddings/RAG search.

### Auth
`app/auth/` — S2S JWT tokens for service-to-service, bearer tokens for user auth.

### Config
`app/configs.py` — Pydantic `BaseSettings`. All config via environment variables. Key prefixes: `CELERY_*`, `QDRANT_*`, `S3_SPACES_*`, `JWT_*`, `LANGFUSE_*`.

## Deployment

### K8s Structure (`k8s/`)
- `dev/`, `stage/`, `production/` — each has `deployments.yml`, `hpa.yml`, `services.yml`, `ingress.yml`
- Workers: `delineate-ai-default-celery`, `delineate-ai-cpu-celery`, `delineate-ai-general-extraction-celery`
- Redis: Memorystore (GCP managed) for dev/staging, separate instance for production
- Restart scripts: `scripts/restart_dev.sh`, `scripts/restart_staging.sh`, `scripts/restart_production.sh`

### Docker Compose (local dev)
- `io-celery`: handles `delineate-ai-queue`, `dosing-table`, `general-extraction`
- `cpu-celery`: handles `cpu-tasks`
- Redis + RabbitMQ + Flower included

## Code Style & Linting

Pre-commit hooks enforce formatting. Run `make pre-commit` before committing.

- **Formatter:** black (Python 3.11 target)
- **Import sorting:** isort
- **Unused imports:** pycln (--all)
- **Linting:** flake8 with bugbear, comprehensions, simplify plugins
- **Security:** bandit (excludes tests/ and app/utils/repl.py)
- **Syntax:** pyupgrade (--py311-plus) — use modern Python 3.11+ syntax (e.g. `X | Y` unions, not `Union[X, Y]`)
- **Large files:** max 70KB enforced by pre-commit
- **Test naming:** pytest-style (`test_` prefix required)

Code conventions:
- **No unnecessary comments.** Code should be self-explanatory. Don't add inline comments that restate what the code does.
- **Docstrings are welcome** for functions, classes, and modules — especially for complex logic or public APIs.
- **Type hints are required** on all function signatures (parameters and return types).
- Don't add comments or docstrings to code you didn't change.

## Code Quality Priorities

- **Memory leaks:** Watch for unbounded lists/dicts, unclosed connections, large objects held in closures, ThreadPoolExecutor futures not being collected, and Celery tasks accumulating data in worker memory across invocations.
- **Potential bugs:** Verify edge cases, null/None handling, race conditions in concurrent code (ThreadPoolExecutor, asyncio), and correct exception types in except blocks (e.g. `SoftTimeLimitExceeded` is `BaseException`, not `Exception`).
- **Security:** Guard against injection (SQL, command, prompt), validate all external input at API boundaries, never log secrets or credentials, and sanitize data before passing to LLM prompts.

## Key Patterns

- **Endpoint module pattern:** Each v3 endpoint follows a layered structure:
  - `routers.py` — FastAPI endpoints (HTTP layer)
  - `services.py` — Business logic / orchestration
  - `tasks.py` — Celery async tasks (background processing)
  - `schemas.py` — Pydantic request/response models
  - `configs.py`, `constants.py`, `prompts.py` — optional config/constants
  - `helpers/` — utility functions specific to the module
- **Agentic module pattern** (e.g. `extraction_templates/`, `agent_chat/`): Extends the base pattern with an `agent_services/` sublayer:
  - `agent_services/graph.py` — LangGraph StateGraph definition (nodes + edges)
  - `agent_services/agents.py` — LLM instantiation + tool binding
  - `agent_services/nodes.py` — Graph node functions (setup → agent → tools → exit loop)
  - `agent_services/prompts.py` — System prompts for the agent
  - `agent_services/schemas.py` — AgentState TypedDict
  - `agent_services/tools/` — Tool functions exposed to the LLM (Pydantic-decorated)
  - Flow: `routers.py` → `services.py` → `graph.py` → nodes loop (agent ↔ tools) → streaming JSON response
  - Uses Redis or Postgres checkpointing for persistent chat threads
- **General extraction pipeline:** `app/v3/endpoints/general_extraction/` — LangGraph workflow with nodes for preprocessing, query generation, context generation, and table finalization. Used by covariate extraction and other modules.
- **Parallel LLM calls:** `ContextThreadPoolExecutor` used in `general_extraction/services/helpers/context_helpers.py` and `query_helpers.py` for concurrent Gemini calls. Concurrency scales with number of labels.
- **Cost tracking:** `@track_all_llm_costs` decorator on Celery tasks propagates cost metadata through the pipeline.
