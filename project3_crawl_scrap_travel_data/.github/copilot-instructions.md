```instructions
## AI Coding Instructions — Travel Data Platform (concise)

This repo contains a FastAPI backend, a React frontend, and data ingestion code. The aim of these instructions is to help code-generating agents make safe, correct, and idiomatic changes quickly.

1) Big picture
- Backend: `web_dashboard/backend/` — FastAPI app created via an application factory (`core/app_factory.py`). Entrypoints: `main.py`, `main_frontend.py`.
- Frontend: `web_dashboard/frontend/` — React SPA. API client: `src/services/apiService.js` (dev proxy -> `http://localhost:8000`).
- Data layer: `data_platform/` holds ingestion, storage and analytics code (RapidAPI ingestion, Mongo helpers).

2) Safe editing patterns (project-specific)
- Async-first where present: prefer `async def` + `motor` for DB usage in `core/dependencies.py` and routers. If making a sync-only change, preserve existing async APIs.
- Graceful optional imports: many modules guard optional features with try/except (see `core/app_factory.py`) — keep that pattern when adding optional integrations.
- Standard API response shape: follow {success, status_code, data, timestamp, request_id} used by existing routers.

3) Important developer workflows (exact commands)
- Start backend (dev):
```powershell
cd "web_dashboard/backend"; python main.py
```
- Start frontend (dev):
```powershell
cd "web_dashboard/frontend"; npm install; npm start
```
- Integrated launcher: `cd "web_dashboard/backend"; python main_frontend.py integrated`
- Run key tests from repo root:
```powershell
python test_rapidapi_integration.py
python test_search_results.py
```

4) Conventions & gotchas
- Config via `.env` and `core.config.get_config()` — prefer reading settings from that helper.
- RapidAPI ingestion lives in `data_platform/ingestion/` and expects `RAPIDAPI_KEY` in env. Tests reference `data_platform.ingestion.google_places_ingestion`.
- MongoDB connection: `MONGO_URI` used across scripts; scripts often use synchronous `pymongo` while backend prefers async `motor` — be careful when changing shared helpers.
- Multiple `main_*.py` entrypoints exist; avoid introducing new entrypoints unless necessary (prefer `main.py` / `main_frontend.py`).

5) Integration touchpoints to inspect before changes
- App lifecycle & routers: `core/app_factory.py`, `web_dashboard/backend/routers/`.
- Startup & env: `main.py`, `core/config.py` (or `core/config` helpers).
- DB helpers: `core/dependencies.py`, `data_platform/storage/`, and scripts at repo root like `generate_tours.py` and `fetch_and_save_places.py` for examples.
- Background tasks: `routers/search.py` demonstrates using FastAPI `BackgroundTasks` and returning operation ids.

6) Small examples (copy-paste)
- Add a background job:
```python
from fastapi import BackgroundTasks
@router.post('/search')
async def start_search(req: SearchRequest, background_tasks: BackgroundTasks):
    search_id = f"search_{int(time.time())}"
    await storage.create_request(search_id, req.dict())
    background_tasks.add_task(process_search_request, search_id, req)
    return {"success": True, "search_id": search_id}
```

7) Quick checklist before committing
- Run unit/integration tests in repo root.
- Preserve async/sync separation (don't mix `pymongo` and `motor` in the same helper).
- Keep graceful fallback behavior for optional features.
- Update `web_dashboard/frontend/src/services/apiService.js` if you change backend routes or response shapes.

If any of the above is unclear or you'd like a longer version (deployment, CI, or security), tell me which area to expand.
```
## AI Coding Instructions — Travel Data Platform

This file gives focused, actionable guidance for code-generating agents working in this repository.

1) Big picture (what to change and where)
- Backend: `web_dashboard/backend/` — FastAPI app built via an application factory (`core/app_factory.py`) and started from `main.py` or `main_frontend.py`.
- Frontend: `web_dashboard/frontend/` — React app; API client lives in `src/services` and the dev proxy points to `http://localhost:8000`.

2) Key patterns to follow (concrete, repository-specific)
- Async-first: use async/await throughout; prefer `motor` for DB calls and initialize via `core.dependencies.get_dependencies()`.
- Background jobs: for long-running tasks add them with FastAPI `BackgroundTasks` and return an operation id. See `routers/search.py` and `main.py` startup pattern.
- Config via env & dataclasses: load `.env` in `web_dashboard/backend/data_platform/`; prefer `core.config.get_config()` to access settings.

3) Important developer workflows (exact commands)
- Start backend (dev):
```powershell
cd "web_dashboard/backend"; python main.py
```
- Start frontend (dev):
```powershell
cd "web_dashboard/frontend"; npm install; npm start
```
- Integrated launcher (starts both):
```powershell
cd "web_dashboard/backend"; python main_frontend.py integrated
```
- Run tests (select):
```powershell
python test_rapidapi_integration.py
python test_search_results.py
```

4) Files and locations to inspect when changing behavior
- App creation & lifecycle: `core/app_factory.py` (lifespan, middleware, routers)
- Startup & env loading: `main.py` (loads `.env` and calls `create_app()`)
- Dependency init / storage: `core/dependencies.py` and `data_platform/storage/`
- Routers: `web_dashboard/backend/routers/` (search, analytics, health, frontend)
- Frontend API client: `web_dashboard/frontend/src/services/apiService.js`

5) Conventions & gotchas
- Never assume synchronous DB drivers — prefer async `motor` APIs already used in the repo.
- Graceful degradation is common: code often imports optional modules inside try/except and skips features if missing (see `core/app_factory.py`). Keep that behavior when adding optional integrations.
- Standardized API response shape is used (success, status_code, data, timestamp, request_id). Mirror it for new endpoints.

6) Integration points & external deps
- RapidAPI integrations live under `data_platform/ingestion/` and expect `RAPIDAPI_KEY` in env.
- MongoDB connection string from `MONGO_URI` env; Redis/Elasticsearch referenced in docs — verify presence before using.

7) Quick examples (copy-paste friendly)
- Add a background search task:
```python
from fastapi import BackgroundTasks

@router.post('/search')
async def start_search(req: SearchRequest, background_tasks: BackgroundTasks):
    search_id = f"search_{int(time.time())}"
    await storage.create_request(search_id, req.dict())
    background_tasks.add_task(process_search_request, search_id, req)
    return {"success": True, "search_id": search_id}
```

8) Project structure & cleanup notes
- Core directories: `web_dashboard/backend/` (FastAPI), `web_dashboard/frontend/` (React), `data_platform/` (ingestion/storage/analytics).
- Avoid legacy/duplicate files: `version1/`, `version2/`, `note/` (personal notebooks), multiple `main_*.py` entrypoints — consolidate to `main.py` and `main_frontend.py`.
- Remove build artifacts: `__pycache__/`, `.pyc`, `frontend/build/` — regenerate as needed.
- Key subdirs: `core/` (config, dependencies), `routers/` (API endpoints), `data_platform/` (data processing pipeline).

9) Where to run tests and health checks
- Health: GET `/health` (Swagger UI at `/docs`)
- Unit/integration tests at repo root: `test_rapidapi_integration.py`, `test_search_results.py`.

If anything here is unclear or you'd like me to expand any section (deployment, CI, or security patterns), tell me which area to elaborate and I will iterate.