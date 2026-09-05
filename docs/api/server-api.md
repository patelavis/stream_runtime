# Server API

`create_app(model_dir, ram_budget, model_id=None, api_key='local', no_auth=False, request_timeout=None, cache_size=0)` returns a FastAPI application. `run_server` launches Uvicorn. The server endpoints are documented in `docs/user-guide/openai-api.md`.
