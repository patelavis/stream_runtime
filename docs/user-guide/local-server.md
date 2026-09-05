# Local Server

```bash
python -m stream_runtime serve --model ./prepared_model --host 127.0.0.1 --port 8000 --ram-budget 512M --api-key local
```

The server is local-only by default. It uses FastAPI/Uvicorn, runs blocking inference in a worker thread, and serializes active generations through an asyncio lock. `--trace` is currently available on the CLI `run` path; server logging is normal Uvicorn/Python logging (**server trace flag: planned**).
