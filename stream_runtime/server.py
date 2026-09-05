"""FastAPI server for completely local, offline inference."""

import asyncio, json, logging, time, uuid
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .server_core import LocalModel, ModelRegistry, RequestManager

log = logging.getLogger("stream_runtime.server")


def error(message, typ="invalid_request_error", code=None, status=400):
    return JSONResponse(
        {"error": {"message": message, "type": typ, "code": code}}, status_code=status
    )


def create_app(
    model_dir,
    ram_budget,
    model_id=None,
    api_key="local",
    no_auth=False,
    request_timeout=None,
    cache_size=0,
):
    from pathlib import Path

    directory = Path(model_dir)
    model_id = model_id or directory.name
    registry = ModelRegistry([directory])
    registry.discover()
    model = LocalModel(model_id, directory, ram_budget, cache_size)
    manager = RequestManager()
    app = FastAPI(title="Stream Runtime Local API", docs_url=None, redoc_url=None)
    app.state.model = model
    app.state.registry = registry
    app.state.manager = manager
    app.state.api_key = api_key
    app.state.no_auth = no_auth
    app.state.request_timeout = request_timeout

    def authorized(request):
        if no_auth:
            return True
        return request.headers.get("authorization", "") == f"Bearer {api_key}"

    @app.get("/health")
    async def health():
        return {"status": "ok", "offline": True, "runtime": "stream-runtime"}

    @app.get("/v1/models")
    async def models():
        data = [
            {"id": mid, "object": "model", "owned_by": "local"}
            for mid in registry.discover()
        ]
        if not data:
            data = [{"id": model_id, "object": "model", "owned_by": "local"}]
        return {"object": "list", "data": data}

    @app.get("/v1/status")
    async def status():
        s = model.status()
        s.update(
            {
                "queue_length": manager.queue_length,
                "active_request": manager.active_request,
            }
        )
        return s

    def prompt_from(body):
        if "messages" in body:
            return "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}"
                for m in body["messages"]
            )
        return str(body.get("prompt", ""))

    async def execute(body, request_id):
        prompt = prompt_from(body)

        async def work():
            return await asyncio.to_thread(model.infer, prompt)

        result = await manager.run(request_id, work)
        return result

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if not authorized(request):
            return error(
                "invalid local API key", "authentication_error", "INVALID_API_KEY", 401
            )
        try:
            body = await request.json()
            text = (
                await asyncio.wait_for(
                    execute(body, uuid.uuid4().hex), app.state.request_timeout
                )
                if app.state.request_timeout
                else await execute(body, uuid.uuid4().hex)
            )
        except asyncio.TimeoutError:
            return error("request timed out", "timeout_error", "REQUEST_TIMEOUT", 408)
        except Exception as exc:
            log.exception("chat request failed")
            return error(str(exc), "server_error", "INFERENCE_ERROR", 500)
        created = int(time.time())
        response = {
            "id": "chatcmpl-local-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": created,
            "model": body.get("model", model_id),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        }
        if not body.get("stream"):
            return response

        async def events():
            for word in text.split(" "):
                chunk = {
                    "id": response["id"],
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": response["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": word + " "},
                            "finish_reason": None,
                        }
                    ],
                }
                yield "data: " + json.dumps(chunk) + "\n\n"
                await asyncio.sleep(0)
            yield "data: " + json.dumps(
                {
                    "id": response["id"],
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": response["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        if not authorized(request):
            return error(
                "invalid local API key", "authentication_error", "INVALID_API_KEY", 401
            )
        body = await request.json()
        text = await execute(body, uuid.uuid4().hex)
        return {
            "id": "cmpl-local-" + uuid.uuid4().hex,
            "object": "text_completion",
            "model": body.get("model", model_id),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    return app


def run_server(model_dir, host="127.0.0.1", port=8000, ram_budget=2**30, **kwargs):
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(model_dir, ram_budget, **kwargs)
    print(
        f"Stream Runtime Server\nModel: {model_dir}\nRAM budget: {ram_budget} bytes\nHost: {host}\nPort: {port}\nInternet: disabled\nStatus: READY"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
