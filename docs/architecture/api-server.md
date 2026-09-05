# API Server

`create_app` constructs the FastAPI application without contacting external services. `run_server` starts Uvicorn. Errors are returned as `{error:{message,type,code}}`; detailed stack traces go to server logs. Default binding is `127.0.0.1`.
