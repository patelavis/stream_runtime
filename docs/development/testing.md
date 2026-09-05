# Testing

Run `pytest -q`. Tests cover range reads, metadata, cache bounds, memory failures, async reads, tiled inference, and the ASGI server. The server test uses `httpx.ASGITransport`, so it does not require a network socket. Add a regression test before changing a memory or streaming invariant.
