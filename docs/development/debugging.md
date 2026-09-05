# Debugging

Use `python -m stream_runtime run ... --trace` to observe node reads, execution, and release. Inspect `SafeTensorStream.read_log`, `bytes_read`, `reads`, and `MemoryManager.stats`. For HTTP issues, run Uvicorn in the foreground and inspect structured client errors plus server logs.
