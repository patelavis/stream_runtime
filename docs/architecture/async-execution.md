# Async Execution

`AsyncTensorStore` runs blocking reads in a `ThreadPoolExecutor`, avoiding the false pattern of calling blocking I/O directly in an async function. The server uses `asyncio.to_thread` for CPU inference. Node dependency execution remains serialized. Cancellation and prefetch cancellation are limited by Python worker semantics and are future work.
