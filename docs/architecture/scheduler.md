# Scheduler

`SyncScheduler` is the simple scheduler. `AsyncScheduler` provides a conservative prefetch admission hook and does not yet retain prefetched tensor data. The HTTP request manager uses an asyncio lock to allow one active generation and queue others. Batching and continuous scheduling are planned.
