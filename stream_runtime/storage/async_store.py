from concurrent.futures import ThreadPoolExecutor


class AsyncTensorStore:
    """Async facade over blocking storage; reads execute in worker threads."""

    def __init__(self, store, max_workers=2):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    async def read(self, name, offset=0, length=None):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool, self.store.read, name, offset, length
        )

    def close(self):
        self.pool.shutdown(wait=True)
