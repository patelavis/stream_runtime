# Adding Schedulers

Schedulers must honor graph dependencies, memory admission, ordered outputs, backpressure, and cleanup. Keep scheduler code independent of FastAPI. Start with a deterministic synchronous test, then add async timing tests. Prefetch must be admitted only when its buffers are accounted for.
