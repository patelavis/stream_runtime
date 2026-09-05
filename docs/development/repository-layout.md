# Repository Layout

`stream_runtime/storage` handles range reads and cache; `memory` handles reservations; `graph` stores manifests; `architecture` analyzes models; `operators` computes; `planner` chooses strategies; `activation` stores activations; `runtime` loads/executes/generates; `server_core.py` owns local model/request state; `server.py` owns HTTP adaptation; `examples` contains runnable scripts; `tests` contains unit/integration tests.
