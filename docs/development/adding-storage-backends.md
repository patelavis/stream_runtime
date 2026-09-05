# Adding Storage Backends

Implement the range-read contract used by `TensorStore` and preserve metadata descriptors. Keep byte reads instrumentable. Add tests proving unrelated ranges are not read. A future sharded or mmap backend must not change the engine's memory semantics.
