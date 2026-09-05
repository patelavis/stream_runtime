# Stream Runtime

A CPU-first prototype for disk-backed, architecture-aware model inference under a user-defined managed memory budget. The complete model remains on local `.safetensors`; only requested metadata and byte ranges are read.

## Guarantees and memory semantics

`--ram-budget` is a hard limit for allocations explicitly managed by this runtime: weights, activations, temporary buffers, cache, and future prefetch reservations. It is not a promise that total process RSS, Python, PyTorch, shared libraries, or filesystem page cache remain below that value. An allocation that would exceed the managed budget raises `MemoryBudgetExceeded`.

## Pipeline

Preparation parses safetensors metadata and emits `manifest.json` plus `metadata.json`. The generic architecture analyzer groups `.weight`/`.bias` tensors into logical computational nodes, preserving module boundaries. At inference, `NodeLoader` selects only the current node's tensors, `NodeExecutor` runs the supported operator, and the loader releases the node before the next dependency-authorized node executes. The engine validates dependencies and can emit trace events.

The storage layer reads the 8-byte header and JSON metadata once, then uses seek/range reads. It exposes `SafeTensorStream.metadata()`, `tensor(name)`, `TensorDescriptor`-like properties (`name`, `shape`, `dtype`, `nbytes`, `file_offset`, `file_length`), `read(offset, length)`, chunk iteration, read counts, byte counts, and read offset/length logs. It never uses `safetensors.torch.load_file`.

Linear operators use output-row tiling when the planner marks a node tiled. The logical node remains intact; only internal weight tiles are streamed. `AsyncTensorStore` provides real worker-thread-backed async reads, and `AsyncScheduler` provides conservative prefetch admission: prefetch is skipped when the managed budget cannot admit it. `TokenGenerator` preserves serialized token emission while permitting async internals to be added later.

## Installation

```bash
python -m pip install -e .
```

Dependencies: Python, PyTorch, NumPy, safetensors, and pytest. CUDA is not required. `DeviceMemoryManager`/VRAM support is intentionally not faked in this version.

## Commands

```bash
python examples/create_test_model.py --output test_model.safetensors
python -m stream_runtime inspect test_model.safetensors
python -m stream_runtime prepare test_model.safetensors --output prepared_model
python -m stream_runtime plan prepared_model --ram-budget 1M
python -m stream_runtime run prepared_model --ram-budget 1M --trace
python -m stream_runtime test
pytest -q
python build_zip.py
```

Supported budgets are raw bytes and binary `K`, `KB`, `M`, `MB`, `G`, and `GB`, such as `512M`, `2GB`, or `1048576`. Use `--input input.pt` with a tensor saved by `torch.save`; use `--output output.pt` to serialize the result.

## Supported first milestone

The included test model is a deterministic sequential linear network. The implementation also contains operator hooks for embedding, ReLU, and residual/add nodes, and the graph/adapter boundaries are designed for later transformer blocks. The generic analyzer is deliberately best-effort and does not claim universal Hugging Face support.

## Validation

The test suite covers safetensors metadata and range reads, chunk/cache behavior, hard memory enforcement, graph dependencies, tiled linear correctness, and end-to-end streaming execution. The CLI self-test creates a safetensors model, prepares it, plans it, and runs it. Reference-vs-streaming comparisons should use `torch.testing.assert_close`; tiled accumulation can differ slightly from a direct computation because floating-point accumulation order may differ.

## Limitations and roadmap

This version does not yet implement disk-backed activation spilling, transformer attention/KV-cache tiling, full reference-model loading, CUDA/VRAM management, quantization, mmap accounting, or overlapped compute/I/O. If an operator has an unavoidable working set larger than the budget and no supported tiling strategy, it must fail rather than silently exceed the budget. Future work adds architecture-specific adapters (Llama, Mistral, Qwen, Gemma, Phi, GPT-2, T5, BERT), stronger activation accounting, safe prefetch data reservation, and GPU devices.
