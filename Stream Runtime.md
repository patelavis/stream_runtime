# Stream Runtime

A CPU-first proof-of-concept for **disk-backed, architecture-aware model inference under a user-defined managed memory budget**. The complete model remains in a local `.safetensors` file; only requested tensor ranges and current operator buffers are loaded.

## Important memory semantics

`--ram-budget` is a hard limit for allocations explicitly managed by this runtime (weights, activations, temporary reservations, and future cache storage). It is **not** a promise that total process RSS, Python, PyTorch, CUDA, shared libraries, or filesystem page cache will remain below that number. An allocation that would exceed the managed budget raises `MemoryBudgetExceeded` rather than silently proceeding.

## Architecture

Preparation reads only safetensors metadata and emits `prepared_model/manifest.json`. The generic adapter groups `.weight` and `.bias` tensors into logical nodes, preserving computational boundaries rather than splitting arbitrary byte ranges. Inference loads the manifest, streams the current node's ranges, computes on CPU, releases reservations, and advances to the next node.

The storage layer uses an 8-byte safetensors header read followed by metadata parsing and `read`/seek range reads. It never calls `safetensors.torch.load_file` and never materializes the complete state dictionary. `TensorStore` hides the storage format from the engine and includes a bounded LRU chunk cache implementation.

Large linear operators use output-row tiling: `Y = X @ W.T` is computed one weight tile at a time. A logical node remains one node; tiling is an internal execution strategy. The deterministic planner marks a node `direct` when its estimated weights fit or `tiled` otherwise.

## Installation

```bash
python -m pip install -e .
```

Dependencies are Python, PyTorch, NumPy, safetensors, and pytest. CPU execution is the supported first milestone. CUDA/VRAM, transformer-specific adapters, disk-backed activations, quantization, and asynchronous prefetching are future work.

## Quick start

```bash
python examples/create_test_model.py --output test_model.safetensors
python -m stream_runtime inspect test_model.safetensors
python -m stream_runtime prepare test_model.safetensors --output prepared_model
python -m stream_runtime plan prepared_model --ram-budget 1M
python -m stream_runtime run prepared_model --ram-budget 1M
```

For a real input, save a PyTorch tensor with `torch.save(x, 'input.pt')` and pass `--input input.pt`; use `--output output.pt` to save the result.

Supported sizes are raw bytes and binary `K`, `KB`, `M`, `MB`, `G`, and `GB`, e.g. `512M`, `2GB`, or `1048576`.

## Tests and packaging

```bash
pytest -q
python build_zip.py
```

The ZIP contains the complete source project. The included prototype tests cover header/metadata parsing, exact ranges, bounded cache behavior, hard allocation failures, tiled linear correctness, and an end-to-end prepared manifest.

## Limitations

This is a working subset, not universal model support. The generic analyzer is best-effort and currently targets sequential linear layers whose tensor names end in `.weight`/`.bias`; it does not claim support for arbitrary Hugging Face graphs. Activation accounting is available through `RamActivationStore` but the demo engine keeps the current PyTorch activation in memory and does not yet implement disk-backed activation spilling. PyTorch's own allocator and process overhead are outside the managed accounting. If an operator has an unavoidable working set larger than the budget and no tiling implementation, execution must be rejected rather than pretending arbitrary models can run with arbitrarily tiny memory.

## Roadmap

Add transformer adapters (Llama, Mistral, Qwen, Gemma, Phi, GPT-2, T5, BERT), attention/KV-cache tiling, disk-backed activations, stronger planner accounting, safe prefetching, mmap with explicit accounting, CUDA/VRAM management, quantization, and I/O/compute overlap.
