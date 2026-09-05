# Stream Runtime

**Stream Runtime is a CPU-first, architecture-aware inference prototype that keeps model weights on local SSD/HDD and streams only the working data required by each computational node.**

## Mission

Available RAM/VRAM should primarily determine execution strategy and performance, rather than determining whether a model can execute at all, provided its operators have a valid working-set strategy. A supported target is `model size > RAM`: for example, a 20 GB model with a 1 GB managed RAM budget and no VRAM should be attempted as SSD → RAM → compute → release → next node. If an operator cannot fit even with a supported tiling strategy, the runtime must fail clearly rather than silently changing the model.

This is not simply quantization. **Quantization** changes numerical representation; **compression** reduces storage size; **offloading** moves model data between devices; **streaming execution** reads only needed ranges; and **architecture-aware execution** chooses complete computational nodes before applying internal operator tiling.

## Architecture

```text
Local model on SSD/HDD
        ↓
Local model registry and manifest
        ↓
Architecture-aware graph
        ↓
Memory planner
        ↓
Node loader and TensorStore
        ↓
Safetensors range reads
        ↓
CPU operator / internal tiles
        ↓
Release managed buffers
        ↓
Next dependency-authorized node
```

The local server adds an async HTTP layer and single-active-request queue above this pipeline. Storage does not know HTTP, operators do not open files directly, and the server does not parse safetensors offsets.

## Current implementation

Implemented and tested are safetensors metadata/range streaming, read instrumentation, strict managed memory reservations, generic graph preparation, tiled CPU linear execution, local FastAPI/Uvicorn serving, local model discovery, JSON and SSE OpenAI-style endpoints, local Bearer authentication, request queueing, worker-thread inference, health/status metrics, trace output, and offline integration tests.

The generic analyzer and local chat adapter are **experimental**. They support the repository's deterministic sequential linear test model; they are not general Hugging Face transformer support. Real local tokenizers, chat templates, autoregressive decoding, attention/KV cache, disk-backed activations, CUDA/VRAM execution, serving cache integration, tool calls, embeddings, batching, and full API parity are **planned or not implemented**.

## Installation and quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python examples/create_test_model.py --output /tmp/model.safetensors
python -m stream_runtime prepare /tmp/model.safetensors --output /tmp/prepared
python -m stream_runtime inspect /tmp/model.safetensors
python -m stream_runtime plan /tmp/prepared --ram-budget 1M
python -m stream_runtime run /tmp/prepared --ram-budget 1M --trace
```

Start the local server:

```bash
python -m stream_runtime serve --model /tmp/prepared --host 127.0.0.1 --port 8000 --ram-budget 1M --api-key local
```

Then use:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer local' \
  -d '{"model":"prepared","messages":[{"role":"user","content":"Hello"}],"stream":false}'
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer local' \
  -d '{"model":"prepared","messages":[{"role":"user","content":"Explain Python generators."}],"stream":true}'
```

Use `python examples/client.py` for model discovery, JSON chat, and SSE chat. A standard OpenAI client can target `http://127.0.0.1:8000/v1` with any local API key, but compatibility is limited to the documented subset.

## Memory model

`--ram-budget` is a hard ceiling for explicitly managed runtime allocations, not a guarantee about total process RSS. The manager tracks current and peak weights, activations, temporary memory, cache, and related categories. Python, PyTorch allocator behavior, shared libraries, OS filesystem cache, and GPU allocations are outside this logical guarantee. Size values accept raw bytes and binary `K`, `KB`, `M`, `MB`, `G`, or `GB`.

## Offline operation

After packages and model files are installed locally, inference performs no model downloads, cloud API calls, telemetry, remote logging, DNS lookup, or external inference. The server binds to `127.0.0.1` by default. Use `--no-auth` only for trusted localhost development; explicitly choosing another host can expose the API to a network.

## Development and documentation

Run the tests with `pytest -q` or the self-test with `python -m stream_runtime test`. Build the distributable archive with `python build_zip.py`. Begin contributor onboarding at [CONTRIBUTING.md](CONTRIBUTING.md), then read [docs/README.md](docs/README.md), [architecture overview](docs/architecture/overview.md), [development setup](docs/development/development-setup.md), and [adding model architectures](docs/development/adding-model-architectures.md).

The documentation tree contains user guides, architecture notes, API references, development guides, a configuration reference, glossary, and an architecture decision record. Each document distinguishes implemented, experimental, planned, and unsupported behavior.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
