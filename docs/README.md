# Documentation

This documentation describes the current Stream Runtime repository, not an aspirational implementation. Read [Getting Started](getting-started/installation.md) first, then [Architecture Overview](architecture/overview.md) and [Development Setup](development/development-setup.md).

## Documentation status

**Implemented:** CPU safetensors range streaming, managed memory accounting, generic sequential linear execution, tiled linear operator, local FastAPI server, model registry, queued single-request serving, JSON/SSE endpoints, local API keys, tracing, and test coverage.

**Experimental:** generic graph analysis, embedding/residual executor hooks, async worker-thread storage facade, conservative prefetch admission, token generator interface, local deterministic chat adapter.

**Planned:** real transformer/tokenizer adapters, disk-backed activations, CUDA/VRAM execution, cache integration in serving, true incremental language-model decoding, and broader OpenAI features.
