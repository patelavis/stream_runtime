# Token Generation

`TokenGenerator.generate` is an async ordered interface. The current server returns a deterministic demo response rather than incremental language-model decoding. External token order is serialized by the generator/API design. KV cache and true autoregressive decoding are planned.
