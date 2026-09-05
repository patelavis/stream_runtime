# Contributing

Thank you for contributing to Stream Runtime. Start with [Development Setup](docs/development/development-setup.md), read [Architecture Overview](docs/architecture/overview.md), and run `pytest -q` before editing.

## Workflow

Open an issue for a substantial behavior change. Create a focused branch, make small commits, add or update tests, run the full suite, and update documentation status. Do not claim support for an architecture or API feature that has not been tested. Keep the managed-memory distinction explicit.

## Pull requests

A PR should explain the problem, design, affected modules, memory implications, test evidence, performance measurements when relevant, and known limitations. Include commands that reproduce validation. Reviews prioritize correctness, offline behavior, memory safety, output fidelity, and API compatibility in that order.

## Commit guidance

Use imperative, focused commit subjects such as `Add range-read instrumentation`. Avoid generated model files, credentials, and large binaries. See `SECURITY.md` for vulnerability reports and `CODE_OF_CONDUCT.md` for community expectations.
