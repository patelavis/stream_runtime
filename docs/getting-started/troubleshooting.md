# Troubleshooting

**`prepared model requires manifest.json and model.path`:** run `prepare` first and point `serve` at the prepared directory. **`local model not found`:** verify the directory is local and contains both files. **`MemoryBudgetExceeded`:** increase the managed budget or use an operator with a supported tiling strategy. **401:** use `Authorization: Bearer <configured key>` or explicitly pass `--no-auth`. **No tokenizer:** this repository currently uses the deterministic demo encoder; real tokenizer support is planned.
