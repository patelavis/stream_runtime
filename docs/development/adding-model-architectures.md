# Adding Model Architectures

1. Identify local config and module hierarchy. 2. Define complete logical boundaries. 3. Map tensor names to nodes. 4. Implement an `ArchitectureAdapter`. 5. Map operators. 6. Add local tokenizer/chat-template handling. 7. Add deterministic reference-vs-streaming tests. 8. Document unsupported cases and confidence. 9. Register the adapter. The current generic adapter is the concrete baseline, but it is not a full transformer architecture.
