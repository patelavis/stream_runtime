# ADR 0001: Logical Nodes, Physical Tiles

**Decision:** preserve architecture-aware logical nodes and use internal tiling for oversized operators. **Reason:** arbitrary byte boundaries do not preserve computation semantics. **Consequence:** every new architecture adapter must define valid node boundaries; every tiled operator must prove its working set and output fidelity.
