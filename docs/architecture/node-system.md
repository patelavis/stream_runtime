# Node System

A node is a complete logical computational unit such as a linear layer. `NodeLoader` selects only its declared weights. `NodeExecutor` maps the node type to an operator. Dependencies are authoritative and the engine rejects unsatisfied dependencies.
