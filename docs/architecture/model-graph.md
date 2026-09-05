# Model Graph

`ModelGraph` serializes format version, architecture, model type, execution order, nodes, and tensor metadata. `Node` contains id, name, type, inputs, outputs, weights, estimated memory, and dependencies. The generic adapter groups names by module prefix rather than by file size.
