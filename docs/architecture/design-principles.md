# Design Principles

Correctness precedes performance. Model size may exceed RAM. Logical nodes are architecture boundaries; physical tiles are internal. The runtime preserves source dtype and weights by default. Unsupported unavoidable working sets fail clearly instead of silently changing the model.
