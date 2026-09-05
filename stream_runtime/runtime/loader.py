class NodeLoader:
    """Loads only the ranges required by one graph node."""

    def __init__(self, store, memory_manager, trace=None):
        self.store = store
        self.mm = memory_manager
        self.trace = trace

    def load(self, node):
        loaded = {}
        for name in node.weights:
            if self.trace:
                self.trace(f"[DISK] read Node {node.id} tensor={name}")
            loaded[name] = self.store.tensor(name)
        return loaded

    def release(self, node, loaded):
        loaded.clear()
        if self.trace:
            self.trace(f"[RAM] release Node {node.id}")
