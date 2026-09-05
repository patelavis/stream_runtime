from collections import OrderedDict


class TensorChunkCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self._items = OrderedDict()
        self.bytes = 0

    def get(self, key):
        if key not in self._items:
            return None
        value, size = self._items.pop(key)
        self._items[key] = (value, size)
        return value

    def put(self, key, value):
        size = len(value)
        if size > self.max_bytes:
            return
        if key in self._items:
            self.bytes -= self._items.pop(key)[1]
        while self.bytes + size > self.max_bytes and self._items:
            _, (_, old) = self._items.popitem(last=False)
            self.bytes -= old
        self._items[key] = (value, size)
        self.bytes += size

    def clear(self):
        self._items.clear()
        self.bytes = 0
