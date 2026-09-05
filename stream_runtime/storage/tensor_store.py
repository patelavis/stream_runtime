from .safetensors_reader import SafeTensorStream
from .cache import TensorChunkCache
class TensorStore:
    def __init__(self, path, cache=None): self.reader=SafeTensorStream(path); self.cache=cache
    def metadata(self, name): return self.reader.metadata(name)
    def read(self, name, offset=0, length=None):
        key=(name,offset,length)
        if self.cache:
            hit=self.cache.get(key)
            if hit is not None: return hit
        out=self.reader.get_tensor(name).read_chunk(offset,length)
        if self.cache: self.cache.put(key,out)
        return out
    def tensor(self,name): return self.reader.get_tensor(name)
    @property
    def bytes_read(self): return self.reader.bytes_read
    @property
    def reads(self): return self.reader.reads
