import json, os, struct
from .tensor import TensorMetadata, StreamedTensor

class SafeTensorStream:
    def __init__(self, path):
        self.path = os.fspath(path); self.file_size = os.path.getsize(self.path)
        with open(self.path, 'rb') as f:
            b = f.read(8)
            if len(b) != 8: raise ValueError('truncated safetensors header')
            self.header_size = struct.unpack('<Q', b)[0]
            if self.header_size > self.file_size - 8 or self.header_size > 128*1024*1024: raise ValueError('invalid header size')
            header = json.loads(f.read(self.header_size))
        self.data_offset = 8 + self.header_size; self._tensors = {}
        for name, item in header.items():
            if name == '__metadata__': continue
            start, end = item['data_offsets']
            if start < 0 or end < start or self.data_offset + end > self.file_size: raise ValueError(f'invalid offsets for {name}')
            self._tensors[name] = TensorMetadata(name, item['dtype'], tuple(item['shape']), self.data_offset+start, self.data_offset+end)
        self.bytes_read = 0; self.reads = 0
    def tensor_names(self): return list(self._tensors)
    def metadata(self, name): return self._tensors[name]
    def get_tensor(self, name): return StreamedTensor(self, self.metadata(name))
    def read_bytes(self, offset, length):
        with open(self.path, 'rb') as f:
            f.seek(offset); data = f.read(length)
        if len(data) != length: raise IOError('short read')
        self.bytes_read += length; self.reads += 1; return data
