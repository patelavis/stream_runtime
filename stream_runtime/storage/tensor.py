from dataclasses import dataclass
import numpy as np

_DTYPE = {"F16": np.float16, "F32": np.float32, "F64": np.float64, "BF16": np.uint16,
          "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
          "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_}

@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple
    data_start: int
    data_end: int
    @property
    def nbytes(self): return self.data_end - self.data_start
    @property
    def np_dtype(self):
        if self.dtype not in _DTYPE: raise ValueError(f"Unsupported safetensors dtype: {self.dtype}")
        return _DTYPE[self.dtype]

class StreamedTensor:
    def __init__(self, reader, metadata): self.reader, self.metadata = reader, metadata
    @property
    def name(self): return self.metadata.name
    @property
    def nbytes(self): return self.metadata.nbytes
    def read_chunk(self, offset=0, size=None):
        size = self.nbytes - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > self.nbytes: raise ValueError("chunk outside tensor")
        return self.reader.read_bytes(self.metadata.data_start + offset, size)
    def iter_chunks(self, chunk_size):
        for offset in range(0, self.nbytes, chunk_size): yield self.read_chunk(offset, min(chunk_size, self.nbytes-offset))
    def to_numpy(self):
        raw = self.read_chunk()
        return np.frombuffer(raw, dtype=self.metadata.np_dtype).reshape(self.metadata.shape).copy()
