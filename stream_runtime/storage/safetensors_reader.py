import os
import struct
import json
import torch
from typing import Dict, List, Optional
from .exceptions import TensorReadError
from .tensor import TensorDescriptor, StreamTensor

class SafeTensorStream:
    """
    A streaming reader for .safetensors files.
    Reads only the header and provides range-based access to tensors.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        # Open the file for reading using a raw file descriptor to support pread
        self._fd = os.open(file_path, os.O_RDONLY)
        self._tensors: Dict[str, TensorDescriptor] = {}
        self._metadata: Optional[dict] = None
        self._parse_header()

    def _parse_header(self):
        """Parses the .safetensors header."""
        try:
            # The header starts with 8 bytes for the length of the JSON string,
            # followed by the JSON itself.
            header_len_bytes = os.pread(self._fd, 8)
            if not header_len_bytes:
                raise TensorReadError("Empty file or missing header length.")

            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_json = os.pread(self._fd, header_len).decode('utf-8')
            self._metadata = json.loads(header_json)

            # Extract tensor metadata
            for name, info in self._metadata['data'].items():
                dtype_str = info['dtype']
                shape = tuple(info['shape'])
                offset = info['data_offset']
                length = info['data_len']

                # Map safetensors dtype strings to torch dtypes
                dtype_map = {
                    "F32": torch.float32,
                    "F16": torch.float16,
                    "BF16": torch.bfloat16,
                    "I64": torch.int64,
                    "U8": torch.uint8,
                }
                dtype = dtype_map.get(dtype_str, torch.float32)

                self._tensors[name] = TensorDescriptor(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    nbytes=length,
                    file_offset=offset,
                    file_length=length
                )
        except Exception as e:
            raise TensorReadError(f"Failed to parse safetensors header: {e}")

    def tensor_names(self) -> List[str]:
        """Returns a list of all tensor names in the file."""
        return list(self._tensors.keys())

    def get_tensor(self, name: str) -> StreamTensor:
        """Returns a handle to a specific tensor descriptor."""
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found in model.")
        return StreamTensor(self._tensors[name], self)

    def metadata(self) -> dict:
        """Returns the raw JSON metadata from the header."""
        return self._metadata if self._metadata else {}

    def pread(self, offset: int, size: int) -> bytes:
        """Wrapper around os.pread for reading specific ranges."""
        return os.pread(self._fd, size, offset)

    def __del__(self):
        if hasattr(self, '_fd') and self._fd is not None:
            os.close(self._fd)
