import torch
from typing import Dict, List, Optional
from .safetensors_reader import SafeTensorStream
from .tensor import TensorDescriptor, StreamTensor
from .exceptions import StorageError, TensorReadError

class TensorStore:
    """Abstracts the storage of tensors and manages access to them."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        # The streamer handles the low-level range reads from .safetensors
        self._streamer = SafeTensorStream(model_path)

        # Cache for descriptors ONLY (never full tensors).
        # Descriptors are tiny, so we can keep them in RAM.
        self._descriptors: Dict[str, TensorDescriptor] = {
            name: self._streamer.get_tensor(name).descriptor
            for name in self._streamer.tensor_names()
        }

    def get_tensor(self, name: str) -> StreamTensor:
        """Returns a handle to a specific tensor for ranged reading."""
        if name not in self._descriptors:
            raise KeyError(f"Tensor '{name}' not found in {self.model_path}")
        return StreamTensor(self._descriptors[name], self._streamer)

    def get_metadata(self) -> dict:
        """Returns the safetensors metadata."""
        return self._streamer.metadata()

    def list_tensors(self) -> List[str]:
        """Lists all available tensor names in the file."""
        return list(self._descriptors.keys())

    def get_descriptor(self, name: str) -> TensorDescriptor:
        if name not in self._descriptors:
            raise KeyError(f"Tensor '{name}' not found.")
        return self._descriptors[name]
