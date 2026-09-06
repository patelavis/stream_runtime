import torch
from typing import Dict, List, Optional
from .exceptions import MemoryBudgetExceeded

class ActivationStore:
    """Manages intermediate activation tensors between model nodes."""
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        # Map node ID -> activation tensor
        self._activations: Dict[int, torch.Tensor] = {}

    def store(self, node_id: int, tensor: torch.Tensor) -> int:
        """Stores an output activation and returns its size in bytes."""
        size = tensor.nelement() * tensor.element_size()
        # This should already be managed by the caller who requested it,
        # but we can double-check here or manage a global registry.
        self._activations[node_id] = tensor
        return size

    def get(self, node_id: int) -> torch.Tensor:
        """Retrieves an output activation for the next node."""
        if node_id not in self._activations:
            raise KeyError(f"No activation found for node {node_id}")
        return self._activations[node_id]

    def release(self, node_id: int):
        """Releases the memory of a specific output activation."""
        if node_id in self._activations:
            tensor = self._activations.pop(node_id)
            # MemoryManager should have already updated its internal count
            # during the 'request' phase of the operator that produced it.
            # We just clear our reference to allow GC.
            pass

    def clear_all(self):
        """Clears all activations (useful for switching requests)."""
        self._activations.clear()
