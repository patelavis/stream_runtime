from abc import ABC, abstractmethod
from typing import Any
import torch
from .exceptions import MemoryBudgetExceeded

class BaseOperator(ABC):
    """Abstract base class for all model operators."""
    @abstractmethod
    def execute(self, input_tensor: torch.Tensor, tensor_store, memory_manager, node_plan) -> torch.Tensor:
        """Perform the computation and return the output tensor."""
        pass

class ReLU(BaseOperator):
    """Activation operator for non-linearity."""
    def execute(self, input_tensor: torch.Tensor, tensor_store, memory_manager, node_plan) -> torch.Tensor:
        # ReLU is an in-place or out-of-place operation that doesn't require weight loading
        # We still account for the activation output in the memory manager if necessary
        return torch.relu(input_tensor)
