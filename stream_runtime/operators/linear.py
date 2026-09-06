import torch
from typing import Optional, List
from .base import BaseOperator
from .exceptions import MemoryBudgetExceeded

class Linear(BaseOperator):
    """Implementation of the Linear/MatMul operator with support for streaming weights."""
    def __init__(self, weight_names: List[str], bias_name: Optional[str] = None):
        self.weight_names = weight_names
        self.bias_name = bias_name

    def execute(self, input_tensor: torch.Tensor, tensor_store, memory_manager, node_plan) -> torch.Tensor:
        """Executes the linear operation (Y = XW + b)."""
        # 1. Get Input activation stats
        input_shape = input_tensor.shape
        out_features = next(iter(self.weight_names))[0].split('.')[-1] # Simplified name parsing
        # In a real implementation, we'd get shape from the descriptor of the weight tensor.
        # For this prototype, let's assume shape is (in_features, out_features)
        # and input is (batch, in_features).

        weight_descriptor = tensor_store.get_descriptor(self.weight_names[0])
        out_features_count = weight_descriptor.shape[0] # Assuming [Out, In] format
        in_features_count = weight_descriptor.shape[1]

        # Output buffer allocation (managed)
        # We'll assume batch size 1 for the prototype to simplify output shape logic.
        output_shape = (input_tensor.shape[0], out_features_count)
        output = torch.zeros(output_shape, dtype=input_tensor.dtype)

        if node_plan.strategy.type == "direct":
            # Load all weights and compute
            with memory_manager.allocation_context(sum(t.descriptor.nbytes for t in [tensor_store.get_tensor(n) for n in self.weight_names]), category='weights'):
                # Read full weight tensor
                weight = tensor_store.get_tensor(self.weight_names[0]).read()
                output = input_tensor @ weight.T
        else:
            # Tiled implementation: Y = sum(X @ W_tile)
            # Logic for tiling over the 'In' dimension of weights
            batch, in_dim = input_tensor.shape

            # Determine tile size from node plan
            tile_size = node_plan.strategy.tile_size

            # Initialize output buffer with zeros and sum into it
            for i in range(0, weight_descriptor.shape[1], tile_size):
                end = min(i + tile_size, weight_descriptor.shape[1])
                current_tile_size = end - i

                # Load current chunk of weights (representing a slice of the input features)
                # Note: this requires careful calculation of byte offsets into the file
                # For simplicity in this prototype version, we'll simulate the chunking.
                # In the final implementation, we use descriptor.file_offset + calc_chunk_offset
                weight_tile = self._read_weight_slice(tensor_store, self.weight_names[0], i, current_tile_size)

                # Partial matrix multiplication: (Batch, Tile_In) @ (Tile_In, Out) -> (Batch, Out)
                partial = input_tensor[:, :current_tile_size] @ weight_tile.T
                output += partial

        if self.bias_name:
            bias = tensor_store.get_tensor(self.bias_name).read()
            output += bias.unsqueeze(0)

        return output

    def _read_weight_slice(self, tensor_store, name: str, start_idx: int, size: int) -> torch.Tensor:
        """Helper to read a specific slice of the weight matrix from disk."""
        desc = tensor_store.get_descriptor(name)
        # For [Out, In] weights, we need to extract columns starting at start_idx
        # This is tricky with raw range reads if it's not contiguous in memory/disk.
        # If the file format allows for strided reads or block-wise storage, this works.
        # For a proof-of-concept, let's assume the weight matrix is stored row-major
        # and we might need to load multiple rows or use clever indexing.

        # SIMPLEST PROTOTYPE VERSION: Read full weights if it's a small model.
        # TRUE STREAMING VERSION: Compute required chunks from the descriptor offsets.
        return tensor_store.get_tensor(name).read()
