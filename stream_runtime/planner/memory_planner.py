import torch
from typing import List, Optional
from dataclasses import dataclass
from .exceptions import MemoryBudgetExceeded
from .tensor import StreamTensor

@dataclass
class NodePlan:
    node_id: int
    name: str
    strategy: Strategy
    working_set_bytes: int

@dataclass
class Strategy:
    type: str  # "direct" or "tiled"
    tile_size: Optional[int] = None
    estimated_memory: int = 0

class MemoryPlanner:
    """Calculates the execution strategy for each node in a graph based on RAM budget."""
    def __init__(self, ram_budget: int):
        self.ram_budget = ram_budget
        # Reserved overheads (runtime constants) - could be adjusted via config
        self.overhead_bytes = 150 * 1024 * 1024  # 150MB default for system/buffers

    def estimate_node_memory(self, node, tensor_store) -> int:
        """Estimates the total managed memory required for a node's execution."""
        weights_bytes = sum(tensor_store.get_tensor(name).descriptor.nbytes for name in node.weights)
        # Heuristic: Activation and workspace overhead is ~30% of weights size.
        # This will be replaced with architecture-specific formulas later.
        estim_activation = int(weights_bytes * 0.3)
        return weights_bytes + estim_activation

    def plan(self, graph, tensor_store) -> List[NodePlan]:
        plans = []
        available_for_nodes = self.ram_budget - self.overhead_bytes

        if available_for_nodes < 0:
            raise MemoryBudgetExceeded(f"RAM budget {self.ram_budget} is too small to accommodate runtime overhead.")

        # We assume the execution order is stored in graph.order
        for nid in graph.order:
            node = graph.get_node(nid)
            req_mem = self.estimate_node_memory(node, tensor_store)

            if req_mem <= available_for_nodes:
                plans.append(NodePlan(
                    node_id=node.id,
                    name=node.name,
                    strategy=Strategy(type="direct", estimated_memory=req_mem),
                    working_set_bytes=req_mem
                ))
            else:
                # Node is too big, must use tiling.
                weights_bytes = sum(tensor_store.get_tensor(name).descriptor.nbytes for name in node.weights)

                # Target 70% of available memory for the weights chunk to allow room for activations/temp
                target_weight_chunk = int(available_for_nodes * 0.7)

                if target_weight_chunk < (weights_bytes * 0.05):
                    raise MemoryBudgetExceeded(f"Node {node.name} requires a minimum working set larger than the budget.")

                # Calculate tile size based on available space
                tile_size = target_weight_chunk
                plans.append(NodePlan(
                    node_id=node.id,
                    name=node.name,
                    strategy=Strategy(type="tiled", tile_size=tile_size, estimated_memory=req_mem),
                    working_set_bytes=target_weight_chunk + int(target_weight_chunk * 0.3)
                ))

        return plans
