import torch
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from .exceptions import ModelNotFoundError

@dataclass
class ModelNode:
    """A single logical unit of computation in a model graph."""
    id: int
    name: str
    node_type: str  # e.g., "Embedding", "TransformerBlock", "Linear"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    weights: List[str] = field(default_factory=list) # Names of tensors in the safetensors file
    dependencies: List[int] = field(default_factory=list) # IDs of nodes that must complete before this one

    def __repr__(self):
        return f"Node({self.id}: {self.name} [{self.node_type}])"

class ExecutionGraph:
    """Represents the ordered graph of execution for a model."""
    def __init__(self):
        self.nodes: Dict[int, ModelNode] = {}
        self.order: List[int] = [] # The deterministic order of execution

    def add_node(self, node: ModelNode):
        self.nodes[node.id] = node
        if node.id not in self.order:
            self.order.append(node.id)

    def get_node(self, node_id: int) -> ModelNode:
        return self.nodes.get(node_id)

    def get_execution_path(self) -> List[ModelNode]:
        """Returns the nodes in their planned execution order."""
        return [self.nodes[nid] for nid in self.order]
