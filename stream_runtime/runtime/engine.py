import torch
from typing import List, Dict, Any, Optional
from .graph.node import ModelNode, ExecutionGraph
from .storage.tensor_store import TensorStore
from .memory.manager import MemoryManager
from .planner.memory_planner import MemoryPlanner
from .activation.store import ActivationStore
from .operators.linear import Linear
from .operators.base import BaseOperator
from .exceptions import ModelNotFoundError, MemoryBudgetExceeded

class RuntimeEngine:
    """The core execution engine for disk-streaming model inference."""
    def __init__(self, manifest_path: str, ram_budget_bytes: int):
        # Load Manifest (Simplified for prototype)
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        self.nodes = [
            ModelNode(**n) for n in data["nodes"]
        ]
        self.graph = ExecutionGraph()
        for node in self.nodes:
            self.graph.add_node(node)

        # Initialize components
        self.tensor_store = TensorStore(data["model_path"])
        self.memory_manager = MemoryManager(ram_budget_bytes)
        self.planner = MemoryPlanner(ram_budget_bytes)
        self.activation_store = ActivationStore(self.memory_manager)

        # Create the memory plan
        self.plans = self.planner.plan(self.graph, self.tensor_store)

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Executes all nodes in the graph sequentially."""
        current_input = input_tensor

        for i, plan in enumerate(self.plans):
            node = self.graph.get_node(plan.node_id)
            print(f"[EXEC] Node {i}: {node.name} ({plan.strategy.type})")

            # 1. Determine operator (Prototype only supports Linear/ReLU)
            if node.node_type == "Linear":
                op = Linear(node.weights)
            elif node.node_type == "ReLU":
                op = ReLU()
            else:
                raise NotImplementedError(f"No operator for {node.node_type}")

            # 2. Execute the operation within a memory-managed context
            # The MemoryManager will enforce our budget here.
            with self.memory_manager.allocation_context(
                plan.working_set_bytes, category='weights'
            ):
                current_input = op.execute(
                    current_input,
                    self.tensor_store,
                    self.memory_manager,
                    plan
                )

            # 3. Store the output activation for use by the next node
            self.activation_store.store(node.id, current_input)

            # Note: In a multi-node chain where Node N+1 only needs Node N's output,
            # we should ideally release Node N-1's output here.
            if i > 0 and i < len(self.plans) - 1:
                prev_node_id = self.plans[i-1].node_id
                # This is simplified; real logic handles dependencies correctly.
                self.activation_store.release(prev_node_id)

        return current_input

import json
