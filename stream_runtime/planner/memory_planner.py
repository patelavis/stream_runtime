from dataclasses import dataclass


@dataclass
class NodePlan:
    node_id: int
    strategy: str
    working_set: int
    tile_bytes: int = 0


class MemoryPlanner:
    def __init__(self, budget, safety=4096):
        self.budget = budget
        self.safety = safety

    def plan(self, graph):
        out = []
        for n in graph.nodes:
            weights = sum(graph.tensors[x]["bytes"] for x in n.weights)
            estimate = weights + self.safety
            if estimate <= self.budget:
                out.append(NodePlan(n.id, "direct", estimate, 0))
            else:
                tile = max(1, self.budget - self.safety)
                out.append(NodePlan(n.id, "tiled", self.budget, tile))
        return out
