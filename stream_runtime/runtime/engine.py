import torch
from ..operators.linear import StreamingLinear
class StreamingEngine:
    def __init__(self,graph,store,manager,plans): self.graph,self.store,self.mm,self.plans=graph,store,manager,plans
    def run(self,x):
        value=torch.as_tensor(x,dtype=torch.float32)
        for node,plan in zip(self.graph.nodes,self.plans):
            weights=[w for w in node.weights if w.endswith('.weight')]
            biases=[w for w in node.weights if w.endswith('.bias')]
            if not weights: continue
            meta=self.store.metadata(weights[0]); tile_rows=None
            if plan.strategy=='tiled':
                bytes_per_row=meta.nbytes//meta.shape[0]; tile_rows=max(1,plan.tile_bytes//max(1,bytes_per_row*meta.shape[1]))
            value=StreamingLinear(self.store,self.mm,weights[0],biases[0] if biases else None,tile_rows).run(value)
            if node.id < len(self.graph.nodes)-1: value=torch.relu(value)
        return value
