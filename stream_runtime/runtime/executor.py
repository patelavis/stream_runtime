import torch
from ..operators.linear import StreamingLinear
class NodeExecutor:
    def __init__(self, store, manager, trace=None): self.store=store; self.mm=manager; self.trace=trace
    def execute(self,node,x,loaded,plan):
        if self.trace: self.trace(f'[EXEC] Node {node.id} {node.type} strategy={plan.strategy}')
        if node.type == 'ReLU': return torch.relu(x)
        if node.type == 'Add': return x + x
        if node.type == 'Embedding':
            name=next(n for n in node.weights if n.endswith('.weight')); w=self.store.tensor(name).to_numpy(); return torch.nn.functional.embedding(x.long(),torch.from_numpy(w))
        name=next((n for n in node.weights if n.endswith('.weight')),None)
        if name is None: return x
        bias=next((n for n in node.weights if n.endswith('.bias')),None)
        meta=self.store.metadata(name); tile_rows=None
        if plan.strategy=='tiled':
            bytes_per_row=max(1,meta.nbytes//meta.shape[0])
            tile_rows=max(1,plan.tile_bytes//bytes_per_row)
            tile_rows=min(meta.shape[0],tile_rows)
        return StreamingLinear(self.store,self.mm,name,bias,tile_rows).run(x)
