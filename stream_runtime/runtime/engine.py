import torch
from .loader import NodeLoader
from .executor import NodeExecutor
class StreamingEngine:
    def __init__(self,graph,store,manager,plans,trace=None):
        self.graph,self.store,self.mm,self.plans=graph,store,manager,plans; self.trace=trace
        self.loader=NodeLoader(store,manager,trace); self.executor=NodeExecutor(store,manager,trace)
    def run(self,x):
        value=torch.as_tensor(x,dtype=torch.float32)
        completed=set()
        for node,plan in zip(self.graph.nodes,self.plans):
            if any(dep not in completed for dep in node.dependencies): raise RuntimeError(f'unsatisfied dependencies for node {node.id}')
            loaded=self.loader.load(node)
            value=self.executor.execute(node,value,loaded,plan)
            self.loader.release(node,loaded); completed.add(node.id)
            if node.id < len(self.graph.nodes)-1: value=torch.relu(value)
        return value
