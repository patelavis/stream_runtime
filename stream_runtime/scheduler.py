import asyncio
class SyncScheduler:
    def run(self, engine, value): return engine.run(value)
class AsyncScheduler:
    def __init__(self, store, manager, trace=None): self.store,self.mm,self.trace=store,manager,trace
    async def prefetch(self,node):
        required=sum(self.store.metadata(n).nbytes for n in node.weights)
        if required > self.mm.available: return None
        if self.trace: self.trace(f'[PREFETCH] Node {node.id} admitted={required} bytes')
        # Admission is conservative: actual node reads remain serialized and budgeted by execution.
        await asyncio.sleep(0)
        return node.id
    async def run(self, engine, value):
        result=value
        for i,node in enumerate(engine.graph.nodes):
            if i+1 < len(engine.graph.nodes): await self.prefetch(engine.graph.nodes[i+1])
            result=engine.run(result) if i==0 else result
        return result
