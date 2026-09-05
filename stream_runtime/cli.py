import argparse, json, os, re
from .storage import SafeTensorStream, TensorStore
from .graph.graph import ModelGraph
from .architecture.registry import select_adapter
from .planner import MemoryPlanner
from .memory import MemoryManager
from .runtime import StreamingEngine

def parse_bytes(s):
    m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(K|KB|M|MB|G|GB)?\s*',str(s).upper())
    if not m: raise argparse.ArgumentTypeError('size must be raw bytes or K/M/G/KB/MB/GB')
    return int(float(m.group(1))*({'K':1024,'KB':1024,'M':2**20,'MB':2**20,'G':2**30,'GB':2**30}.get(m.group(2),1)))
def locate_model(path):
    if os.path.isdir(path):
        files=[os.path.join(path,x) for x in os.listdir(path) if x.endswith('.safetensors')]
        if not files: raise FileNotFoundError('no .safetensors file found')
        return files[0]
    return path
def prepare(args):
    model=locate_model(args.model); reader=SafeTensorStream(model); nodes,tensors=select_adapter().analyze(reader)
    for n in nodes: n.estimated_memory=sum(tensors[x]['bytes'] for x in n.weights)
    os.makedirs(args.output,exist_ok=True); ModelGraph('generic','generic',nodes,tensors).save(os.path.join(args.output,'manifest.json'))
    with open(os.path.join(args.output,'model.path'),'w') as f: f.write(os.path.abspath(model))
    print(f'Prepared {model} -> {args.output}/manifest.json ({len(nodes)} nodes)')
def inspect(args):
    r=SafeTensorStream(locate_model(args.model)); print(f'Model: {r.path}\nFile size: {r.file_size}\nTensor count: {len(r.tensor_names())}')
    total=0; largest=None
    for n in r.tensor_names():
        m=r.metadata(n); total+=m.nbytes
        if largest is None or m.nbytes>largest.nbytes: largest=m
        print(f'  {n}: dtype={m.dtype} shape={m.shape} bytes={m.nbytes} offset={m.data_start}')
    print(f'Total parameter bytes: {total}\nLargest tensor: {largest.name if largest else "none"}')
def load(args):
    g=ModelGraph.load(os.path.join(args.prepared,'manifest.json')); return g,locate_model(open(os.path.join(args.prepared,'model.path')).read().strip())
def plan(args):
    g,_=load(args); plans=MemoryPlanner(parse_bytes(args.ram_budget)).plan(g); print('========== MEMORY PLAN ==========')
    for n,p in zip(g.nodes,plans): print(f'Node {n.id}: {n.name}\n  strategy: {p.strategy}\n  working set: {p.working_set} bytes'+(f'\n  tile size: {p.tile_bytes} bytes' if p.tile_bytes else ''))
    print(f'Peak planned memory: {max((p.working_set for p in plans),default=0)} bytes\n==================================')
def run(args):
    g,path=load(args); budget=parse_bytes(args.ram_budget); mm=MemoryManager(budget); store=TensorStore(path); plans=MemoryPlanner(budget).plan(g)
    import torch
    first_weight=next((w for w in g.nodes[0].weights if w.endswith('.weight')), None)
    input_dim=g.tensors[first_weight]['shape'][1] if first_weight and len(g.tensors[first_weight]['shape']) > 1 else 1
    x=torch.load(args.input) if args.input else torch.zeros(1,input_dim)
    y=StreamingEngine(g,store,mm,plans).run(x)
    if args.output: torch.save(y,args.output)
    print(f'Status: SUCCESS\nOutput shape: {tuple(y.shape)}\nBytes read: {store.bytes_read}\nReads: {store.reads}\nPeak managed: {mm.stats.peak}\nWeights: {mm.stats.weights}')
def main():
    p=argparse.ArgumentParser(prog='stream-runtime'); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare'); a.add_argument('model'); a.add_argument('--output',required=True); a.set_defaults(func=prepare)
    a=sub.add_parser('inspect'); a.add_argument('model'); a.set_defaults(func=inspect)
    a=sub.add_parser('plan'); a.add_argument('prepared'); a.add_argument('--ram-budget',required=True); a.set_defaults(func=plan)
    a=sub.add_parser('run'); a.add_argument('prepared'); a.add_argument('--ram-budget',required=True); a.add_argument('--input'); a.add_argument('--output'); a.set_defaults(func=run)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
