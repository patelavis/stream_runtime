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
    json.dump({'model_path':os.path.abspath(model),'architecture':'generic','tensor_count':len(tensors),'parameter_bytes':sum(x['bytes'] for x in tensors.values()),'nodes':len(nodes)},open(os.path.join(args.output,'metadata.json'),'w'),indent=2)
    with open(os.path.join(args.output,'model.path'),'w') as f: f.write(os.path.abspath(model))
    print(f'Prepared {model} -> {args.output}/manifest.json ({len(nodes)} nodes)')
def inspect(args):
    r=SafeTensorStream(locate_model(args.model)); print(f'Model: {r.path}\nFile size: {r.file_size}\nArchitecture: generic\nTensor count: {len(r.tensor_names())}')
    total=0; largest=None
    for n in r.tensor_names():
        m=r.metadata(n); total+=m.nbytes
        if largest is None or m.nbytes>largest.nbytes: largest=m
        print(f'  {n}: dtype={m.dtype} shape={m.shape} bytes={m.nbytes} offset={m.data_start}')
    print(f'Total parameter bytes: {total}\nParameter count: {sum(__import__("math").prod(r.metadata(n).shape) for n in r.tensor_names())}\nLargest tensor: {largest.name if largest else "none"}')
def load(args):
    g=ModelGraph.load(os.path.join(args.prepared,'manifest.json')); return g,locate_model(open(os.path.join(args.prepared,'model.path')).read().strip())
def plan(args):
    g,_=load(args); plans=MemoryPlanner(parse_bytes(args.ram_budget)).plan(g); print('========== MEMORY PLAN ==========')
    for n,p in zip(g.nodes,plans): print(f'Node {n.id}: {n.name}\n  strategy: {p.strategy}\n  estimated memory: {p.working_set} bytes\n  tile size: {p.tile_bytes or 0} bytes\n  prefetch: no')
    print(f'Peak planned memory: {max((p.working_set for p in plans),default=0)} bytes\n==================================')
def run(args):
    g,path=load(args); budget=parse_bytes(args.ram_budget); mm=MemoryManager(budget); store=TensorStore(path); plans=MemoryPlanner(budget).plan(g)
    import torch
    first_weight=next((w for w in g.nodes[0].weights if w.endswith('.weight')), None)
    input_dim=g.tensors[first_weight]['shape'][1] if first_weight and len(g.tensors[first_weight]['shape']) > 1 else 1
    x=torch.load(args.input) if args.input else torch.zeros(1,input_dim)
    trace=(print if args.trace else None); y=StreamingEngine(g,store,mm,plans,trace=trace).run(x)
    if args.output: torch.save(y,args.output)
    print(f'Status: SUCCESS\nOutput shape: {tuple(y.shape)}\nNodes executed: {len(g.nodes)}\nBytes read: {store.bytes_read}\nReads: {store.reads}\nPeak managed: {mm.stats.peak}\nWeights: {mm.stats.weights}')
def test_cmd(args):
    import subprocess,sys,tempfile
    with tempfile.TemporaryDirectory() as d:
        model=os.path.join(d,'model.safetensors'); prepared=os.path.join(d,'prepared')
        subprocess.check_call([sys.executable,'examples/create_test_model.py','--output',model])
        prepare(argparse.Namespace(model=model,output=prepared)); plan(argparse.Namespace(prepared=prepared,ram_budget='1M')); run(argparse.Namespace(prepared=prepared,ram_budget='1M',input=None,output=None,trace=False))
def serve_cmd(args):
    from .server import run_server
    model=args.model or args.model_pos
    if not model: raise SystemExit('serve requires --model MODEL or a model directory')
    run_server(model,host=args.host,port=args.port,ram_budget=parse_bytes(args.ram_budget),api_key=args.api_key,no_auth=args.no_auth,request_timeout=args.request_timeout,cache_size=parse_bytes(args.cache_size) if args.cache_size else 0)
def models_cmd(args):
    import urllib.request
    print(urllib.request.urlopen(args.url.rstrip('/')+'/v1/models',timeout=5).read().decode())
def status_cmd(args):
    import urllib.request
    print(urllib.request.urlopen(args.url.rstrip('/')+'/v1/status',timeout=5).read().decode())
def main():
    p=argparse.ArgumentParser(prog='stream-runtime'); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare'); a.add_argument('model'); a.add_argument('--output',required=True); a.set_defaults(func=prepare)
    a=sub.add_parser('inspect'); a.add_argument('model'); a.set_defaults(func=inspect)
    a=sub.add_parser('plan'); a.add_argument('prepared'); a.add_argument('--ram-budget',required=True); a.set_defaults(func=plan)
    a=sub.add_parser('run'); a.add_argument('prepared'); a.add_argument('--ram-budget',required=True); a.add_argument('--input'); a.add_argument('--output'); a.add_argument('--trace',action='store_true'); a.set_defaults(func=run)
    a=sub.add_parser('test'); a.set_defaults(func=test_cmd)
    a=sub.add_parser('serve'); a.add_argument('model_pos',nargs='?'); a.add_argument('--model'); a.add_argument('--host',default='127.0.0.1'); a.add_argument('--port',type=int,default=8000); a.add_argument('--ram-budget',required=True); a.add_argument('--cache-size'); a.add_argument('--api-key',default='local'); a.add_argument('--no-auth',action='store_true'); a.add_argument('--request-timeout',type=float); a.set_defaults(func=serve_cmd)
    a=sub.add_parser('models'); a.add_argument('--url',default='http://127.0.0.1:8000'); a.set_defaults(func=models_cmd)
    a=sub.add_parser('status'); a.add_argument('--url',default='http://127.0.0.1:8000'); a.set_defaults(func=status_cmd)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
