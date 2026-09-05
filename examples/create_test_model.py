import argparse, os, torch
from safetensors.torch import save_file
p=argparse.ArgumentParser(); p.add_argument('--output',default='test_model.safetensors'); p.add_argument('--in-dim',type=int,default=16); p.add_argument('--hidden',type=int,default=32); p.add_argument('--out-dim',type=int,default=8); p.add_argument('--seed',type=int,default=0); a=p.parse_args(); torch.manual_seed(a.seed)
state={'linear1.weight':torch.randn(a.hidden,a.in_dim),'linear1.bias':torch.randn(a.hidden),'linear2.weight':torch.randn(a.out_dim,a.hidden),'linear2.bias':torch.randn(a.out_dim)}
os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True); save_file(state,a.output); print(a.output)
