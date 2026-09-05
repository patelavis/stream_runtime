import numpy as np, torch
class StreamingLinear:
    def __init__(self,store,manager,weight_name,bias_name=None,tile_rows=None): self.store=store; self.mm=manager; self.weight=store.metadata(weight_name); self.bias_name=bias_name; self.tile_rows=tile_rows
    def run(self,x):
        x=torch.as_tensor(x,dtype=torch.float32); out_rows=self.weight.shape[0]; in_dim=self.weight.shape[1]; tile=self.tile_rows or out_rows
        out=torch.empty((*x.shape[:-1],out_rows),dtype=torch.float32)
        for start in range(0,out_rows,tile):
            rows=min(tile,out_rows-start); raw=self.store.read(self.weight.name,start*in_dim*self.weight.nbytes//(self.weight.shape[0]*self.weight.shape[1]),rows*in_dim*self.weight.nbytes//(self.weight.shape[0]*self.weight.shape[1]))
            with self.mm.allocation(len(raw),'weights'):
                w=torch.from_numpy(np.frombuffer(raw,dtype=self.weight.np_dtype).reshape(rows,in_dim).copy()).float()
                part=torch.matmul(x,w.T)
            if self.bias_name:
                bmeta=self.store.metadata(self.bias_name); braw=self.store.read(self.bias_name,start*bmeta.nbytes//bmeta.shape[0],rows*bmeta.nbytes//bmeta.shape[0])
                b=torch.from_numpy(np.frombuffer(braw,dtype=bmeta.np_dtype).copy()).float(); part=part+b
            out[...,start:start+rows]=part
        return out
