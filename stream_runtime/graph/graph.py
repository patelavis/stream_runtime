import json
from .node import Node
class ModelGraph:
    def __init__(self, architecture='generic', model_type='unknown', nodes=None, tensors=None): self.architecture=architecture; self.model_type=model_type; self.nodes=nodes or []; self.tensors=tensors or {}
    def to_dict(self): return {'format_version':1,'architecture':self.architecture,'model_type':self.model_type,'execution_order':[n.id for n in self.nodes],'nodes':[n.to_dict() for n in self.nodes],'tensors':self.tensors}
    def save(self,path): json.dump(self.to_dict(),open(path,'w'),indent=2)
    @classmethod
    def load(cls,path):
        d=json.load(open(path)); return cls(d['architecture'],d['model_type'],[Node(**n) for n in d['nodes']],d['tensors'])
