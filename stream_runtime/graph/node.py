from dataclasses import dataclass, asdict
@dataclass
class Node:
    id:int; name:str; type:str; inputs:list; outputs:list; weights:list; estimated_memory:int=0; dependencies:list=None
    def __post_init__(self):
        if self.dependencies is None: self.dependencies=[] if self.id == 0 else [self.id-1]
    def to_dict(self): return asdict(self)
