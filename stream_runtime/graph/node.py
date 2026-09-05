from dataclasses import dataclass, asdict
@dataclass
class Node:
    id:int; name:str; type:str; inputs:list; outputs:list; weights:list; estimated_memory:int=0
    def to_dict(self): return asdict(self)
