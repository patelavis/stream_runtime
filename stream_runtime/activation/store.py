class ActivationStore:
    def write(self,key,value): raise NotImplementedError
    def read(self,key): raise NotImplementedError
    def release(self,key): raise NotImplementedError
class RamActivationStore(ActivationStore):
    def __init__(self,manager): self.manager=manager; self.items={}
    def write(self,key,value):
        n=value.numel()*value.element_size(); r=self.manager.reserve(n,'activations'); self.items[key]=(value,r); return value
    def read(self,key): return self.items[key][0]
    def release(self,key):
        item=self.items.pop(key,None)
        if item: item[1].release()
