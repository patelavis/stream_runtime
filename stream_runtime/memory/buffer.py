class ManagedBuffer:
    def __init__(self, manager, data, category='temporary'):
        self.manager=manager; self.data=data; self.reservation=manager.reserve(len(data),category)
    def release(self):
        if self.reservation: self.reservation.release(); self.reservation=None; self.data=None
    def __enter__(self): return self.data
    def __exit__(self,*args): self.release()
