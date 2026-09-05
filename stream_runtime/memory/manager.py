from dataclasses import dataclass
from contextlib import contextmanager
from ..exceptions import MemoryBudgetExceeded


@dataclass
class MemoryStats:
    current: int = 0
    peak: int = 0
    weights: int = 0
    activations: int = 0
    cache: int = 0
    temporary: int = 0


class MemoryManager:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.stats = MemoryStats()

    @property
    def available(self):
        return self.budget_bytes - self.stats.current

    def used(self):
        return self.stats.current

    def available_bytes(self):
        return self.available

    def peak(self):
        return self.stats.peak

    def reserve(self, nbytes, category="temporary"):
        if nbytes < 0 or nbytes > self.available:
            raise MemoryBudgetExceeded(
                nbytes, max(0, self.available), self.budget_bytes, category
            )
        self.stats.current += nbytes
        self.stats.peak = max(self.stats.peak, self.stats.current)
        if hasattr(self.stats, category):
            setattr(self.stats, category, getattr(self.stats, category) + nbytes)
        return Reservation(self, nbytes, category)

    def release(self, nbytes, category="temporary"):
        self.stats.current -= nbytes
        if self.stats.current < 0:
            raise RuntimeError("memory accounting underflow")
        if hasattr(self.stats, category):
            setattr(
                self.stats, category, max(0, getattr(self.stats, category) - nbytes)
            )

    @contextmanager
    def allocation(self, nbytes, category="temporary"):
        r = self.reserve(nbytes, category)
        try:
            yield r
        finally:
            r.release()

    def report(self):
        return self.stats


class Reservation:
    def __init__(self, manager, nbytes, category):
        self.manager = manager
        self.nbytes = nbytes
        self.category = category
        self.active = True

    def release(self):
        if self.active:
            self.manager.release(self.nbytes, self.category)
            self.active = False
