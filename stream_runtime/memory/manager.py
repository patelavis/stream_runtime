import torch
from typing import Optional, List
from .exceptions import MemoryBudgetExceeded

@dataclass
class NodePlan:
    node_id: int
    name: str
    strategy: Strategy
    working_set_bytes: int

@dataclass
class Strategy:
    type: str  # "direct" or "tiled"
    tile_size: Optional[int] = None
    estimated_memory: int = 0

class MemoryManager:
    """Manages the user-defined RAM budget for model and runtime operations."""
    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.stats = MemoryStats()
        self.peak_usage = 0

    def available(self) -> int:
        return self.budget - self.stats.total

    def request(self, size: int, category: str) -> bool:
        """Check if an allocation can fit within the budget."""
        if self.stats.total + size > self.budget:
            return False

        if category == 'weights':
            self.stats.weights_bytes += size
        elif category == 'activations':
            self.stats.activations_bytes += size
        elif category == 'temporary':
            self.stats.temporary_bytes += size
        elif category == 'cache':
            self.stats.cache_bytes += size
        elif category == 'prefetch':
            self.stats.prefetch_bytes += size
        else:
            raise ValueError(f"Unknown memory category: {category}")

        self.peak_usage = max(self.peak_usage, self.stats.total)
        return True

    def release(self, size: int, category: str):
        """Release a previously requested allocation."""
        if category == 'weights':
            self.stats.weights_bytes = max(0, self.stats.weights_bytes - size)
        elif category == 'activations':
            self.stats.activations_bytes = max(0, self.stats.activations_bytes - size)
        elif category == 'temporary':
            self.stats.temporary_bytes = max(0, self.stats.temporary_bytes - size)
        elif category == 'cache':
            self.stats.cache_bytes = max(0, self.stats.cache_bytes - size)
        elif category == 'prefetch':
            self.stats.prefetch_bytes = max(0, self.stats.prefetch_bytes - size)

    def report(self) -> dict:
        """Returns the current memory usage statistics."""
        return {
            "budget": self.budget,
            "current": self.stats.total,
            "peak": self.peak_usage,
            "available": self.available(),
            "weights": self.stats.weights_bytes,
            "activations": self.stats.activations_bytes,
            "temporary": self.stats.temporary_bytes,
            "cache": self.stats.cache_bytes,
            "prefetch": self.stats.prefetch_bytes,
        }

class AllocationContext:
    def __init__(self, manager: MemoryManager, size: int, category: str):
        self.manager = manager
        self.size = size
        self.category = category

    def __enter__(self):
        if not self.manager.request(self.size, self.category):
            raise MemoryBudgetExceeded(f"Requested {self.size} bytes for {self.category}, "
                                      f"but only {self.manager.available()} available.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.release(self.size, self.category)

@dataclass
class MemoryStats:
    """Current tracked memory usage."""
    weights_bytes: int = 0
    activations_bytes: int = 0
    temporary_bytes: int = 0
    cache_bytes: int = 0
    prefetch_bytes: int = 0

    @property
    def total(self) -> int:
        return (self.weights_bytes + self.activations_bytes +
                self.temporary_bytes + self.cache_bytes + self.prefetch_bytes)
