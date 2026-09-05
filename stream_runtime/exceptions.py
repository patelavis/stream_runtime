class StreamRuntimeError(Exception):
    """Base runtime error."""

class MemoryBudgetExceeded(StreamRuntimeError):
    def __init__(self, requested, available, budget, category="unknown"):
        super().__init__(f"{category} allocation of {requested} bytes exceeds available {available} bytes (budget {budget})")
        self.requested, self.available, self.budget, self.category = requested, available, budget, category

class UnsupportedOperation(StreamRuntimeError):
    pass
