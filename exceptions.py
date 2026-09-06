import os
from typing import List, Tuple, Optional

class MemoryBudgetExceeded(Exception):
    """Raised when a requested memory allocation exceeds the user-defined budget."""
    pass

class ModelNotFoundError(Exception):
    """Raised when a specified model or manifest cannot be found."""
    pass

class UnsupportedArchitectureError(Exception):
    """Raised when the runtime does not support the provided model architecture."""
    pass

class StorageError(Exception):
    """Base class for storage-related errors."""
    pass

class TensorReadError(StorageError):
    """Raised when a specific tensor range cannot be read from disk."""
    pass
