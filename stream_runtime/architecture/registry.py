from .generic import ArchitectureAdapter
ARCHITECTURE_ADAPTERS=[ArchitectureAdapter()]
def select_adapter(config=None): return ARCHITECTURE_ADAPTERS[0]
