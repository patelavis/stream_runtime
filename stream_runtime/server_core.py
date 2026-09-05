"""Offline local serving primitives; no network calls are made by this module."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import asyncio, json, os, uuid
from .graph.graph import ModelGraph
from .memory import MemoryManager
from .planner import MemoryPlanner
from .storage import TensorStore
from .runtime import StreamingEngine


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"


@dataclass
class LocalModel:
    model_id: str
    directory: Path
    ram_budget: int
    cache_size: int = 0

    def __post_init__(self):
        self.state = ModelState.UNLOADED
        self.graph = None
        self.model_path = None
        self.mm = None
        self.store = None
        self.plans = None
        self.nodes_executed = 0
        self.requests = 0
        self.last_error = None

    def initialize(self):
        self.state = ModelState.INITIALIZING
        manifest = self.directory / "manifest.json"
        path_file = self.directory / "model.path"
        if not manifest.exists() or not path_file.exists():
            raise FileNotFoundError(
                "prepared model requires manifest.json and model.path"
            )
        self.graph = ModelGraph.load(manifest)
        self.model_path = Path(path_file.read_text().strip())
        if not self.model_path.exists():
            raise FileNotFoundError(f"model file missing: {self.model_path}")
        self.mm = MemoryManager(self.ram_budget)
        self.store = TensorStore(self.model_path)
        self.plans = MemoryPlanner(self.ram_budget).plan(self.graph)
        self.state = ModelState.READY

    def vectorize(self, text):
        import torch

        first = next(
            (w for w in self.graph.nodes[0].weights if w.endswith(".weight")), None
        )
        dim = (
            self.graph.tensors[first]["shape"][1]
            if first and len(self.graph.tensors[first]["shape"]) > 1
            else 1
        )
        values = [(sum(ord(c) for c in text[i::dim]) % 997) / 997.0 for i in range(dim)]
        return torch.tensor([values], dtype=torch.float32)

    def infer(self, text):
        import torch

        if self.state == ModelState.UNLOADED:
            self.initialize()
        self.state = ModelState.RUNNING
        self.requests += 1
        try:
            out = StreamingEngine(self.graph, self.store, self.mm, self.plans).run(
                self.vectorize(text)
            )
            self.nodes_executed += len(self.graph.nodes)
            token = int(torch.argmax(out, dim=-1).reshape(-1)[-1]) if out.numel() else 0
            return f"Local model response (token {token}) for: {text}"
        finally:
            self.state = ModelState.READY

    def status(self):
        return {
            "model": self.model_id,
            "state": self.state.value,
            "ram_budget": self.ram_budget,
            "managed_ram": self.mm.used() if self.mm else 0,
            "peak_ram": self.mm.peak() if self.mm else 0,
            "nodes_executed": self.nodes_executed,
            "disk_bytes_read": self.store.bytes_read if self.store else 0,
            "model_on_disk": bool(self.model_path and self.model_path.exists()),
        }


class ModelRegistry:
    def __init__(self, roots):
        self.roots = [Path(x) for x in roots]

    def discover(self):
        found = {}
        for root in self.roots:
            if not root.exists():
                continue
            candidates = [root] + [x for x in root.iterdir() if x.is_dir()]
            for d in candidates:
                try:
                    if (d / "manifest.json").exists() and (d / "model.path").exists():
                        found[d.name] = d
                except OSError:
                    continue
        return found

    def get(self, model_id):
        models = self.discover()
        if model_id not in models:
            raise KeyError(f"local model not found: {model_id}")
        return models[model_id]


class RequestManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.active = False
        self.queue_length = 0
        self.active_request = None

    async def run(self, request_id, fn):
        self.queue_length += 1
        async with self.lock:
            self.queue_length -= 1
            self.active = True
            self.active_request = request_id
            try:
                return await fn()
            finally:
                self.active = False
                self.active_request = None
