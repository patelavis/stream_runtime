import torch, tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from stream_runtime.storage import TensorStore
from stream_runtime.memory import MemoryManager
from stream_runtime.graph.graph import ModelGraph
from stream_runtime.planner import MemoryPlanner
from stream_runtime.runtime import StreamingEngine

p = sys.argv[1] if len(sys.argv) > 1 else "test_model.safetensors"
prepared = sys.argv[2] if len(sys.argv) > 2 else "prepared_model"
g = ModelGraph.load(os.path.join(prepared, "manifest.json"))
path = open(os.path.join(prepared, "model.path")).read().strip()
x = torch.randn(1, g.tensors[g.nodes[0].weights[0]]["shape"][1])
y = StreamingEngine(
    g, TensorStore(path), MemoryManager(1024 * 1024), MemoryPlanner(1024 * 1024).plan(g)
).run(x)
print(y)
