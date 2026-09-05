import os, asyncio, torch, pytest
from safetensors.torch import save_file
from stream_runtime.memory import MemoryManager
from stream_runtime.exceptions import MemoryBudgetExceeded
from stream_runtime.storage import SafeTensorStream, TensorChunkCache, TensorStore
from stream_runtime.graph.graph import ModelGraph
from stream_runtime.planner import MemoryPlanner
from stream_runtime.runtime import StreamingEngine

def model(tmp_path):
 p=str(tmp_path/'m.safetensors'); state={'linear1.weight':torch.arange(32,dtype=torch.float32).reshape(8,4),'linear1.bias':torch.zeros(8),'linear2.weight':torch.ones(2,8),'linear2.bias':torch.zeros(2)}; save_file(state,p); return p,state

def test_memory_hard_limit():
 mm=MemoryManager(1024); r=mm.reserve(512); assert mm.available==512
 with pytest.raises(MemoryBudgetExceeded): mm.reserve(600)
 r.release(); assert mm.stats.current==0

def test_cache_bound():
 c=TensorChunkCache(10); c.put('a',b'123456'); c.put('b',b'123456'); assert c.bytes<=10

def test_stream_ranges(tmp_path):
    p,state=model(tmp_path); r=SafeTensorStream(p); t=r.get_tensor('linear1.weight'); assert t.metadata.shape==(8,4); assert len(t.read_chunk(0,16))==16; assert r.bytes_read==16
    assert t.shape==(8,4) and t.file_offset == t.offset and t.file_length == t.nbytes

def test_async_range_read(tmp_path):
    from stream_runtime.storage import AsyncTensorStore
    p,state=model(tmp_path); a=AsyncTensorStore(TensorStore(p))
    data=asyncio.run(a.read('linear1.weight',0,16)); a.close()
    assert len(data)==16

def test_tiled_end_to_end(tmp_path):
 p,state=model(tmp_path); r=SafeTensorStream(p); nodes,tensors=__import__('stream_runtime.architecture.generic',fromlist=['ArchitectureAdapter']).ArchitectureAdapter().analyze(r); g=ModelGraph('generic','generic',nodes,tensors); mm=MemoryManager(256); store=TensorStore(p); y=StreamingEngine(g,store,mm,MemoryPlanner(256).plan(g)).run(torch.ones(1,4)); ref=torch.relu(torch.ones(1,4)@state['linear1.weight'].T)@state['linear2.weight'].T; assert torch.allclose(y,ref); assert store.bytes_read < os.path.getsize(p)
