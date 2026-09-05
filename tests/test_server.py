import asyncio, json
import httpx
from stream_runtime.server import create_app
from test_core import model
from stream_runtime.architecture.generic import ArchitectureAdapter
from stream_runtime.graph.graph import ModelGraph

def prepared(tmp_path):
    p,_=model(tmp_path); from stream_runtime.storage import SafeTensorStream
    r=SafeTensorStream(p); nodes,tensors=ArchitectureAdapter().analyze(r); d=tmp_path/'prepared'; d.mkdir(); ModelGraph('generic','generic',nodes,tensors).save(d/'manifest.json'); (d/'model.path').write_text(p); return d

def test_local_offline_api(tmp_path):
    d=prepared(tmp_path); app=create_app(d,256,api_key='local')
    async def run():
        transport=httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,base_url='http://test') as c:
            assert (await c.get('/health')).json()['offline'] is True
            assert (await c.get('/v1/models')).json()['data'][0]['owned_by']=='local'
            assert (await c.post('/v1/chat/completions',json={'model':'x','messages':[{'role':'user','content':'hi'}]})).status_code==401
            h={'Authorization':'Bearer local'}; r=await c.post('/v1/chat/completions',headers=h,json={'model':'x','messages':[{'role':'user','content':'hi'}]}); assert r.json()['choices'][0]['message']['role']=='assistant'
            s=await c.post('/v1/chat/completions',headers=h,json={'model':'x','messages':[{'role':'user','content':'hi'}],'stream':True}); assert 'data: [DONE]' in s.text
            st=(await c.get('/v1/status')).json(); assert st['peak_ram'] <= 256 and st['model_on_disk']
    asyncio.run(run())
