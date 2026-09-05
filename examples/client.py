"""Local-only client; uses urllib and never contacts a cloud service."""
import json, sys, urllib.request
BASE=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8000/v1'

def call(path,payload=None,stream=False):
    req=urllib.request.Request(BASE+path,method='POST' if payload else 'GET',headers={'Content-Type':'application/json','Authorization':'Bearer local'})
    if payload: req.data=json.dumps(payload).encode()
    with urllib.request.urlopen(req,timeout=30) as r:
        if stream:
            for line in r:
                if line.startswith(b'data: '): print(line.decode().strip()[6:])
        else: print(r.read().decode())
call('/models')
body={'model':'local','messages':[{'role':'user','content':'Hello'}],'stream':False}; call('/chat/completions',body)
body['stream']=True; body['messages'][0]['content']='Explain Python generators.'; call('/chat/completions',body,True)
