import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault('runpod', types.SimpleNamespace())
spec = importlib.util.spec_from_file_location('candidate_worker', Path(__file__).with_name('handler.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class Response:
    status = 200
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def text(self):
        return json.dumps({'choices':[{'message':{'content':'OK'},'finish_reason':'stop'}],
                           'usage':{'prompt_tokens':123,'completion_tokens':2}})
    async def json(self): return self.data
    @property
    def content(self):
        async def lines():
            yield b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n'
            yield b'data: [DONE]\n'
        return lines()

class Session:
    requests = []
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def post(self, url, json):
        self.requests.append(json)
        return Response()
    def get(self, url):
        response = Response()
        response.data = ([{'id':0,'n_ctx':131072,'prompt':'PRIVATE'}] if url.endswith('/slots')
                         else {'status':'ok','secret':'PRIVATE'})
        return response

class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self): Session.requests.clear()

    async def test_native_usage_and_tool_schema_forwarding(self):
        payload = {'messages':[{'role':'user','content':'hi'}],
                   'tools':[{'type':'function','function':{'name':'test','parameters':{'type':'object'}}}],
                   'tool_choice':'auto','cache_prompt':True,'id_slot':1,'ignored_secret':'PRIVATE'}
        with patch.object(worker.aiohttp, 'ClientSession', Session):
            result = [value async for value in worker.handler({'input':payload})]
        self.assertEqual(result[0]['response']['usage']['prompt_tokens'],123)
        self.assertEqual(Session.requests[0]['tools'],payload['tools'])
        self.assertEqual(Session.requests[0]['id_slot'],1)
        self.assertNotIn('ignored_secret',Session.requests[0])

    async def test_stream_requests_native_usage_and_preserves_done(self):
        with patch.object(worker.aiohttp, 'ClientSession', Session):
            result = [value async for value in worker.handler({'input':{'messages':[{'role':'user','content':'hi'}], 'stream':True}})]
        self.assertEqual(Session.requests[0]['stream_options'],{'include_usage':True})
        self.assertEqual(result[0]['chunk']['usage']['completion_tokens'],2)
        self.assertEqual(result[-1],{'done':True})

    async def test_optional_ops_token_is_enforced(self):
        with patch.dict(os.environ, {'P13B_DIAGNOSTICS':'1','P13B_OPS_TOKEN':'test'}):
            result = [value async for value in worker.handler({'input':{'p13b_action':'diagnostics','ops_token':'wrong'}})]
        self.assertEqual(result,[{'error':'diagnostics unauthorized'}])

    async def test_diagnostics_filters_prompt_and_secret_environment(self):
        async def unavailable(*args, **kwargs): raise FileNotFoundError()
        with patch.dict(os.environ,{'P13B_DIAGNOSTICS':'1','P13B_OPS_TOKEN':'','UPSTREAM_API_KEY':'PRIVATE'}), patch.object(worker.aiohttp,'ClientSession',Session), patch('asyncio.create_subprocess_exec',unavailable):
            result = [value async for value in worker.handler({'input':{'p13b_action':'diagnostics'}})]
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertEqual(result[0]['diagnostics']['slots'][0]['n_ctx'],131072)

if __name__ == '__main__': unittest.main()
