import os, unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from server import app

class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'AGENT_TOOL_TOKEN':'test'});self.env.start()
        self.client=TestClient(app)
    def tearDown(self):self.client.close();self.env.stop()
    def call(self,code,language='python',timeout=10):
        response=self.client.post('/execute',headers={'Authorization':'Bearer test'},json={'code':code,'language':language,'timeout_s':timeout})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()
    def test_unauthorized(self):self.assertEqual(self.client.post('/execute',json={'code':'print(5)'}).status_code,401)
    def test_python_and_library(self):self.assertEqual(self.call('import json; print(2+3)')['stdout'].strip(),'5')
    def test_service_token_is_not_in_child_environment(self):self.assertEqual(self.call('import os; print(os.getenv("AGENT_TOOL_TOKEN","ABSENT"))')['stdout'].strip(),'ABSENT')
    def test_javascript(self):self.assertEqual(self.call('console.log(2+3)','javascript')['stdout'].strip(),'5')
    def test_timeout_then_recovery(self):
        self.assertTrue(self.call('while True: pass',timeout=1)['timed_out'])
        self.assertEqual(self.call('print(7)')['stdout'].strip(),'7')
    def test_large_output_is_bounded(self):
        result=self.call('print("x"*50000)')
        self.assertTrue(result['output_truncated']);self.assertLessEqual(len(result['stdout']),32768)

if __name__=='__main__':unittest.main()
