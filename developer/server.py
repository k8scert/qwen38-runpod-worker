"""Internal test developer tools; not a multi-tenant execution service."""
import asyncio
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from typing import Literal
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import hmac

app=FastAPI(title='P13B Developer Tools',version='0.1.0')
bearer=HTTPBearer(auto_error=False)
gate=asyncio.Semaphore(1)

def authorized(credentials:HTTPAuthorizationCredentials=Depends(bearer)):
    expected=os.getenv('AGENT_TOOL_TOKEN','')
    if not expected:raise HTTPException(503,'Tool service not configured')
    if not credentials or not hmac.compare_digest(credentials.credentials,expected):
        raise HTTPException(401,'Invalid tool token')

class Program(BaseModel):
    language:Literal['python','javascript']='python'
    code:str=Field(min_length=1,max_length=16000)
    timeout_s:int=Field(default=10,ge=1,le=30)

@app.get('/health')
def health():return {'status':'ok','configured':bool(os.getenv('AGENT_TOOL_TOKEN'))}

@app.post('/execute',dependencies=[Depends(authorized)],operation_id='execute_program')
async def execute(program:Program):
    """Run Python or JavaScript and return real output, exit status and timeout."""
    try:await asyncio.wait_for(gate.acquire(),2)
    except TimeoutError:raise HTTPException(429,'Developer tool busy')
    try:
        with tempfile.TemporaryDirectory(prefix='p13b-tool-') as workspace:
            script=Path(workspace)/('task.py' if program.language=='python' else 'task.js')
            script.write_text(program.code)
            started=time.monotonic();timed_out=False
            with open(Path(workspace)/'stdout','wb') as out,open(Path(workspace)/'stderr','wb') as err:
                child_env={'PATH':os.getenv('PATH','/usr/local/bin:/usr/bin:/bin'),'HOME':workspace,'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1'}
                process=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).with_name('supervisor.py')),program.language,str(script),str(program.timeout_s),cwd=workspace,stdout=out,stderr=err,start_new_session=True,env=child_env)
                try:await asyncio.wait_for(process.wait(),program.timeout_s)
                except TimeoutError:
                    timed_out=True
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    await process.wait()
                except BaseException:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    await process.wait()
                    raise
                finally:
                    # Children must not outlive a finished or disconnected tool call.
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            def read(name):
                path=Path(workspace)/name
                with path.open('rb') as stream:data=stream.read(32768)
                return data.decode(errors='replace'),path.stat().st_size>32768
            stdout,out_cut=read('stdout');stderr,err_cut=read('stderr')
            return {'stdout':stdout,'stderr':stderr,'returncode':process.returncode,
                    'timed_out':timed_out,'output_truncated':out_cut or err_cut,
                    'elapsed_s':round(time.monotonic()-started,3)}
    finally:gate.release()
