import os, resource, shutil, sys
language,script,timeout=sys.argv[1:]
resource.setrlimit(resource.RLIMIT_CPU,(int(timeout)+1,int(timeout)+2))
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
if language=='python':
    executable=sys.executable;args=[executable,'-I',script]
else:
    executable=shutil.which('node')
    if not executable:raise SystemExit('Node unavailable')
    args=[executable,script]
os.execv(executable,args)
