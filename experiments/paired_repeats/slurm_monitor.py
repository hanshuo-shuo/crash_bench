"""Read job state without treating a control-plane outage as a compute failure."""
import subprocess

FAILED_STATES={'FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
LIVE_STATES={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_HOLD','SIGNALING','STAGE_OUT'}

def read_job_state(job):
    if not str(job).isdigit():
        raise ValueError('Expected one numeric job ID')
    errors=[]
    # squeue reads the controller, so a healthy running job needs no accounting DB.
    for name,args in [('squeue',['squeue','-h','-j',str(job),'-o','%i|%T']),
                      ('sacct',['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State'])]:
        try:
            result=subprocess.run(args,text=True,capture_output=True,timeout=8,check=False)
        except (OSError,subprocess.TimeoutExpired) as error:
            errors.append(name+': '+type(error).__name__);continue
        if result.returncode:
            errors.append(name+': '+result.stderr.strip()[:240]);continue
        for line in result.stdout.splitlines():
            fields=line.strip().split('|')
            if len(fields)>=2 and fields[0].strip()==str(job) and fields[1].strip():
                state=fields[1].strip().split()[0].rstrip('+')
                if state in LIVE_STATES|FAILED_STATES|{'COMPLETED'}:
                    return {'state':state,'source':name,'query_errors':errors}
    # Empty results, unavailable databases and command timeouts are UNKNOWN.
    # The bounded worker can continue serving authorized requests and retry later.
    return {'state':'UNKNOWN','source':None,'query_errors':errors}
