"""Background neural matching and ICP. Archives are immutable; no map state is written."""
import sys,json,subprocess,os,time
from pathlib import Path
import numpy as np
from runtime_paths import geometry_python
from pointdsc_matcher import PointDSCMatcher
ROOT=Path(__file__).resolve().parent
matcher=None;refiner=None
try:
    for line in sys.stdin:
        job=json.loads(line)
        if job.get('command')=='close':break
        started=time.perf_counter()
        try:
            if matcher is None:matcher=PointDSCMatcher();matcher.torch.set_num_threads(1)
            proposals=[];failures=[]
            with np.load(job['packet']) as packet:
                for ref in job['references']:
                    try:
                        pose=matcher.estimate(packet[f'source_{ref}'],packet[f'target_{ref}'])
                        proposals.append(dict(reference=ref,pose=pose.tolist(),target_path=str(Path(job['archives'])/f'{ref}.npz')))
                    except (ValueError,RuntimeError) as exc:failures.append(dict(reference=ref,reason=str(exc)))
            if refiner is None:
                refiner=subprocess.Popen([str(geometry_python()),'-u',str(ROOT/'loop_refine_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                print('@@REFINER '+json.dumps(dict(pid=refiner.pid)),flush=True)
            refiner.stdin.write(json.dumps(dict(source_path=str(Path(job['archives'])/f"{job['source']}.npz"),proposals=proposals))+'\n');refiner.stdin.flush()
            for response in refiner.stdout:
                if response.startswith('@@REFINE '):result=json.loads(response[9:]);break
            else:raise RuntimeError('后台ICP进程退出')
            result.update(source=job['source'],version=job['version'],seconds=time.perf_counter()-started)
            result.setdefault('failures',[]).extend(failures)
        except Exception as exc:result=dict(error=str(exc),source=job['source'],version=job['version'])
        print('@@LOOP '+json.dumps(result),flush=True)
finally:
    if refiner is not None:
        refiner.stdin.close()
        try:refiner.wait(timeout=2)
        except subprocess.TimeoutExpired:refiner.terminate();refiner.wait()
