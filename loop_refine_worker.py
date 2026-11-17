"""Read-only ICP service used exclusively by the background loop process."""
import sys,json
from geometry_worker import Frame,refine,rigid_pose
for line in sys.stdin:
    try:
        job=json.loads(line);source=Frame.load(job['source_path']);links=[];failures=[]
        for proposal in job['proposals']:
            try:
                target=Frame.load(proposal['target_path'])
                pose,overlap,rmse,info=refine(source,target,rigid_pose(proposal['pose']))
                links.append(dict(target=proposal['reference'],pose=pose.tolist(),information=info.tolist(),overlap=overlap,rmse_mm=rmse*1000))
            except ValueError as exc:failures.append(dict(reference=proposal['reference'],reason=str(exc)))
        result=dict(links=links,failures=failures)
    except Exception as exc:result=dict(error=str(exc))
    print('@@REFINE '+json.dumps(result),flush=True)
