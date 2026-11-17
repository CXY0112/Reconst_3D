"""Compare native optimizers on identical saved views and synthetic translations.

This is an offline integration/timing check, not a ground-truth SLAM benchmark.
"""
import argparse,csv,hashlib,json,re,sqlite3,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rtab_client import RtabClient
from validate_rtab_vo import render


def database_parameters(path):
    with sqlite3.connect(path) as db:
        text=db.execute('SELECT parameters FROM Info ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        counts=db.execute('SELECT type,count(*) FROM Link GROUP BY type').fetchall()
    return dict(item.split(':',1) for item in text.split(';') if ':' in item),dict(counts)


def run(folder,frames,k,optimizer):
    folder.mkdir();client=RtabClient(folder,optimizer=optimizer);rows=[];poses=np.empty((0,13))
    try:
        for i,(rgb,depth,stamp,shift) in enumerate(frames):
            started=time.perf_counter();r=client.frame(rgb,depth,k,stamp)
            pose_error=None
            if r['status']==2 and shift is not None:
                p=r['poses'][r['poses'][:,0]==r['node']][0,1:].reshape(3,4)
                pose_error=float(np.linalg.norm(p[:,3]-[0,-shift,0])*1000)
            rows.append(dict(frame=i,status=r['status'],node=r['node'],total_ms=(time.perf_counter()-started)*1000,
                odometry_ms=r['timings'][0],graph_ms=r['timings'][1],inliers=r['timings'][2],
                synthetic_translation_error_mm=pose_error))
            poses=r['poses']
        client.close()
    finally:client.close(force=True)
    with (folder/'frames.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    np.save(folder/'poses.npy',poses)
    params,links=database_parameters(folder/'map.db')
    if optimizer=='isam2':
        assert params['Optimizer/Strategy']=='2' and params['GTSAM/Incremental']=='true',params
    result=dict(optimizer=client.optimizer_info,frames=len(rows),lost=sum(r['status']==0 for r in rows),
        keyframes=sum(r['status']==2 for r in rows),final_graph_nodes=len(poses),link_types=links,
        database_optimizer={key:value for key,value in params.items() if key.startswith(('Optimizer/','GTSAM/'))})
    for key in ('total_ms','odometry_ms','graph_ms'):
        values=[r[key] for r in rows if key!='graph_ms' or r[key]>0]
        result[key]=dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),max=float(np.max(values)))
    errors=[r['synthetic_translation_error_mm'] for r in rows if r['synthetic_translation_error_mm'] is not None]
    if errors:result['synthetic_translation_error_mm']=dict(p50=float(np.median(errors)),max=float(max(errors)))
    (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(folder.name,json.dumps(result),flush=True)
    return result,params


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/test_new/setp1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=180)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();names=sorted((source/'rgb').glob('*.png'))
    if not names:raise ValueError('No saved RGB-D frames')
    k=np.array(json.loads(re.search(r'camera_matrix:.*?data: (\[.*?\])',(source/'hp60c.yaml').read_text(),re.S).group(1))).reshape(3,3)
    files=[source/'hp60c.yaml',*[p for path in names for p in (path,source/'depth'/path.name)]]
    digest=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before=digest();real=[]
    for i,path in enumerate(names):
        # Deliberately retain original intervals: these are sparse saved keyframes,
        # not a continuous 10 Hz video and not a fair live tracking benchmark.
        real.append((cv2.imread(str(path)),cv2.imread(str(source/'depth'/path.name),-1),float(path.stem),None))
    rgb,depth=real[0][:2];synthetic=[]
    for i in range(args.frames):
        phase=i%120;shift=.005*(phase if phase<=60 else 120-phase)
        color,dep=render(rgb,depth,k,shift);synthetic.append((color,dep,1+i*.1,shift))
    report=dict(source=str(source),limitations='Saved keyframes are sparse; synthetic views are one translated image. No independent camera trajectory ground truth. Not evidence of long-sequence speedup.',runs={})
    for name,frames in (('synthetic',synthetic),('saved_keyframes',real)):
        baseline_params=None
        for optimizer in ('default','isam2'):
            result,params=run(args.output/(name+'_'+optimizer),frames,k,optimizer)
            report['runs'][name+'_'+optimizer]=result
            if baseline_params is None:baseline_params=params
            else:report[name+'_parameter_differences']={key:[baseline_params.get(key),value] for key,value in params.items() if baseline_params.get(key)!=value}
    report['source_hashes_unchanged']=before==digest();assert report['source_hashes_unchanged']
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
