"""Audit saved RGB-D/poses for pixel-level consistency; no pose corrections saved.

PnP inliers are a diagnostic image-space consensus, not independent ground truth.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation


def main():
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--masks',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);s=a.source
    g=json.loads((s/'geometry_poses.json').read_text());c=json.loads((s/'sdk_calibration.json').read_text());k=np.array([[c['fxrgb'],0,c['cxrgb']],[0,c['fyrgb'],c['cyrgb']],[0,0,1.]])
    stamps={r['capture']:r['stamp'] for r in map(json.loads,(s/'frames.jsonl').read_text().splitlines())}
    captures={r['node']:r['capture'] for r in map(json.loads,(s/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted')}
    orb=cv2.ORB_create(nfeatures=1200,fastThreshold=10);frames=[];quality=[]
    for node,pose in zip(g['node_ids'],g['poses']):
        name=f'{stamps[captures[node]]:.6f}.png';rgb=cv2.imread(str(s/'rgb'/name));gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);depth=cv2.imread(str(s/'depth'/name),-1)
        mask=cv2.imread(str(a.masks/name),0) if a.masks else None
        if a.masks and mask is None:raise ValueError('Missing diagnostic mask')
        kp,d=orb.detectAndCompute(gray,mask)
        frames.append(dict(node=node,pose=np.array(pose),name=name,rgb=rgb,depth=depth,kp=kp,desc=d))
        quality.append(dict(node=node,laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()),mean_gray=float(gray.mean()),fraction_all_rgb_ge250=float(np.mean(np.all(rgb>=250,axis=2))),fraction_depth_valid=float(np.mean(depth>0))))
    print(f'Loaded {len(frames)} frames',flush=True)
    pairs=[];matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    for prev,cur in zip(frames,frames[1:]):
        row=dict(nodes=[prev['node'],cur['node']]);pairs.append(row)
        if prev['desc'] is None or cur['desc'] is None:continue
        matches=[x[0] for x in matcher.knnMatch(prev['desc'],cur['desc'],k=2) if len(x)==2 and x[0].distance<.75*x[1].distance]
        xyz=[];uv=[]
        for m in matches:
            u,v=prev['kp'][m.queryIdx].pt;x,y=round(u),round(v)
            if not 0<=x<prev['depth'].shape[1] or not 0<=y<prev['depth'].shape[0]:continue
            z=float(prev['depth'][y,x])/1000
            if .15<z<4:
                xyz.append([(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z]);uv.append(cur['kp'][m.trainIdx].pt)
        row['matches_with_depth']=len(xyz)
        if len(xyz)<12:continue
        xyz=np.array(xyz);uv=np.array(uv);cv2.setRNGSeed(42)
        ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,uv,k,None,iterationsCount=200,reprojectionError=3.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<12:continue
        ids=inliers.ravel()
        relative=np.linalg.inv(cur['pose'])@prev['pose'];predxyz=xyz[ids]@relative[:3,:3].T+relative[:3,3]
        projected=np.column_stack((predxyz[:,0]/predxyz[:,2]*k[0,0]+k[0,2],predxyz[:,1]/predxyz[:,2]*k[1,1]+k[1,2]))
        fitted=cv2.projectPoints(xyz[ids],rv,tv,k,None)[0].reshape(-1,2)
        errors=np.linalg.norm(projected-uv[ids],axis=1);fit_errors=np.linalg.norm(fitted-uv[ids],axis=1)
        delta_r=Rotation.from_matrix(cv2.Rodrigues(rv)[0]@relative[:3,:3].T).magnitude()*180/np.pi
        row.update(pnp_inliers=len(ids),export_projection_median_px=float(np.median(errors)),export_projection_p90_px=float(np.percentile(errors,90)),
                   pnp_projection_median_px=float(np.median(fit_errors)),translation_difference_m=float(np.linalg.norm(tv.ravel()-relative[:3,3])),rotation_difference_deg=float(delta_r))
        print(f'{row["nodes"]}: {len(ids)} inliers, exported median {np.median(errors):.2f}px',flush=True)
    valid=[r for r in pairs if 'pnp_inliers' in r]
    summary=dict(valid_pairs=len(valid),total_pairs=len(pairs),median_pair_export_projection_px=float(np.median([r['export_projection_median_px'] for r in valid])),
        median_pair_pnp_projection_px=float(np.median([r['pnp_projection_median_px'] for r in valid])),pairs_over_5px=sum(r['export_projection_median_px']>5 for r in valid),
        median_pose_rotation_difference_deg=float(np.median([r['rotation_difference_deg'] for r in valid])),median_pose_translation_difference_mm=float(np.median([r['translation_difference_m']*1000 for r in valid])))
    report=dict(summary=summary,quality=quality,pairs=pairs,caveat=__doc__)
    (a.output/'input_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
