"""Compare live GPU Gaussian rasterizer against saved Brush eval images."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gaussian_scene import GaussianScene,saved_cameras
from gaussian_renderer import GaussianRenderer
import tkinter as tk

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.result/'report.json').read_text());model=a.result/report['model_relative']
    before=hashlib.sha256(model.read_bytes()).hexdigest()
    scene=GaussianScene(model);cameras={c['name']:c for c in saved_cameras(model)}
    root=tk.Tk();root.withdraw();host=tk.Frame(root);host.pack();root.update_idletasks()
    renderer=GaussianRenderer(host);rows=[];montage=[]
    try:
        for view in report['evaluation_views']:
            target=cv2.imread(str(a.result/view['render_relative']))[:,:,::-1];h,w=target.shape[:2]
            c=cameras[view['name']];iw,ih=c['size'];intr=c['intrinsics']*np.array([w/iw,h/ih,w/iw,h/ih])
            start=time.perf_counter();actual=renderer.draw_scene(scene,c['pose'],intr,w,h,readback=True,present=False)
            elapsed=(time.perf_counter()-start)*1000
            mse=np.mean((actual.astype(float)-target)**2)
            row=dict(name=view['name'],psnr_to_brush_db=float(10*np.log10(255**2/max(mse,1e-12))),mae=float(np.mean(np.abs(actual.astype(float)-target))),ms=elapsed)
            rows.append(row);print(json.dumps(row),flush=True)
            cv2.imwrite(str(a.output/view['name']),actual[:,:,::-1])
            montage.append(np.concatenate([target,actual,np.minimum(np.abs(target.astype(int)-actual)*4,255).astype(np.uint8)],axis=1))
        cv2.imwrite(str(a.output/'comparison.png'),np.concatenate(montage,axis=0)[:,:,::-1])
        result=dict(gpu=renderer.name,views=rows,source_unchanged=before==hashlib.sha256(model.read_bytes()).hexdigest(),
            mean_psnr_to_brush_db=float(np.mean([r['psnr_to_brush_db'] for r in rows])))
        (a.output/'report.json').write_text(json.dumps(result,indent=2))
    finally:renderer.close();root.destroy()

if __name__=='__main__':main()
