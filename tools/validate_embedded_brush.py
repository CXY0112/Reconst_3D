"""Visible offline workbench, real Gaussian viewport verification in both stages."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from desktop_display import enable_dpi_awareness
enable_dpi_awareness()
import tkinter as tk
from scene_workbench import SceneWorkbench


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'finish').exists():raise ValueError('Choose a fresh validation output folder')
    root=tk.Tk();app=SceneWorkbench(root);app.dataset=a.dataset.resolve()
    root.title('3DGS Scene 修复验证');root.geometry('1450x900')
    app.select_stage('scene_reconstruction');start=time.monotonic();records={};last=None
    errors=[];root.report_callback_exception=lambda *args:errors.append(str(args))
    def check():
        nonlocal last
        if app.closing:return
        for name in ('scene_reconstruction','review'):
            view=app.views.get(name);preview=getattr(view,'splat_preview',None)
            renderer=getattr(preview,'renderer',None)
            if renderer and renderer.draw_count:
                previous=records.get(name,{})
                records[name]=dict(gpu=renderer.gpu.name if renderer.gpu else None,
                    scene_only=not hasattr(preview,'tabs'),size=[renderer.host.winfo_width(),renderer.host.winfo_height()],
                    error=renderer.error,draw_count=renderer.draw_count,render_ms=renderer.render_ms,
                    interaction_observed=previous.get('interaction_observed',False) or not np.allclose(renderer.pose,renderer.initial_pose),
                    camera_restored=bool(np.allclose(renderer.pose,renderer.initial_pose)))
        current=tuple((name,r['draw_count']) for name,r in records.items())
        if current!=last:
            print(json.dumps(records,ensure_ascii=False),flush=True);last=current
            (a.output/'live.json').write_text(json.dumps(records,indent=2))
        if (a.output/'finish').is_file() or time.monotonic()-start>900:app.close();return
        root.after(300,check)
    root.after(300,check)
    try:root.mainloop()
    finally:
        checked=all(r['gpu'] and r['scene_only'] and r['error'] is None for r in records.values())
        result=dict(dataset=str(app.dataset),records=records,callback_errors=errors,
            all_phases_passed=set(records)=={'scene_reconstruction','review'} and checked and not errors)
        (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
