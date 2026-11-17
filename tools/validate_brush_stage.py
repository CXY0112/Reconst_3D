"""Offline actual main-workbench Brush stage integration using an existing dataset."""
import argparse,json,sys,time
from pathlib import Path
import tkinter as tk
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scene_workbench import SceneWorkbench
from brush_reconstruction import LABEL,validate_result
from scene_stages import SPLAT_VIEW


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();a.report.parent.mkdir(parents=True,exist_ok=True)
    root=tk.Tk();root.withdraw();app=SceneWorkbench(root);app.dataset=a.dataset.resolve();errors=[]
    root.report_callback_exception=lambda *args:errors.append(str(args))
    start=time.monotonic();seen=[]
    def wait(view,timeout=700):
        deadline=time.monotonic()+timeout;last=None
        while view.working:
            root.update();time.sleep(.03)
            state=(view.status.get(),round(float(view.bar['value'])))
            if state!=last:
                seen.append(dict(text=state[0],percent=state[1]));print(json.dumps(seen[-1],ensure_ascii=True),flush=True);last=state
            if time.monotonic()>deadline:raise TimeoutError('Workbench integration timeout')
        root.update()
    try:
        view=app.select_stage('scene_reconstruction');wait(view)
        if view.scene is None:raise RuntimeError('Cannot load saved scene')
        view.method.set(LABEL);view.method_changed();view.start_base();wait(view)
        if view.base is None or view.base[1].get('method')!='brush_3dgs':raise RuntimeError('Brush result missing')
        target,report,mesh,_=view.base;model=validate_result(target,report)
        review=app.open_review();wait(review)
        checks=dict(gaussian_result=mesh is None,model_valid=model.is_file(),
                    stage3_preview=view.splat_preview.renderer.scene is not None,
                    review_loaded=review.right_mode.get()==SPLAT_VIEW and review.current_model==model,
                    review_preview=review.splat_preview.renderer.scene is not None,
                    metrics_gaussians=review.metrics_values.get('gaussians')==report['gaussians'],
                    callbacks_clean=not errors,
                    manifest_saved=json.loads((app.dataset/'dataset.json').read_text())['branches']['rgbd_scene']['stages']['3']['stale'] is False)
        result=dict(dataset=str(app.dataset),target=str(target),checks=checks,passed=all(checks.values()),
                    model=str(model),gaussians=report['gaussians'],steps=report['parameters']['steps'],
                    mean_psnr_db=report['mean_psnr_db'],elapsed_seconds=time.monotonic()-start,progress=seen,callback_errors=errors)
        a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True,indent=2))
        if not result['passed']:raise RuntimeError('Brush main-workbench integration failed')
    finally:app.close()


if __name__=='__main__':main()
