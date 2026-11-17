"""Exercise real workbench tabs; record CPU submission and GPU query times separately."""
from pathlib import Path
import ctypes as ct
import time
import numpy as np


def run_gui(source,folder,mark):
    import cv2
    import tkinter as tk
    from workbench import Workbench
    from mesh_view import MeshPanel
    from result_paths import RESULTS
    from unittest.mock import patch
    root=tk.Tk();app=Workbench(root,'acquisition');results={};hooks={}
    def pump_until(condition,timeout=300):
        deadline=time.monotonic()+timeout
        while not condition() and time.monotonic()<deadline:
            root.update();time.sleep(.003)
        if not condition():raise TimeoutError('GUI workload timed out')
    def instrument(panel):
        renderer=panel.renderer
        if renderer is None:raise RuntimeError('GPU renderer not initialized: '+panel.info.get())
        if id(renderer) in hooks:return hooks[id(renderer)]
        record={'cpu_ms':[],'gpu_ms':[],'gpu':renderer.name,'buffer_bytes':0}
        name='draw_mesh' if isinstance(panel,MeshPanel) else 'draw'
        original=getattr(renderer,name)
        gen=renderer._extension('glGenQueries',None,ct.c_int,ct.POINTER(ct.c_uint))
        begin=renderer._extension('glBeginQuery',None,ct.c_uint,ct.c_uint)
        end=renderer._extension('glEndQuery',None,ct.c_uint)
        get=renderer._extension('glGetQueryObjectui64v',None,ct.c_uint,ct.c_uint,ct.POINTER(ct.c_uint64))
        query=ct.c_uint()
        renderer.gl.wglMakeCurrent(renderer.dc,renderer.context)
        if all((gen,begin,end,get)):gen(1,ct.byref(query))
        def timed(*args,**kwargs):
            renderer.gl.wglMakeCurrent(renderer.dc,renderer.context)
            if query.value:begin(0x88BF,query.value)
            start=time.perf_counter();value=original(*args,**kwargs)
            if query.value:
                end(0x88BF);ns=ct.c_uint64();get(query.value,0x8866,ct.byref(ns));record['gpu_ms'].append(ns.value/1e6)
            record['cpu_ms'].append((time.perf_counter()-start)*1000)
            return value
        setattr(renderer,name,timed);hooks[id(renderer)]=record
        return record
    def draw_phase(name,panels,seconds=5,live=None):
        # Warm up uploads before measuring steady camera motion.
        for p in panels:p.request_draw()
        for _ in range(20):root.update();time.sleep(.005)
        records=[instrument(p) for p in panels]
        for record in records:record['cpu_ms'].clear();record['gpu_ms'].clear()
        mark(folder,name+'_start');start=time.monotonic();next_motion=start;next_live=start;live_times=[];rgb_times=[]
        while time.monotonic()-start<seconds:
            now=time.monotonic()
            if now>=next_motion:
                for panel in panels:panel.camera.yaw+=.0008;panel.request_draw()
                next_motion=now+1/30
            if live is not None and now>=next_live:
                scan,points,color=live
                t=time.perf_counter();scan.rgb_panel.update(color);rgb_times.append((time.perf_counter()-t)*1000)
                t=time.perf_counter();scan.live_panel.update(points);live_times.append((time.perf_counter()-t)*1000)
                next_live=now+.1
            root.update();time.sleep(.001)
        elapsed=time.monotonic()-start;mark(folder,name+'_end')
        def distribution(values):return {'samples':len(values),'p50':float(np.median(values)) if values else None,
            'p90':float(np.percentile(values,90)) if values else None,'max':float(max(values)) if values else None}
        results[name]={'wall_seconds':elapsed,'viewport':[(p.label.winfo_width(),p.label.winfo_height()) for p in panels],
            'panels':[{'gpu':r['gpu'],'cpu_submission_and_wait_ms':distribution(r['cpu_ms']),
                'gpu_elapsed_ms':distribution(r['gpu_ms']),'actual_frames_per_second':len(r['cpu_ms'])/elapsed,
                'geometry_buffer_bytes':int(p.packed.nbytes+p.triangles.nbytes) if isinstance(p,MeshPanel) else int(p.points.nbytes)} for p,r in zip(panels,records)],
            'live_cloud_update_ms':distribution(live_times),'rgb_update_ms':distribution(rgb_times),
            'note':'30Hz requested camera motion; live cloud/RGB requested at 10Hz; no camera I/O'}
    try:
        scan=app.views['acquisition'];mark(folder,'gui_stage1_import_start');scan.import_cloud(source/'map_cloud.ply')
        pump_until(lambda:not scan.importing);mark(folder,'gui_stage1_import_end')
        if not scan.has_map:raise ValueError('Map import failed')
        import json
        frame=json.loads((source/'frames.jsonl').read_text(encoding='utf-8').splitlines()[0]);name=f'{frame["stamp"]:.6f}'
        points=np.load(source/'pointcloud_raw'/(name+'.npy')).reshape(-1,3)/1000
        points=points[np.isfinite(points).all(axis=1)&(points[:,2]>=.15)&(points[:,2]<=4)]
        color=cv2.imread(str(source/'rgb_raw'/(name+'.png')))
        calibration=json.loads((source/'sdk_calibration.json').read_text(encoding='utf-8'))
        scan.live_panel.intrinsics=tuple(calibration[k] for k in ('fxir','fyir','cxir','cyir'))
        draw_phase('gui_stage1_display',[scan.map_panel],live=(scan,points,color))
        pre=app.select_stage('preprocessing');pre.backend.set('卡尔曼表面融合')
        mark(folder,'gui_stage2_load_start');pre.load(source);pump_until(lambda:pre.working is None)
        mark(folder,'gui_stage2_compute_start');pre.process();pump_until(lambda:pre.working is None)
        if pre.result is None:raise ValueError('Preprocess failed')
        mark(folder,'gui_stage2_compute_end')
        mark(folder,'gui_stage2_save_start');pre.save(folder/'setp2_gui');pump_until(lambda:pre.working is None)
        mark(folder,'gui_stage2_save_end')
        draw_phase('gui_stage2_display',[pre.before,pre.after])
        geometry=app.select_stage('reconstruction');geometry.set_source(pre.saved_path);pump_until(lambda:geometry.working is None)
        mark(folder,'gui_stage3_compute_start')
        with patch('geometry_window.stage_result_path',return_value=folder/'setp3_gui'):
            geometry.start();pump_until(lambda:geometry.working is None)
        if geometry.output is None:raise ValueError('Geometry failed')
        mark(folder,'gui_stage3_compute_end');draw_phase('gui_stage3_display',[geometry.before,geometry.after])
        appearance=app.select_stage('appearance');appearance.set_source(geometry.output);pump_until(lambda:appearance.working is None)
        mark(folder,'gui_stage4_compute_start')
        with patch('appearance_window.stage_result_path',return_value=folder/'setp4_gui'):
            appearance.start();pump_until(lambda:appearance.working is None)
        if appearance.output is None:raise ValueError('Appearance failed')
        mark(folder,'gui_stage4_compute_end');draw_phase('gui_stage4_display',[appearance.before,appearance.after])
        mark(folder,'gui_all_tabs_retained_idle_start')
        start=time.monotonic()
        while time.monotonic()-start<5:root.update();time.sleep(.01)
        mark(folder,'gui_all_tabs_retained_idle_end')
        results['loaded_counts']={'raw_map':len(scan.map_panel.points),'processed':len(pre.result.points),
            'mesh_vertices':len(appearance.mesh[0]),'mesh_triangles':len(appearance.mesh[1]),'live_raw_points':len(points)}
        results['note']='Actual workbench; stage1 is imported replay, stage2-4 run in actual UI with earlier tabs retained; no SDK acquisition or PointDSC replay in this GUI process'
        app.close()
    finally:
        try:root.destroy()
        except tk.TclError:pass
    return results
