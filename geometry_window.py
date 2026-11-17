from dataset_store import route_dataset
from runtime_paths import geometry_python
"""Third-stage controls and embedded mesh view; reconstruction runs off the UI thread."""
from pathlib import Path
from contextlib import nullcontext
from dataset_store import replace_stage, stage_folder, dataset_for
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from result_paths import RESULTS, stage_result_path, latest_preprocessed, result_reports
from saved_sources import reference
from map_cloud_view import MapCloudPanel
from mesh_view import MeshPanel
from live_cloud_view import ViewCamera
from geometry_methods import METHODS,DESCRIPTIONS,is_geometry_report

ROOT = Path(__file__).resolve().parent


def display_points(world):
    world = np.asarray(world, dtype=np.float32)
    if world.ndim != 2 or world.shape[1] != 3 or not np.isfinite(world).all():
        raise ValueError('预处理结果包含无效坐标')
    return np.ascontiguousarray(np.column_stack((-world[:, 1], -world[:, 2], world[:, 0])))


class GeometryWindow:
    def __init__(self, root):
        self.root = root; self.source = None; self.output = None; self.report = None
        self.working = None; self.closed = False
        self.events = queue.Queue(); self.cancel = threading.Event(); self.lock = threading.Lock(); self.process = None
        self.last_views = None
        controls = ttk.Frame(root, padding=8); controls.pack(fill='x')
        self.method=tk.StringVar(value='Poisson')
        self.method_box=ttk.Combobox(controls,textvariable=self.method,values=METHODS,state='readonly',width=9)
        self.method_box.pack(side='left',padx=3)
        self.method_box.bind('<<ComboboxSelected>>',lambda _:self.method_text.set(DESCRIPTIONS[self.method.get().lower()]))
        self.source_button = ttk.Button(controls, text='导入重建源目录', command=self.open_source); self.source_button.pack(side='left', padx=3)
        self.latest_button = ttk.Button(controls, text='使用最新二阶段结果', command=self.use_latest); self.latest_button.pack(side='left', padx=3)
        self.start_button = ttk.Button(controls, text='开始几何重建', command=self.start); self.start_button.pack(side='left', padx=3)
        self.cancel_button = ttk.Button(controls, text='取消', command=self.cancel_job); self.cancel_button.pack(side='left', padx=3)
        self.result_button = ttk.Button(controls, text='导入重建结果', command=self.open_result); self.result_button.pack(side='left', padx=3)
        self.sync = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='同步视角', variable=self.sync, command=self.sync_from_left).pack(side='left', padx=10)
        self.wireframe = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text='线框', variable=self.wireframe, command=lambda: self.after.set_wireframe(self.wireframe.get())).pack(side='left')
        self.source_text = tk.StringVar(value='请选择二阶段结果、跳过记录或一阶段扫描；TSDF直接读取源深度图')
        ttk.Label(root, textvariable=self.source_text, padding=(10, 4), wraplength=1400).pack(fill='x')
        self.method_text=tk.StringVar(value=DESCRIPTIONS['poisson'])
        ttk.Label(root, textvariable=self.method_text, padding=(10, 4),wraplength=1400).pack(fill='x')
        self.status = tk.StringVar(value='就绪 · 无需连接相机')
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1400).pack(fill='x')
        self.bar = ttk.Progressbar(root, maximum=100); self.bar.pack(fill='x', padx=10)
        panels = ttk.Panedwindow(root, orient='horizontal'); panels.pack(fill='both', expand=True)
        left = ttk.Labelframe(panels, text='输入点云'); right = ttk.Labelframe(panels, text='最终几何网格')
        panels.add(left, weight=1); panels.add(right, weight=1)
        self.before = MapCloudPanel(left); self.after = MeshPanel(right)
        self.summary = tk.StringVar(value='结果自动保存至 setp3_时间 文件夹；原始数据只读，默认仅保存最终网格与报告。')
        ttk.Label(root, textvariable=self.summary, padding=8, wraplength=1400).pack(fill='x')
        self._buttons(); self.timer = root.after(40, self._poll)

    def _buttons(self):
        idle = self.working is None
        if hasattr(self,'method_box'):self.method_box.configure(state='readonly' if idle else 'disabled')
        for button in (self.source_button, self.latest_button, self.result_button):
            button.configure(state='normal' if idle else 'disabled')
        self.start_button.configure(state='normal' if idle and self.source is not None else 'disabled')
        self.cancel_button.configure(state='normal' if not idle else 'disabled')

    def _begin(self, kind, text):
        self.working = kind; self.cancel.clear(); self.status.set(text); self.bar['value'] = 0; self._buttons()

    def open_source(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=str(RESULTS), title='选择二阶段结果、跳过记录或一阶段扫描目录')
        if path: self.set_source(path)

    def use_latest(self):
        if self.working: return
        try:
            if getattr(self,'dataset',None):
                source=stage_folder(self.dataset,2)
                if not (source/'report.json').is_file():raise FileNotFoundError('请先完成或跳过二阶段')
                self.set_source(source)
            else:self.set_source(latest_preprocessed())
        except Exception as exc: messagebox.showerror('暂无预处理结果', str(exc), parent=self.root)

    def set_source(self, path):
        if route_dataset(self,path):return
        if self.working: return
        folder = Path(path).resolve(); folder = folder if folder.is_dir() else folder.parent
        self._begin('load', '正在读取预处理结果…')
        def worker():
            try:
                points = self._source_points(folder)
                self.events.put(('source', (folder, points)))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _source_points(folder):
        from geometry_inputs import source_info
        info=source_info(folder)
        if info['kind']=='preprocessed':
            if not info['report'].get('complete'):raise ValueError('预处理结果尚未完整保存')
            return display_points(np.load(info['cloud']))
        from cloud_import import load_cloud
        return load_cloud(info['cloud']).points

    def open_result(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=str(RESULTS), title='选择已保存的几何重建目录（setp3）')
        if path: self.load_result(path)

    def _command(self, command):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONIOENCODING='utf-8')
        if self.cancel.is_set(): raise InterruptedError()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', env=env,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        with self.lock: self.process = process
        if self.cancel.is_set(): process.terminate()
        lines = []
        try:
            for line in process.stdout:
                lines.append(line.strip()); lines = lines[-40:]
                match = re.match(r'([1-6])/6 ', line)
                if match: self.events.put(('progress', int(match[1])))
                rgb_match = re.match(r'RGB (\d+)/(\d+) ', line)
                if rgb_match: self.events.put(('rgb_progress', tuple(map(int, rgb_match.groups()))))
            status = process.wait()
            if self.cancel.is_set(): raise InterruptedError()
            if status: raise RuntimeError('\n'.join(lines)[-3000:] or f'后台进程退出：{status}')
        finally:
            process.stdout.close()
            with self.lock: self.process = None

    def _packet_job(self, folder, reconstruct):
        try:
            context = replace_stage(folder, 3, getattr(self, 'dataset', None)) if reconstruct else nullcontext(folder)
            with context as destination:
                payload = self._packet_data(destination, reconstruct)
                if self.cancel.is_set(): raise InterruptedError()
            self.events.put(('mesh', (folder, *payload)))
        except InterruptedError: self.events.put(('cancelled', None))
        except Exception as exc: self.events.put(('error', str(exc)))

    def _packet_data(self, folder, reconstruct):
        RESULTS.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.geometry_preview_', dir=RESULTS) as temporary:
            if Path(temporary).resolve().parent != RESULTS.resolve(): raise ValueError('Unsafe preview directory')
            packet = Path(temporary)/'mesh.npz'
            started = time.perf_counter()
            if reconstruct:
                self._command([str(geometry_python()), '-u', str(ROOT/'tools/reconstruct_geometry.py'),
                               '--worker', '--method',self.job_method,'--source', str(self.source), '--output', str(folder), '--preview-packet', str(packet)])
            else:
                metadata = json.loads((folder/'report.json').read_text(encoding='utf-8'))
                if not is_geometry_report(metadata):
                    raise ValueError('请选择完整的正式几何重建结果目录')
                self._command([str(geometry_python()), '-u', str(ROOT/'mesh_read_worker.py'), str(folder/'mesh.ply'), str(packet)])
            report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
            if reconstruct:
                report['wall_seconds_including_worker_startup'] = time.perf_counter()-started
                (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            with np.load(packet) as data:
                mesh = tuple(data[key].copy() for key in ('vertices', 'triangles', 'normals'))
            source = reference(report['input'], folder)
            points = self._source_points(source) if source is not None else None
            return report, mesh, points

    def start(self):
        if self.working or self.source is None: return
        if not getattr(self,'can_write',lambda:True)():return
        folder = stage_folder(self.dataset, 3) if getattr(self, 'dataset', None) else stage_result_path(3)
        self.job_method='poisson' if getattr(self,'beta',False) else self.method.get().lower()
        if self.method.get() not in METHODS:raise ValueError('请选择Poisson、NKSR或TSDF')
        self._begin('reconstruct', '正在启动几何重建…')
        threading.Thread(target=self._packet_job, args=(folder, True), daemon=True).start()

    def load_result(self, path):
        if route_dataset(self,path):return
        if self.working: return
        folder = Path(path).resolve(); folder = folder if folder.is_dir() else folder.parent
        self._begin('mesh_load', '正在读取三角网格…')
        threading.Thread(target=self._packet_job, args=(folder, False), daemon=True).start()

    def cancel_job(self):
        self.cancel.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                try: self.process.terminate()
                except OSError: pass
        if self.working: self.status.set('正在停止后台任务…已完整保存的结果将保留')

    def sync_from_left(self):
        if self.sync.get():
            src = self.before.camera
            self.after.camera = ViewCamera(src.position, src.yaw, src.pitch,src.roll); self.after.request_draw()
        self.last_views = self._view_states()

    def _view_states(self):
        return [tuple(panel.camera.position)+(panel.camera.yaw, panel.camera.pitch,panel.camera.roll) for panel in (self.before, self.after)]

    def _sync_views(self):
        states = self._view_states()
        if self.sync.get() and self.last_views is not None:
            changed = [states[i] != self.last_views[i] for i in (0, 1)]
            if any(changed):
                src, dst = (self.after, self.before) if changed[1] and not changed[0] else (self.before, self.after)
                dst.camera = ViewCamera(src.camera.position, src.camera.yaw, src.camera.pitch,src.camera.roll); dst.request_draw()
                states = self._view_states()
        self.last_views = states

    def _poll(self):
        if self.closed: return
        while True:
            try: kind, payload = self.events.get_nowait()
            except queue.Empty: break
            if kind in ('error','cancelled') and getattr(self,'on_task_failed',None):self.on_task_failed(str(payload or '已取消'))
            if kind == 'progress':
                names = ('Poisson重建', '观测支持裁剪', '受约束平滑', '减面及检查', '平面感知优化', '最终检查及保存')
                if getattr(self,'job_method','poisson')!='poisson':
                    names=('读取输入和模型环境','模型重建和网格提取','检查网格','评估点云一致性','校验输入','保存结果')
                self.bar['value'] = (payload-1)/6*100; self.status.set(f'{payload}/6 · {names[payload-1]}…'); continue
            job=self.working;self.working = None
            if self.cancel.is_set():
                self.status.set('已停止；未完成的结果不会作为正式输出使用')
            elif kind == 'source':
                self.source, points = payload; self.output = self.report = None
                self.before.clear(); self.after.clear(); self.before.set_cloud(points)
                self.source_text.set(str(self.source)); self.status.set(f'已载入 {len(points):,} 点，可开始几何重建')
                self.sync_from_left()
            elif kind == 'mesh':
                self.output, self.report, mesh, points = payload
                selected=self.report.get('method','poisson')
                actual_label={'poisson':'Poisson','nksr':'NKSR','noksr':'NoKSR','tsdf':'TSDF'}[selected]
                self.method.set(actual_label if actual_label in METHODS else 'Poisson')
                self.method_text.set(DESCRIPTIONS[selected] if actual_label in METHODS else '已载入历史NoKSR结果；新重建可选Poisson、NKSR或TSDF。')
                if job=='reconstruct' and getattr(self,'on_stage_saved',None): self.on_stage_saved(3,self.output)
                if points is not None:
                    self.source = reference(self.report['input'],self.output); self.before.clear(); self.before.set_cloud(points)
                    self.source_text.set(str(self.source))
                self.after.clear(); self.after.set_mesh(*mesh); self.sync_from_left(); self.bar['value'] = 100
                final = self.report['final']
                elapsed=self.report.get('elapsed_seconds')
                timing=f'核心 {elapsed:.1f}s' if elapsed is not None else '复用历史结果，本次未计时'
                self.status.set(f'完成 · {final["vertices"]:,} 顶点 · {final["triangles"]:,} 三角面 · {timing}')
                self.summary.set(f'已保存：{self.output} · {actual_label} · {len(self.report.get("decimation_attempts",[]))} 次减面尝试 · 点面距离表示输入一致性，不是绝对精度')
            elif kind == 'error':
                self.status.set('操作失败 · 原显示和已保存数据保留')
                messagebox.showerror('几何重建', payload, parent=self.root)
            self._buttons()
        self._sync_views(); self.timer = self.root.after(40, self._poll)

    def shutdown(self):
        self.closed = True; self.root.after_cancel(self.timer)
