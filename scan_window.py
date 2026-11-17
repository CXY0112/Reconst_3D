"""Interactive incremental RTAB-Map scanner. All Tk calls stay on the UI thread."""
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time

import cv2
import numpy as np

from reconstruct import ROOT, Calibration, depth_points
from result_paths import stage_result_path
from dataset_store import replace_stage, stage_folder, dataset_for, create_dataset
from rtabmap_scan import RGBDCamera, register_rgbd, write_configuration
from online_backend import OnlineBackend, export_map


def read_cloud(path):
    from cloud_import import load_cloud
    return load_cloud(path).points


def depth_preview(depth, k):
    v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
    z = depth[::4, ::4].astype(float) / 1000
    valid = (z >= .15) & (z <= 4)
    return np.column_stack(((u[valid]-k[0, 2])*z[valid]/k[0, 0],
                            (v[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]))


def capture_result_text(count,report):
    text=_capture_result_text(count,report)
    tracking=report.get('tracking')
    if not tracking:return text
    score=tracking.get('confidence')
    label=f'质量评分 {score:.2f}' if score is not None else '质量评分待建立'
    state=tracking.get('state','NORMAL')
    prefix=f'{state} · {label}'
    if state=='RECOVERY':
        prefix+=f' · 连续确认 {tracking.get("recovery_good_count",0)}/{tracking.get("recovery_good_required",3)} · 暂停建图，请缓慢回到最后可信视角'
        recovery=tracking.get('recovery_method','')
        if recovery in ('RGB-D ORB/PnP','MAC + ICP'):prefix+=' · 恢复配准：'+recovery
        elif tracking.get('mac_status')=='running':prefix+=' · MAC 正在寻找恢复位姿'
    return prefix+' · '+text


def _capture_result_text(count,report):
    """Display actual frontend fields; geometric frontends have no RGB/ICP report."""
    name=report.get('method') or '配准'
    if not report.get('accepted'):
        return report.get('reason') or ('已跟踪，尚未保留新关键帧' if report.get('skipped') else '本次未加入地图')
    if 'reference' in report and report['reference'] is None and name!='RTAB-Map RGB-D F2M':
        return f'{name} · 基准位置已建立 · 请缓慢移动并保持重叠'
    parts=[f'{name} · 第 {count} 帧关键帧已接收']
    if 'geometry_edges' in report:parts.append(f'几何约束 {report["geometry_edges"]} 条')
    elif 'rgb_inliers' in report:parts.append(f'RGB内点 {report["rgb_inliers"]}')
    else:
        frontend=report.get('frontend') or {}
        if frontend.get('inliers') is not None:
            label='几何支持点' if frontend.get('inlier_definition') else '配准内点'
            parts.append(f'{label} {frontend["inliers"]}')
    validation=report.get('validation') or {}
    overlap=validation.get('overlap',report.get('overlap'))
    rmse=validation.get('rmse_m')
    rmse_mm=rmse*1000 if rmse is not None else report.get('rmse_mm')
    if overlap is not None:parts.append(f'几何重叠 {overlap:.0%}')
    if rmse_mm is not None:parts.append(f'几何残差 {rmse_mm:.1f}mm')
    if report.get('matching_ms') is not None:parts.append(f'前端 {report["matching_ms"]:.0f}ms')
    return ' · '.join(parts)


class ScanSession:
    def __init__(self, events, camera_factory=RGBDCamera, backend_factory=OnlineBackend, fps=10):
        (ROOT / 'results').mkdir(exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix='.pending_scan_', dir=ROOT / 'results'))
        for name in ('rgb', 'depth'):
            (self.folder / name).mkdir()
        self.events, self.camera_factory, self.backend_factory = events, camera_factory, backend_factory
        self.stop = threading.Event()
        self.capture_done = threading.Event()
        self.lock = threading.Lock()
        self.names = []
        self.k = None
        self.shape = None
        self.failed = False
        self.has_preview = False
        self.live_frame = None
        self.live_rgb = None
        self.incoming = queue.Queue(maxsize=2)
        self.dropped = 0
        self.processed = 0
        self.fps = fps
        self.minimum_frames = 2

    def submit(self, item):
        try:
            self.incoming.put_nowait(item)
        except queue.Full:
            try:
                self.incoming.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.incoming.put_nowait(item)

    def start(self):
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.map_thread = threading.Thread(target=self._mapping, daemon=True)
        self.capture_thread.start()
        self.map_thread.start()

    def _capture(self):
        try:
            with (self.folder / 'frames.jsonl').open('w', encoding='utf-8') as manifest:
                with self.camera_factory() as camera:
                    previous = None
                    deadline = 0
                    preview_deadline = 0
                    while not self.stop.is_set():
                        try:
                            (rgb, dep), calibration = camera.read(timeout=2 if self.names else 20)
                        except TimeoutError:
                            if self.stop.is_set():
                                break
                            raise
                        now = time.monotonic()
                        self.live_rgb = rgb[0]
                        ids = rgb[2], dep[2]
                        if now < deadline or (previous and (ids[0] == previous[0] or ids[1] == previous[1])):
                            continue
                        if self.stop.is_set():
                            break
                        depth_frame = getattr(camera,'depth_frame','ir')
                        color, depth, k = register_rgbd(rgb[0], dep[0], calibration,depth_frame=depth_frame)
                        name = f'{dep[3]/1000:.6f}.png'
                        for directory, image in [('rgb', color), ('depth', depth)]:
                            if not cv2.imwrite(str(self.folder / directory / name), image):
                                raise OSError('Cannot save scan frame')
                        with self.lock:
                            if self.k is None:
                                self.k, self.shape = k, depth.shape
                                write_configuration(self.folder, k, depth.shape, 10)
                                (self.folder / 'sdk_calibration.json').write_text(json.dumps(
                                    {n: getattr(calibration, n) for n, _ in Calibration._fields_}, indent=2))
                            self.names.append(name)
                            count = len(self.names)
                        manifest.write(json.dumps({'stamp': dep[3]/1000, 'rgb_id': ids[0], 'depth_id': ids[1],
                                                   'rgb_sdk_stamp': rgb[3], 'depth_sdk_stamp': dep[3]})+'\n')
                        manifest.flush()
                        self.submit((count, color, depth, k, dep[3]/1000))
                        if now >= preview_deadline:
                            self.live_frame = depth_points(dep[0],calibration,stride=1,camera=depth_frame)
                            self.live_intrinsics = tuple(getattr(calibration,n+depth_frame) for n in ('fx','fy','cx','cy'))
                            self.events.put(('frame', (count, None)))
                            preview_deadline = now + .5
                        previous = ids
                        deadline = now + 1/self.fps
        except Exception as exc:
            self.failed = True
            self.events.put(('error', f'相机采集失败：{exc}\n恢复数据：{self.folder}'))
            self.stop.set()
        finally:
            self.capture_done.set()

    def _publish(self, backend, count, final=False):
        started = time.perf_counter()
        points, nodes = backend.preview()
        elapsed = (time.perf_counter()-started)*1000
        with (self.folder/'preview_metrics.csv').open('a', encoding='utf-8') as log:
            if log.tell() == 0:
                log.write('capture_frame,fusion_ms,fused_points,node_count\n')
            log.write(f'{count},{elapsed:.3f},{len(points)},{len(nodes)}\n')
        if hasattr(backend,'preview_stats'):
            with (self.folder/'fusion_details.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(capture=count,**backend.preview_stats))+'\n')
        self.events.put(('fusion_time', elapsed))
        self.events.put(('final' if final else 'map', (count, points, nodes)))
        self.has_preview = len(points) > 0

    def _mapping(self):
        backend = None
        last_count = 0
        last_preview = 0
        try:
            backend = self.backend_factory(self.folder)
            with (self.folder/'online_metrics.csv').open('w', encoding='utf-8') as metrics:
                metrics.write('capture_frame,processed_frame,tracking_status,processing_ms,dropped_frames,odometry_ms,graph_ms,inliers,features,matches\n')
                while not self.capture_done.is_set() or not self.incoming.empty():
                    if self.failed:
                        break
                    if last_count and callable(getattr(backend,'poll_background',None)) and backend.poll_background():
                        self._publish(backend,last_count)
                    try:
                        count, color, depth, k, stamp = self.incoming.get(timeout=.2)
                    except queue.Empty:
                        continue
                    started = time.perf_counter()
                    result = self._process_frame(backend,count,color,depth,k,stamp)
                    elapsed = (time.perf_counter()-started)*1000
                    self.processed += 1
                    times = getattr(backend, 'timings', np.zeros(5))
                    metrics.write(f'{count},{self.processed},{result},{elapsed:.3f},{self.dropped},'+','.join(f'{v:.3f}' for v in times)+'\n')
                    metrics.flush()
                    self.events.put(('tracking', (count, self.processed, result, elapsed, self.dropped)))
                    self.events.put(('backend_times', times.copy()))
                    if result == 0:
                        with (self.folder/'tracking_rejections.log').open('a', encoding='utf-8') as rejected:
                            rejected.write(f'frame {count}: {backend.error()}\n')
                    last_count = count
                    if result > 0 and getattr(self,'_should_publish',lambda r:True)(result) and (getattr(self, '_publish_every_capture', lambda: False)() or time.monotonic()-last_preview >= .5):
                        self._publish(backend, count)
                        last_preview = time.monotonic()
                    self._after_process(backend,count,result)
            if self.failed:
                return
            if callable(getattr(self,'_finish_processing',None)):self._finish_processing()
            if len(self.names) < self.minimum_frames:
                backend.close()
                backend = None
                self.events.put(('empty', None))
                return
            if callable(getattr(backend,'finish_background',None)):
                self.events.put(('warning','正在收尾后台回环，等待预算5秒；地图提交可能额外耗时，未完成任务将取消'))
                backend.finish_background()
            points, nodes = getattr(backend,'final_preview',backend.preview)()
            if not len(nodes):
                raise RuntimeError('尚未接受可靠拍摄，未生成地图；原始数据已保留')
            self._before_close(backend,points,nodes)
            backend.close()
            backend = None
            self.events.put(('final', (last_count, points, nodes)))
        except Exception as exc:
            self.failed = True
            self.stop.set()
            self.events.put(('error', f'建图失败：{exc}\n恢复数据：{self.folder}'))
        finally:
            if callable(getattr(self,'_finish_processing',None)):
                try:self._finish_processing()
                except Exception as exc:self.events.put(('warning',f'后台点云处理失败：{exc}'))
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    self.events.put(('warning', f'地图数据库关闭失败：{exc}'))

    def discard(self):
        target = self.folder.resolve()
        parent = (ROOT / 'results').resolve()
        if target.parent != parent or not target.name.startswith('.pending_scan_'):
            raise ValueError('Refusing to remove a folder outside this scan session')
        if target.exists():
            shutil.rmtree(target)

    def _after_process(self,backend,count,result):
        pass

    def _process_frame(self,backend,count,color,depth,k,stamp):
        return backend.process(color,depth,k,stamp)

    def _before_close(self,backend,points,nodes):
        pass

    def _finalize_saved(self, folder):
        pass

    def save(self, dataset=None):
        if (self.folder/'map.db').exists() and not getattr(self,'skip_native_cloud_export',False):
            export_map(self.folder,getattr(self,'export_voxel_size',.01),getattr(self,'export_decimation',4))
        target = stage_folder(dataset,1) if dataset else stage_result_path(1)
        parent = (ROOT/'results').resolve()
        if self.folder.resolve().parent != parent or not self.folder.name.startswith('.pending_scan_'):
            raise ValueError('Unsafe scan save paths')
        if dataset:
            with replace_stage(target,1,dataset) as temporary:
                shutil.copytree(self.folder,temporary)
                write_configuration(temporary,self.k,self.shape,getattr(self,'source_fps',10))
                ini=temporary/'source.ini'
                ini.write_text(ini.read_text(encoding='utf-8').replace(temporary.as_posix(),target.as_posix()),encoding='utf-8')
                self._finalize_saved(temporary)
            self.discard()
            self.folder=target
        else:
            if target.resolve().parent != parent or target.exists():raise ValueError('Unsafe scan target')
            shutil.move(str(self.folder),str(target));self.folder=target
            write_configuration(target,self.k,self.shape,getattr(self,'source_fps',10))
            self._finalize_saved(target)
        return target


class ScanWindow:
    def __init__(self, root, *, embedded=False, on_preprocess=None):
        import tkinter as tk
        from tkinter import ttk
        self.root = root
        self.on_preprocess = on_preprocess
        self.closed = False
        cv2.setNumThreads(1)
        if not embedded:
            root.title('HP60C 环境扫描 · RTAB-Map')
            root.geometry('1450x850')
        self.events = queue.Queue()
        self.session = None
        self.busy = False
        self.importing = False
        self.has_map = False
        self.close_pending = False
        self.tracking_lost = False
        self.stage_times = {'odometry': 0., 'graph': 0., 'fusion': 0., 'draw': 0.}
        self.draw_points = 0
        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='开始扫描', command=self.start)
        self.start_button.pack(side='left', padx=5)
        self.finish_button = ttk.Button(controls, text='完成扫描', command=self.finish, state='disabled')
        self.finish_button.pack(side='left', padx=5)
        self.import_button = ttk.Button(controls, text='导入点云', command=self.import_cloud)
        self.import_button.pack(side='left', padx=5)
        self.status = tk.StringVar(value='就绪 · 点击开始扫描后连接相机')
        ttk.Label(controls, textvariable=self.status, wraplength=620).pack(side='left', padx=15)
        options = ttk.Frame(root, padding=(15, 0, 15, 5))
        options.pack(fill='x')
        self.import_unit = tk.StringVar(value='自动')
        self.import_coordinates = tk.StringVar(value='自动')
        self.preprocess_button = ttk.Button(options, text='点云预处理', command=self.open_preprocessing)
        self.preprocess_button.pack(side='right', padx=5)
        panels = ttk.Panedwindow(root, orient='horizontal')
        panels.pack(fill='both', expand=True)
        left = self.map_frame = ttk.Labelframe(panels, text='拍摄融合地图 · 拍摄成功后更新')
        right = ttk.Frame(panels)
        panels.add(left, weight=1)
        panels.add(right, weight=1)
        right_panels = ttk.Panedwindow(right, orient='vertical')
        right_panels.pack(fill='both', expand=True)
        rgb_frame = ttk.Labelframe(right_panels, text='实时 RGB')
        cloud_frame = ttk.Labelframe(right_panels, text='实时点云 · 未加入地图')
        right_panels.add(rgb_frame, weight=1)
        right_panels.add(cloud_frame, weight=2)
        from rgb_preview import RGBPreviewPanel
        self.rgb_panel = RGBPreviewPanel(rgb_frame)
        from live_cloud_view import LiveCloudPanel
        from map_cloud_view import MapCloudPanel
        self.live_panel = LiveCloudPanel(cloud_frame)
        self.map_panel = MapCloudPanel(left,on_render=self._map_rendered)
        self.performance = tk.StringVar(value='耗时：跟踪 / 图优化 / 空间融合 / 绘图（ms）')
        ttk.Label(root, textvariable=self.performance, padding=5).pack(fill='x')
        # Session lifecycle is handled by buttons/window close, not pyplot shortcuts.
        if not embedded:
            root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(100, self.poll)
        root.bind('<Destroy>', self._destroyed, add='+')

    def _destroyed(self, event):
        if event.widget is self.root:
            self.closed = True
            if self.timer:self.root.after_cancel(self.timer)

    def start(self):
        if self.busy or self.importing:
            return
        self.session = ScanSession(self.events)
        self.busy, self.has_map = True, False
        self.tracking_lost = False
        self.map_panel.clear()
        self.live_panel.reset()
        self.rgb_panel.clear()
        self.start_button.configure(state='disabled')
        self.finish_button.configure(state='normal')
        self.status.set('正在连接 HP60C…')
        self.session.start()

    def import_cloud(self, path=None):
        from ui_dialogs import filedialog
        if self.busy or self.importing:
            return
        if path is None:
            path = filedialog.askopenfilename(parent=self.root, title='导入已有点云',
                initialdir=str(ROOT/'results'), filetypes=[('点云文件', '*.ply *.pcd *.npy *.npz *.xyz *.txt *.csv'), ('所有文件', '*.*')])
        if not path:
            return
        unit = {'自动':'auto', '米':'m', '毫米':'mm'}[self.import_unit.get()]
        coordinates = {'自动':'auto', '世界：X前 Y左 Z上':'world', '相机：X右 Y下 Z前':'camera'}[self.import_coordinates.get()]
        self.importing = True
        self.start_button.configure(state='disabled')
        self.import_button.configure(state='disabled')
        self.status.set('正在后台读取点云…')
        threading.Thread(target=self._read_import, args=(Path(path), unit, coordinates), daemon=True).start()

    def _read_import(self, path, unit, coordinates):
        from cloud_import import load_cloud
        try:
            self.events.put(('cloud_loaded', (path, load_cloud(path, unit, coordinates))))
        except Exception as exc:
            self.events.put(('cloud_load_error', str(exc)))

    def open_preprocessing(self):
        if self.busy or self.importing:
            return
        if self.on_preprocess is not None:
            source = getattr(self, 'imported_path', None)
            if self.session and (self.session.folder/'map_cloud.ply').is_file():
                source = self.session.folder/'map_cloud.ply'
            metadata = getattr(self, 'imported_metadata', None) if source == getattr(self, 'imported_path', None) else None
            self.on_preprocess(source, metadata)
            return
        import tkinter as tk
        from preprocess_window import PreprocessWindow
        window = getattr(self, 'preprocessing', None)
        if window is not None and not window.closed:
            window.root.lift()
            return
        source = getattr(self, 'imported_path', None)
        if self.session and (self.session.folder/'map_cloud.ply').is_file():
            source = self.session.folder/'map_cloud.ply'
        self.preprocessing = PreprocessWindow(tk.Toplevel(self.root), source)
        metadata = getattr(self, 'imported_metadata', None)
        if metadata and source == getattr(self, 'imported_path', None):
            self.preprocessing.unit.set({'m':'米', 'mm':'毫米'}[metadata[0]])
            self.preprocessing.axes.set({'world':'世界', 'camera':'相机'}[metadata[1]])

    def finish(self):
        if self.session and self.busy:
            self.session.stop.set()
            self.finish_button.configure(state='disabled')
            self.status.set('已请求停止采集 · 正在完成地图优化…')
        elif self.session and self.has_map and self.session.folder.exists() and self.session.folder.name.startswith('.pending_scan_'):
            self.request_save()

    def close(self):
        if self.busy:
            self.close_pending = True
            self.finish()
        else:
            self.root.destroy()

    def draw(self, points, count, nodes=None):
        self.draw_points = len(points)
        self.map_panel.set_cloud(points,nodes)

    def _map_rendered(self,elapsed,count):
        self.stage_times['draw'] = elapsed
        self._update_performance()
        now=time.monotonic()
        if self.session and self.session.folder.is_dir() and (now-getattr(self,'_last_render_log',0)>=.5 or count!=getattr(self,'_last_render_count',None)):
            self._last_render_log,self._last_render_count=now,count
            try:
                with (self.session.folder/'render_metrics.csv').open('a',encoding='utf-8') as log:
                    if log.tell() == 0:
                        log.write('draw_ms,display_points\n')
                    log.write(f'{elapsed:.3f},{count}\n')
            except OSError:
                pass

    def poll(self):
        if self.closed:
            return
        from ui_dialogs import messagebox
        latest_frame = None
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'cloud_loaded':
                path, cloud = payload
                self.imported_path = path
                self.imported_metadata = (cloud.unit, cloud.coordinates)
                self.importing = False
                self.session = None
                if hasattr(self,'reference_panel'):self.reference_panel.clear()
                self.stage_times = dict.fromkeys(self.stage_times, 0.)
                self.map_panel.clear()
                self.draw(cloud.points, 0, np.empty((0, 3)))
                self.has_map = True
                unit_label = '毫米' if cloud.unit == 'mm' else '米'
                axes_label = '相机坐标' if cloud.coordinates == 'camera' else '世界坐标'
                self.status.set(f'已导入 {path.name} · {len(cloud.points):,} 点 · 源文件{unit_label}/{axes_label} · 已过滤 {cloud.removed} 个无效点')
                self.start_button.configure(state='normal')
            elif kind == 'cloud_load_error':
                self.importing = False
                self.start_button.configure(state='normal')
                self.status.set('导入失败 · 原显示保留')
                messagebox.showerror('导入失败', payload, parent=self.root)
            elif kind == 'frame':
                latest_frame = payload
                if not hasattr(self.session,'request_capture') and not self.session.stop.is_set() and not self.has_map and not self.tracking_lost:
                    self.status.set(f'扫描中 · 已采集 {payload[0]} 帧 · 在线增量跟踪')
            elif kind == 'video_stats':
                self.video_stats=payload
            elif kind == 'tracking':
                count, processed, result, elapsed, dropped = payload
                self.tracking_lost = result == 0
                label = '跟踪丢失，请缓慢返回最近成功视野' if result == 0 else '在线跟踪'
                self.status.set(f'{label} · 采集 {count} / 处理 {processed} 帧 · {elapsed:.0f}ms · 跳过过期帧 {dropped}')
            elif kind == 'capture_ready':
                self.capture_button.configure(state='normal')
                self.status.set('相机就绪 · 自动采样中，请缓慢移动并保持重叠' if getattr(self.session,'video_mode',False) else '相机就绪 · 请保持稳定，点击拍摄当前位置')
            elif kind == 'capture_matching':
                self.status.set(f'第 {payload} 次拍摄 · 正在寻找重叠位置并精配准…')
            elif kind == 'local_preparing':
                self.status.set(f'第 {payload} 次拍摄 · 局部清洗与表面估计…')
            elif kind == 'tracking_reference':
                if hasattr(self,'reference_panel'):self.reference_panel.set_reference(payload)
            elif kind == 'capture_result':
                count,report = payload
                if not self.session.stop.is_set():
                    self.capture_button.configure(state='normal')
                self.status.set(capture_result_text(count,report))
                if hasattr(self,'reference_panel'):self.reference_panel.set_tracking(report.get('tracking') or {})
            elif kind == 'backend_times':
                self.stage_times['odometry'], self.stage_times['graph'] = payload[:2]
                self._update_performance()
            elif kind == 'fusion_time':
                self.stage_times['fusion'] = payload
                self._update_performance()
            elif kind in ('map', 'final'):
                count, points, nodes = payload
                self.has_map = True
                self.draw(points, count, nodes)
                self.status.set(f'地图已更新至第 {count} 帧 · {len(nodes)} 个扫描节点 · {len(points):,} 点')
                if kind == 'final':
                    self.busy = False
                    self.finish_button.configure(state='disabled')
                    try:
                        if getattr(self,'beta',False) or messagebox.askyesno('扫描完成', '是否保存此次扫描？\n将保存融合点云、轨迹、数据库和原始 RGB-D 数据。', parent=self.root):
                            self.request_save()
                        else:
                            self.session.discard()
                            self.status.set('扫描完成 · 未保存，临时数据已清理')
                    except Exception as exc:
                        messagebox.showerror('保存/清理失败', f'{exc}\n数据位置：{self.session.folder}', parent=self.root)
                    if not self.busy:
                        self.start_button.configure(state='normal')
            elif kind == 'saved':
                self.busy = False
                self.start_button.configure(state='normal')
                self.status.set(f'扫描已保存：{payload}')
                if getattr(self,'on_stage_saved',None):self.on_stage_saved(1,payload)
                if not getattr(self,'beta',False):messagebox.showinfo('已保存', str(payload), parent=self.root)
            elif kind == 'save_error':
                self.busy = False
                self.start_button.configure(state='normal')
                messagebox.showerror('保存失败', payload, parent=self.root)
            elif kind in ('mapping', 'finishing'):
                self.status.set(f'正在构建前 {payload} 帧的融合地图…' + ('采集继续' if kind == 'mapping' else ''))
            elif kind == 'warning':
                self.status.set(payload)
            elif kind == 'error':
                self.session.stop.set()
                self.finish_button.configure(state='disabled')
                messagebox.showerror('扫描失败', payload, parent=self.root)
            elif kind == 'empty':
                self.busy = False
                self.session.discard()
                self.finish_button.configure(state='disabled')
                self.start_button.configure(state='normal')
                self.status.set('有效帧不足，未生成地图')
        if self.session and getattr(self.session, 'live_rgb', None) is not None:
            rgb = self.session.live_rgb
            self.session.live_rgb = None
            self.rgb_panel.update(rgb)
        if self.session and getattr(self.session, 'live_frame', None) is not None:
            live = self.session.live_frame
            self.session.live_frame = None
            intrinsics = getattr(self.session,'live_intrinsics',None)
            if intrinsics is not None:
                self.live_panel.intrinsics = intrinsics
            self.live_panel.update(live)
        elif latest_frame and latest_frame[1] is not None:
            self.live_panel.update(latest_frame[1])
        if self.session and self.session.failed and not self.session.capture_thread.is_alive() and not self.session.map_thread.is_alive():
            self.busy = False
            self.start_button.configure(state='normal')
        if self.close_pending and not self.busy:
            self.root.destroy()
            return
        self.import_button.configure(state='disabled' if self.busy or self.importing else 'normal')
        self.preprocess_button.configure(state='disabled' if self.busy or self.importing else 'normal')
        if not self.busy:
            pending=self.session and not self.session.failed and self.has_map and self.session.folder.exists() and self.session.folder.name.startswith('.pending_scan_')
            self.finish_button.configure(state='normal' if pending and not self.importing else 'disabled')
        self.timer = self.root.after(30, self.poll)

    def request_save(self):
        if self.busy or self.importing or self.session is None or not self.has_map:return
        if not getattr(self,'can_write',lambda:True)():return
        if not self.session.folder.name.startswith('.pending_scan_'):
            self.status.set(f'扫描已保存：{self.session.folder}');return
        from ui_dialogs import simpledialog,messagebox
        dataset=getattr(self,'dataset',None)
        if dataset is None:
            name=simpledialog.askstring('保存扫描','请设置数据名称（例如 test），所有阶段将存入 results/此名称',parent=self.root)
            if name is None:
                self.status.set('尚未保存；临时扫描保留，可点击“完成”继续保存');return
            try:dataset=create_dataset(ROOT/'results',name)
            except Exception as exc:
                messagebox.showerror('数据名称不可用',str(exc),parent=self.root);return
            self.dataset=dataset
        self.busy=True;self.start_button.configure(state='disabled')
        self.status.set('正在保存扫描到数据集…')
        threading.Thread(target=self._save,daemon=True).start()

    def _save(self):
        try:
            self.events.put(('saved', self.session.save(dataset=self.dataset) if getattr(self,'dataset',None) else self.session.save()))
        except Exception as exc:
            self.events.put(('save_error', f'{exc}\n恢复数据：{self.session.folder}'))

    def _update_performance(self):
        t = self.stage_times
        if self.session is None:
            self.performance.set(f'点云查看 · {self.draw_points:,} 点 · 左侧绘图 {t["draw"]:.0f} ms · 点击左侧后 WASD 移动 / QE 升降 / 按住左键转向 / R 复位')
            return
        voxel_mm = getattr(self.session,'export_voxel_size',.01)*1000 if self.session else 10
        registration = 'RGB-D视觉里程计' if getattr(self.session,'video_mode',False) else '几何图匹配+ICP' if hasattr(self.session,'request_capture') else '跟踪'
        self.performance.set(f'耗时(ms) · {registration} {t["odometry"]:.0f} | 图优化 {t["graph"]:.0f} | '
                             f'空间融合 {t["fusion"]:.0f} | 左侧绘图 {t["draw"]:.0f} · {voxel_mm:g}mm体素，每格1点')


def launch():
    import tkinter as tk
    root = tk.Tk()
    ScanWindow(root)
    root.mainloop()


def run_timed(seconds, fps=10):
    events = queue.Queue()
    session = ScanSession(events, fps=fps)
    session.start()
    started = None
    try:
        while session.map_thread.is_alive() or not events.empty():
            if started is not None and time.monotonic()-started >= seconds:
                session.stop.set()
            try:
                kind, payload = events.get(timeout=.1)
            except queue.Empty:
                continue
            if kind == 'frame':
                if started is None:
                    started = time.monotonic()
            elif kind in ('map', 'final'):
                print(f'Online map: capture frame {payload[0]}, {len(payload[2])} nodes, {len(payload[1])} display points', flush=True)
            elif kind == 'error':
                raise RuntimeError(payload)
    except KeyboardInterrupt:
        session.stop.set()
        session.capture_thread.join()
        session.map_thread.join()
    except Exception:
        session.stop.set()
        session.capture_thread.join()
        session.map_thread.join()
        raise
    finally:
        session.stop.set()
    session.capture_thread.join()
    session.map_thread.join()
    if session.failed:
        raise RuntimeError(f'Online scan failed; recovery data: {session.folder}')
    print('Saved:', session.save(), flush=True)


if __name__ == '__main__':
    launch()
