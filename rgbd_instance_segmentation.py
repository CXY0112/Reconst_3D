"""RGB instances lifted to immutable scene points; prompted editing only."""
from pathlib import Path
import copy, json, os, queue, subprocess, sys, tempfile, threading, time, uuid
import cv2
import numpy as np
from preprocess import _check
from scene_segmentation import ROOT,SegmentedScene,fingerprint

def worker_command(control):
    script=ROOT/'tools/mobilesam_worker.py'
    explicit=os.environ.get('RECONST_MOBILESAM_PYTHON')
    if explicit:return [explicit,str(script),'--control',str(control)],False
    marker=ROOT/'vendor/mobilesam/runtime.json'
    if os.name=='nt' and marker.is_file():
        from nksr_runtime import wsl_path
        info=json.loads(marker.read_text(encoding='utf-8'));distribution=info['distribution'];python=info['python']
        if not distribution or not all(c.isalnum() or c in '._-' for c in distribution):raise ValueError('无效WSL发行版')
        if not python.startswith('/') or any(c in python for c in '\n\r\0'):raise ValueError('无效Python路径')
        return ['wsl.exe','--distribution',distribution,'--exec','env','OMP_NUM_THREADS=4',
            'PYTHONUNBUFFERED=1',python,wsl_path(script),'--control',wsl_path(control)],True
    return [sys.executable,str(script),'--control',str(control)],False


class MobileSamClient:
    """One isolated model process reused across box prompts."""
    def __init__(self):
        self.process=None;self.folder=None;self.temp=None;self.messages=queue.Queue();self.lock=threading.Lock();self.closed=False

    def _start(self):
        if self.closed:raise RuntimeError('MobileSAM窗口已经关闭')
        if self.process is not None and self.process.poll() is None:return
        self._stop()
        self.temp=tempfile.TemporaryDirectory(prefix='.mobilesam_',dir=ROOT/'results');self.folder=Path(self.temp.name)
        command,self.wsl=worker_command(self.folder);self.messages=queue.Queue();messages=self.messages
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        process=self.process
        def read():
            for line in process.stdout:messages.put(line)
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()

    def request(self,image,mode,box=None,cancel=None):
        with self.lock:
            _check(cancel);self._start();ident=uuid.uuid4().hex;output=self.folder/(ident+'.npz')
            def path(p):
                if self.wsl:
                    from nksr_runtime import wsl_path
                    return wsl_path(p)
                return str(p)
            request=dict(id=ident,image=path(image),mode=mode,output=path(output),box=box)
            self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
            tail=[]
            try:
                while True:
                    if self.closed or (cancel is not None and cancel.is_set()):
                        self._stop();_check(cancel);raise RuntimeError('MobileSAM已停止')
                    try:line=self.messages.get(timeout=.1)
                    except queue.Empty:
                        if self.process.poll() is not None:raise RuntimeError('MobileSAM运行失败：'+''.join(tail)[-3000:])
                        continue
                    tail.append(line);tail=tail[-30:]
                    try:message=json.loads(line)
                    except (ValueError,TypeError):continue
                    if message.get('id')!=ident:continue
                    if not message.get('ok'):raise RuntimeError(message.get('error','MobileSAM推理失败'))
                    with np.load(output,allow_pickle=False) as data:masks=data['masks'].copy();scores=data['scores'].copy()
                    return masks,scores,message
            finally:
                if output.exists():output.unlink()

    def _stop(self):
        if self.folder and self.folder.exists():(self.folder/'stop').touch(exist_ok=True)
        if self.process is not None:
            if self.process.poll() is None:
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            for stream in (self.process.stdin,self.process.stdout):
                if stream:
                    try:stream.close()
                    except OSError:pass
            if hasattr(self,'thread'):self.thread.join(timeout=1)
        self.process=None
        if self.temp:self.temp.cleanup()
        self.temp=None;self.folder=None

    def close(self):
        self.closed=True
        # Can be called while request() owns the lock. The request thread will
        # stop and reap its worker; don't race its temporary-file cleanup.
        if self.lock.acquire(blocking=False):
            try:self._stop()
            finally:self.lock.release()
        elif self.folder and self.folder.exists():(self.folder/'stop').touch(exist_ok=True)


def source_frames(scene):
    from tsdf_reconstruction import sdk_frames
    source=scene.source
    session=None
    # Prefer the moved dataset's relative reference over old absolute paths.
    if (source/'geometry_poses.json').is_file():session=source
    else:
        report=json.loads((source/'report.json').read_text(encoding='utf-8'))
        if report.get('source_session_relative'):
            candidate=(source/report['source_session_relative'].replace('\\','/')).resolve()
            if (candidate/'geometry_poses.json').is_file():session=candidate
        if session is None:
            for parent in (source.parent,source.parent.parent):
                candidate=parent/'setp1'
                if (candidate/'geometry_poses.json').is_file():session=candidate;break
        if session is None:
            from saved_sources import reference
            session=reference(report.get('source_session'),source)
    if session is None:raise ValueError('框选/自动RGB分割需要同一数据集的原始RGB-D、内参和相机位姿')
    frames,k,_=sdk_frames(session)
    for frame in frames:
        frame['rgb']=session/'rgb_raw'/frame['depth'].name
        if not frame['rgb'].is_file():raise ValueError('扫描缺少配准后的原始RGB照片')
    return frames,k


def observation_fingerprint(scene):
    frames,_=source_frames(scene)
    if not frames:raise ValueError('没有可用于分割的有效RGB-D观测')
    return fingerprint(frames[0]['depth'].parent.parent)


def project_visible(points,frame,k,depth_gate=.030):
    depth=cv2.imread(str(frame['depth']),cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim!=2:raise ValueError('无效SDK深度图')
    h,w=depth.shape;local=(points-frame['pose'][:3,3])@frame['pose'][:3,:3];z=local[:,2]
    uv=np.rint(local[:,:2]/np.maximum(z,.001)[:,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(np.int32)
    good=(z>=.15)&(z<=4)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids=np.flatnonzero(good);raw=depth[uv[ids,1],uv[ids,0]]*.001
    ids=ids[(raw>=.15)&(raw<=4)&(abs(raw-z[ids])<=depth_gate)]
    return ids,uv[ids],depth.shape


def lift_mask(mask,ids,uv,shape,erode_pixels=2):
    mask=np.asarray(mask,bool)
    if mask.shape!=tuple(shape):raise ValueError('RGB掩码和SDK深度图分辨率不一致，不能映射')
    if erode_pixels:
        mask=cv2.erode(mask.astype(np.uint8),np.ones((2*erode_pixels+1,)*2,np.uint8)).astype(bool)
    return ids[mask[uv[:,1],uv[:,0]]]


def prompt_rgbd(scene,client,frame,k,box,cancel=None):
    _check(cancel)
    before=observation_fingerprint(scene)
    masks,scores,timing=client.request(frame['rgb'],'box',box=list(map(float,box)),cancel=cancel)
    if masks.shape[0]!=1:raise ValueError('无效框选掩码')
    ids,uv,shape=project_visible(scene.points_world,frame,k)
    selected=lift_mask(masks[0],ids,uv,shape)
    if len(selected)<10:raise ValueError('该框选区域没有足够的深度一致点；请更换照片或框选范围')
    _check(cancel)
    if observation_fingerprint(scene)!=before:raise ValueError('框选期间RGB-D或位姿已变化，请重新读取')
    return dict(indices=selected,mask=masks[0],capture=int(frame['capture']),stamp=frame['depth'].stem,
                box=list(map(float,box)),score=float(scores[0]),timing=timing,observation_fingerprint=before)


def apply_prompt(scene,proposal,action='new',target=None):
    """Copy-on-write label edit; geometry/RGB untouched. Caller retains undo state."""
    indices=np.asarray(proposal['indices'],np.int64)
    if indices.ndim!=1 or not len(indices) or indices.min()<0 or indices.max()>=len(scene.rgb):raise ValueError('无效框选点索引')
    result=copy.copy(scene);result.report=copy.deepcopy(scene.report)
    before=proposal.get('observation_fingerprint')
    previous=scene.report.get('rgbd_observation_fingerprint')
    if before and previous and before!=previous:raise ValueError('原始观测已更新，不能与之前分割混用，请重新分割')
    if before:result.report['rgbd_observation_fingerprint']=before
    result.report.setdefault('backend','rgbd_box_only')
    result.report.setdefault('device',proposal.get('timing',{}).get('device','cpu'))
    result.instance=scene.instance.copy() if scene.instance is not None else np.zeros(len(scene.rgb),np.int32)
    result.semantic=scene.semantic.copy() if scene.semantic is not None else np.full(len(scene.rgb),19,np.int32)
    unassigned=result.report.get('unassigned_instance_id')
    if scene.instance is None:unassigned=0;result.report['unassigned_instance_id']=0
    if action in ('add','remove'):
        if target is None or target==unassigned or not np.any(result.instance==target):raise ValueError('请先选择一个已有物体')
    if action=='new':
        ident=int(result.instance.max())+1;result.instance[indices]=ident;result.semantic[indices]=19
    elif action=='add':
        ident=int(target);result.instance[indices]=ident
        result.semantic[indices]=int(np.bincount(scene.semantic[scene.instance==target]).argmax())
    elif action=='remove':
        if unassigned is None:
            unassigned=int(result.instance.max())+1;result.report['unassigned_instance_id']=unassigned
        indices=indices[result.instance[indices]==target]
        if not len(indices):raise ValueError('框选结果与选中物体没有重合')
        result.instance[indices]=unassigned;result.semantic[indices]=19;ident=int(target)
    else:raise ValueError('未知框选修改方式')
    result.report.setdefault('prompt_edits',[]).append(dict(action=action,target=ident,
        capture=int(proposal['capture']),stamp=proposal['stamp'],box=proposal['box'],affected_points=len(indices)))
    result.report['complete']=True
    return result,ident
