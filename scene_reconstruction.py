"""Geometry reconstruction and isolated mesh operations shared by stages 3–5."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,queue,shutil,subprocess,tempfile,threading,uuid
import numpy as np
from preprocess import _check
from scene_segmentation import ROOT,load_scene,load_segmentation,fingerprint
from rgbd_instance_segmentation import observation_fingerprint
from runtime_paths import geometry_python


def write_json(path,data):
    path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def segmentation_digest(scene):
    if scene.instance is None:raise ValueError('请先在第四阶段框选物体并保存')
    h=hashlib.sha256(scene.report['source_fingerprint'].encode())
    h.update(np.ascontiguousarray(scene.instance).tobytes())
    return h.hexdigest()


def result_path(folder,relative):
    folder=Path(folder).resolve();path=(folder/relative).resolve()
    if folder not in path.parents:raise ValueError('模型路径超出结果目录')
    return path


def load_inputs(source,segmentation=None,cancel=None):
    scene=load_scene(source,cancel=cancel);warning=''
    if segmentation and (Path(segmentation)/'labels.npz').is_file():
        try:
            scene=load_segmentation(segmentation,scene.source,cancel=cancel)
            scene.report['_segmentation_file']=str(Path(segmentation)/'labels.npz')
            scene.report['_segmentation_file_sha256']=digest(scene.report['_segmentation_file'])
        except (ValueError,OSError) as exc:warning='分割标签不可用，请重新保存分割：'+str(exc)
    return scene,warning


@contextmanager
def replace_base(target):
    target=Path(target).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        report=json.loads((target/'report.json').read_text(encoding='utf-8'))
        if report.get('schema')!='rgbd_scene_base_v1':raise ValueError('目标不是本阶段的基础重建结果，不能覆盖')
    token=uuid.uuid4().hex;pending=target.parent/('.pending_base_'+token);backup=target.parent/('.previous_base_'+token)
    pending.mkdir()
    try:
        yield pending
        if not (pending/'report.json').is_file():raise ValueError('基础重建没有生成完整报告，保留旧结果')
        report=json.loads((pending/'report.json').read_text(encoding='utf-8'))
        if not report.get('complete') or not result_path(pending,report.get('model_relative','geometry/mesh.ply')).is_file():raise ValueError('基础重建输出不完整，保留旧结果')
        if report.get('representation')=='gaussian_splats':
            from brush_reconstruction import validate_result
            validate_result(pending,report)
        if target.exists():target.rename(backup)
        try:pending.rename(target)
        except BaseException:
            if backup.exists():backup.rename(target)
            raise
    finally:
        for p in (pending,backup):
            if p.exists() and p.resolve().parent==target.parent and p.name.endswith(token):shutil.rmtree(p)


def run_command(command,cancel=None,progress=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    _check(cancel)
    process=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    messages=queue.Queue();tail=[]
    def read():
        for line in process.stdout:messages.put(line)
    thread=threading.Thread(target=read,daemon=True);thread.start()
    try:
        while process.poll() is None or not messages.empty():
            _check(cancel)
            try:line=messages.get(timeout=.1)
            except queue.Empty:continue
            tail.append(line);tail=tail[-30:]
            if progress:progress(line.strip())
        _check(cancel)
        if process.returncode:raise RuntimeError('后台处理失败：\n'+''.join(tail)[-3000:])
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=10)
        thread.join(timeout=2);process.stdout.close()


def read_mesh(path,cancel=None):
    with tempfile.TemporaryDirectory(prefix='.scene4_preview_',dir=ROOT/'results') as tmp:
        packet=Path(tmp)/'mesh.npz'
        run_command([geometry_python(),ROOT/'tools/scene_reconstruction_worker.py','--read',path,'--packet',packet],cancel)
        with np.load(packet,allow_pickle=False) as data:
            return tuple(data[k].copy() for k in ('vertices','triangles','normals')),data['colors'].copy() if 'colors' in data else None


def base_reconstruction(scene,target,method='poisson',cancel=None,progress=None,smoothing_iterations=30):
    if method not in ('poisson','tsdf','nksr','brush_3dgs'):raise ValueError('未知基础重建方法')
    target=Path(target).resolve()
    if target==scene.source or target in scene.source.parents or scene.source in target.parents:raise ValueError('输出不能覆盖输入点云')
    if (target/'report.json').is_file():
        from scene_stages import result_kind
        old=json.loads((target/'report.json').read_text(encoding='utf-8'))
        kind='gaussian' if method=='brush_3dgs' else 'mesh'
        if result_kind(old)!=kind:raise ValueError('几何结构与 3DGS 必须保存到不同目录，已有结果已保留')
    if method=='brush_3dgs':
        from brush_reconstruction import session_for
        session=session_for(scene.source)
    else:
        from geometry_inputs import source_info
        session=source_info(scene.source)['session']
    with replace_base(target) as pending:
        if method=='brush_3dgs':
            from brush_reconstruction import reconstruct
            reconstruct(scene,pending,session,cancel,progress)
        else:
            run_command([geometry_python(),'-u',ROOT/'tools/scene_reconstruction_worker.py',
                '--source',scene.source,'--output',pending,'--method',method,
                *(['--session',session] if session else []),
                '--smoothing-iterations',smoothing_iterations],cancel,progress)
        if fingerprint(scene.source)!=scene.report['source_fingerprint']:raise ValueError('输入点云发生变化，旧结果保留')
        mesh=read_mesh(pending/'geometry/mesh.ply',cancel)[0] if method!='brush_3dgs' else None
        colors=None
        _check(cancel)
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    return target,report,mesh,colors
