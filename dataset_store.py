"""Named datasets and transactional replacement of their four stored stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import uuid

STAGES = {1:'setp1', 2:'setp2', 3:'setp3', 4:'setp4'}
REQUIRED = {1:('map_cloud.ply','frames.jsonl','geometry_poses.json','sdk_calibration.json','registrations.jsonl'),
            2:('processed_world.npy','surfel_attributes.npz','report.json'),
            3:('mesh.ply','report.json'), 4:('mesh_colored.ply','color_attributes.npz','report.json')}


def read_manifest(folder):
    try:return json.loads((Path(folder)/'dataset.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def write_manifest(folder, data):
    folder=Path(folder)
    temporary=folder/('.dataset_'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temporary,folder/'dataset.json')
    finally:
        temporary.unlink(missing_ok=True)


def create_dataset(parent, name):
    name=name.strip()
    if (not name or name in ('.','..') or name.endswith(('.', ' ')) or
        any(c in '<>:"/\\|?*' or ord(c)<32 for c in name) or
        name.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'{s}{i}' for s in ('COM','LPT') for i in range(1,10)]}):
        raise ValueError('请输入有效的数据名称，不能包含路径或系统保留字符')
    parent=Path(parent).resolve();parent.mkdir(parents=True,exist_ok=True)
    folder=parent/name
    folder.mkdir(exist_ok=False)
    try:write_manifest(folder,dict(schema_version=1,name=name,stages={}))
    except Exception:folder.rmdir();raise
    return folder


def dataset_for(path):
    if not path:return None
    path=Path(path).resolve()
    for candidate in (path,path.parent,path.parent.parent):
        if (candidate/'dataset.json').is_file():return candidate
    return None


def route_dataset(view, path):
    """Folder imports in any tab also accept the enclosing dataset."""
    path=Path(path)
    root=path if (path/'dataset.json').is_file() or any((path/name).is_dir() for name in STAGES.values()) else dataset_for(path)
    if root and (root==path or root!=getattr(view,'dataset',None)):
        callback=getattr(view,'on_import_dataset',None)
        if callback:
            callback(root)
            return True
    return False


def stage_folder(folder, stage):
    return Path(folder).resolve()/STAGES[stage]


def required_stage_files(path, stage):
    if stage==2:
        try:
            report=json.loads((Path(path)/'report.json').read_text(encoding='utf-8'))
            if report.get('complete') and report.get('skipped') and report.get('backend')=='skip':
                from saved_sources import reference
                session=reference(report.get('source_session'),path)
                if session is not None and (session/'map_cloud.ply').is_file():return ('report.json',)
        except (OSError,ValueError,TypeError):pass
    return REQUIRED[stage]


def inspect_dataset(folder):
    folder=Path(folder).resolve()
    if not (folder/'dataset.json').is_file() and not any((folder/name).is_dir() for name in STAGES.values()):
        raise ValueError('请选择包含dataset.json或setp1–setp4子目录的数据集目录')
    manifest=read_manifest(folder);states={}
    for stage,name in STAGES.items():
        path=folder/name
        complete=all((path/file).is_file() for file in required_stage_files(path,stage))
        skipped=False
        if complete and stage>1:
            try:
                report=json.loads((path/'report.json').read_text(encoding='utf-8'))
                complete=bool(report.get('complete'));skipped=bool(stage==2 and report.get('skipped'))
            except (OSError,ValueError):complete=False
        entry=manifest.get('stages',{}).get(str(stage),{})
        states[stage]=dict(path=path,complete=complete,skipped=skipped,stale=bool(entry.get('stale',False)),
                           state='需更新' if complete and entry.get('stale') else '已跳过' if complete and skipped else '已完成' if complete else '未完成' if path.exists() else '未生成')
    return states


def _remove_owned(path, parent):
    path=Path(path).resolve();parent=Path(parent).resolve()
    if path.parent!=parent or not path.name.startswith(('.pending_setp','.previous_setp')):
        raise ValueError('Unsafe transaction cleanup')
    if path.is_dir():shutil.rmtree(path)


@contextmanager
def replace_stage(target, stage, dataset=None):
    """Legacy destinations stay create-only. Dataset results replace only after success."""
    target=Path(target).resolve()
    dataset=Path(dataset).resolve() if dataset else dataset_for(target)
    if dataset is None:
        yield target
        return
    if target!=stage_folder(dataset,stage):raise ValueError('输出目录与当前数据集阶段不一致')
    token=uuid.uuid4().hex
    working=dataset/f'.pending_setp{stage}_{token}'
    previous=dataset/f'.previous_setp{stage}_{token}'
    try:
        yield working
        if not all((working/name).is_file() for name in required_stage_files(working,stage)):raise ValueError('阶段输出不完整，保留原有结果')
        if stage>1:
            report=json.loads((working/'report.json').read_text(encoding='utf-8'))
            if not report.get('complete'):raise ValueError('阶段未成功完成，保留原有结果')
        manifest=read_manifest(dataset) or dict(schema_version=1,name=dataset.name,stages={})
        stages=manifest.setdefault('stages',{})
        upstream_stale=any(stages.get(str(n),{}).get('stale',False) for n in range(1,stage))
        stages[str(stage)]=dict(folder=STAGES[stage],stale=upstream_stale,updated_at=datetime.now(timezone.utc).isoformat())
        for number in range(stage+1,5):
            if stage_folder(dataset,number).is_dir():
                manifest['stages'].setdefault(str(number),dict(folder=STAGES[number]))['stale']=True
        if target.exists():target.rename(previous)
        try:
            working.rename(target)
            write_manifest(dataset,manifest)
        except BaseException:
            if target.exists():target.rename(working)
            if previous.exists():previous.rename(target)
            raise
        _remove_owned(previous,dataset)
    finally:
        _remove_owned(working,dataset)
