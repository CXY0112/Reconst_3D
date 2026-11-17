"""Non-destructive full-resolution instance labels on the colored stage-2 scene."""
from dataclasses import dataclass
from pathlib import Path
import colorsys, hashlib, json, os, shutil, uuid
import numpy as np
from preprocess import ProcessingCancelled, _check
from scene_cloud_processing import write_colored_ply,colorize,session_for

ROOT=Path(__file__).resolve().parent
CLASS_NAMES=('墙面','地面','柜子','床','椅子','沙发','桌子','门','窗','书架','图片','台面','书桌','窗帘','冰箱','浴帘','马桶','洗手池','浴缸','其他物体')


def fingerprint(folder):
    from scene_provenance import point_source_fingerprint
    return point_source_fingerprint(stage2_folder(folder))


def stage2_folder(source):
    source=Path(source).resolve()
    if source.is_file():source=source.parent
    for folder in (source/'rgbd_scene/setp2',source/'setp2',source):
        if (folder/'report.json').is_file():
            report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
            if report.get('complete') and report.get('skipped') and report.get('scene_rgb_schema')=='colored_scene_skip_v1':
                session=(folder/report['source_session_relative']).resolve()
                if not (session/'map_cloud.ply').is_file():raise ValueError('找不到跳过记录所引用的一阶段点云，请保留同一数据集的setp1')
                return session_for(session)
        if all((folder/name).is_file() for name in ('processed_world.npy','surfel_attributes.npz','report.json')):
            return folder
    for folder in (source/'setp1',source):
        if (folder/'map_cloud.ply').is_file() and (folder/'geometry_poses.json').is_file():return session_for(folder)
    raise ValueError('请选择一阶段完整扫描或二阶段完整场景彩色点云')


@dataclass
class SegmentedScene:
    source: Path
    points_world: np.ndarray
    rgb: np.ndarray
    instance: np.ndarray | None
    semantic: np.ndarray | None
    report: dict

    @property
    def points(self):
        p=self.points_world
        return np.column_stack((-p[:,1],-p[:,2],p[:,0])).astype(np.float32)

    def objects(self):
        if self.instance is None:return []
        objects=[]
        for ident,count in zip(*np.unique(self.instance,return_counts=True)):
            if ident==self.report.get('unassigned_instance_id'):continue
            selected=self.instance==ident;labels,counts=np.unique(self.semantic[selected],return_counts=True)
            label=int(labels[counts.argmax()]);points=self.points_world[selected]
            objects.append(dict(id=int(ident),points=int(count),class_id=label,
                name=self.report.get('object_names',{}).get(str(int(ident)),CLASS_NAMES[label] if 0<=label<len(CLASS_NAMES) else '未分类'),
                structural=label in (0,1),bounds_world_m=[points.min(0).tolist(),points.max(0).tolist()]))
        return sorted(objects,key=lambda x:-x['points'])


def load_scene(source,progress=None,cancel=None):
    _check(cancel)
    folder=stage2_folder(source)
    if not (folder/'processed_world.npy').is_file():
        from cloud_import import load_cloud
        before=fingerprint(folder)
        loaded=load_cloud(folder/'map_cloud.ply')
        _check(cancel)
        rgb,_,color_report=colorize(loaded.points,folder,progress,cancel)
        points=np.column_stack((loaded.points[:,2],-loaded.points[:,0],-loaded.points[:,1]))
        if fingerprint(folder)!=before:raise ValueError('一阶段扫描数据仍在变化，请完成并保存扫描后再分割')
        return SegmentedScene(folder,points,rgb,None,None,dict(source_fingerprint=before,
            input_kind='stage1',preprocessing_skipped=True,color=color_report))
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if not report.get('complete') or report.get('scene_rgb_schema')!='colored_scene_v1':
        raise ValueError('需要彩色场景分支二阶段的完整结果，请在 main.py rgbd 中完成预处理')
    points=np.load(folder/'processed_world.npy',allow_pickle=False).astype(np.float32)
    with np.load(folder/'surfel_attributes.npz',allow_pickle=False) as attr:rgb=attr['rgb'].copy()
    if points.ndim!=2 or points.shape[1]!=3 or rgb.shape!=points.shape or not len(points) or not np.isfinite(points).all():
        raise ValueError('二阶段 XYZ/RGB 点数或形状不一致')
    return SegmentedScene(folder,points,rgb,None,None,dict(source_fingerprint=fingerprint(folder),input_kind='stage2',preprocessing_skipped=False))


def instance_colors(labels):
    labels=np.asarray(labels,dtype=np.int64)
    unique,inverse=np.unique(labels,return_inverse=True)
    palette=np.array([[160,160,160] if i<0 else [round(x*255) for x in colorsys.hsv_to_rgb((int(i)*.61803398875)%1,.65,.9)] for i in unique],np.uint8)
    return palette[inverse]


def save_segmentation(scene,target):
    target=Path(target).resolve();source=scene.source.resolve()
    if target==source or target in source.parents or source in target.parents:raise ValueError('不得覆盖二阶段源数据或其父目录')
    if scene.instance is None:raise ValueError('尚未运行分割')
    if fingerprint(source)!=scene.report['source_fingerprint']:raise ValueError('来源数据已变化，请重新分割（二阶段或原始扫描）')
    if scene.report.get('rgbd_observation_fingerprint'):
        from rgbd_instance_segmentation import observation_fingerprint
        if observation_fingerprint(scene)!=scene.report['rgbd_observation_fingerprint']:
            raise ValueError('原始RGB-D或位姿已变化，分割标签失效，请重新分割')
    if target.exists():
        try:owned=json.loads((target/'report.json').read_text(encoding='utf-8')).get('schema')=='scene_segmentation_v1'
        except (OSError,ValueError):owned=False
        if not owned:raise ValueError('目标包含其他数据，不能覆盖')
    token=uuid.uuid4().hex;staging=target.parent/('.pending_segmentation_'+token);backup=target.parent/('.previous_segmentation_'+token)
    target.parent.mkdir(parents=True,exist_ok=True);staging.mkdir()
    committed=False
    try:
        np.savez_compressed(staging/'labels.npz',instance=scene.instance,semantic=scene.semantic)
        report=dict(scene.report,schema='scene_segmentation_v1',complete=True,
            source_relative=os.path.relpath(source,target),source_absolute=str(source),
            input_points=len(scene.points_world),objects=len(scene.objects()),class_names=list(CLASS_NAMES),
            note='Predicted classes/instances, not ground truth. Full-resolution indices reference the saved source point cloud. No geometry changed.')
        (staging/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (staging/'objects.json').write_text(json.dumps(scene.objects(),ensure_ascii=False,indent=2),encoding='utf-8')
        if target.exists():target.rename(backup)
        try:staging.rename(target);committed=True
        except BaseException:
            if backup.exists():backup.rename(target)
            raise
        return target
    finally:
        for path in ((staging,backup) if committed else (staging,)):
            if path.exists() and path.resolve().parent==target.parent and path.name.endswith(token):shutil.rmtree(path)


def region_colors(scene,selected=(),pending=None,assigned=True):
    """Neutral context plus instance/proposal highlights; measured RGB is immutable."""
    colors=np.full((len(scene.points),3),160,np.uint8)
    if scene.instance is not None:
        valid=scene.instance!=scene.report.get('unassigned_instance_id',-1)
        if assigned:colors[valid]=instance_colors(scene.instance)[valid]
        if selected:colors[np.isin(scene.instance,selected)]=[184,238,213]
    if pending is not None:colors[np.asarray(pending,dtype=int)]=[246,190,93]
    return colors


def load_segmentation(folder,source=None,progress=None,cancel=None):
    folder=Path(folder).resolve();report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report.get('schema')!='scene_segmentation_v1' or not report.get('complete'):raise ValueError('无效的三阶段分割结果')
    candidate=Path(source) if source is not None else folder/report['source_relative']
    if fingerprint(candidate)!=report['source_fingerprint']:raise ValueError('来源已更新，原分割标签失效，请重新分割（二阶段或原始扫描）')
    scene=load_scene(candidate,progress,cancel)
    if scene.report['source_fingerprint']!=report['source_fingerprint']:raise ValueError('来源已更新，原分割标签失效，请重新分割（二阶段或原始扫描）')
    with np.load(folder/'labels.npz',allow_pickle=False) as data:
        instance=data['instance'].copy();semantic=data['semantic'].copy()
    if instance.shape!=(len(scene.rgb),) or semantic.shape!=instance.shape or instance.dtype.kind not in 'iu' or semantic.dtype.kind not in 'iu':raise ValueError('标签文件与点云数量不一致')
    if np.any(instance<0) or np.any((semantic<0)|(semantic>=len(CLASS_NAMES))):raise ValueError('标签超出有效范围')
    scene.instance=instance;scene.semantic=semantic;scene.report=report
    if report.get('rgbd_observation_fingerprint'):
        from rgbd_instance_segmentation import observation_fingerprint
        if observation_fingerprint(scene)!=report['rgbd_observation_fingerprint']:
            raise ValueError('原始RGB-D或位姿已变化，保存的分割标签已失效')
    return scene


def export_object(scene,ident,path):
    if int(ident)==scene.report.get('unassigned_instance_id'):raise ValueError('未分配区域不是独立物体')
    selected=scene.instance==int(ident)
    if not selected.any():raise ValueError('未选择有效物体')
    write_colored_ply(path,scene.points_world[selected],scene.rgb[selected])
    return int(selected.sum())
