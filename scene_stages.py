"""Versioned RGB-D stage locations; older datasets remain read-only compatible."""
from pathlib import Path
import json


def read_report(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def source_for(dataset):
    root=Path(dataset);pre=root/'rgbd_scene/setp2'
    stages=read_report(root/'dataset.json').get('branches',{}).get('rgbd_scene',{}).get('stages',{})
    if stages.get('2',{}).get('stale'):return root/'setp1'
    return pre if (pre/'report.json').is_file() else root/'setp1'


def ply_count(path):
    try:
        with Path(path).open('rb') as stream:
            for _ in range(200):
                line=stream.readline()
                if line.startswith(b'element vertex '):return int(line.split()[-1])
                if line.strip()==b'end_header':break
    except (OSError,ValueError):pass
    return None


MESH_VIEW='几何结构'
SPLAT_VIEW='3DGS 三维效果'


def result_kind(report):
    return 'gaussian' if report.get('representation')=='gaussian_splats' or report.get('method')=='brush_3dgs' else 'mesh'


def result_target(stage_root,kind):
    if kind not in ('mesh','gaussian'):raise ValueError('未知场景结果类型')
    return Path(stage_root)/('brush_3dgs' if kind=='gaussian' else 'mesh')


def results_for_stage(stage_root):
    stage_root=Path(stage_root);results={}
    for path in (result_target(stage_root,'mesh'),result_target(stage_root,'gaussian'),
                 stage_root/'base',stage_root.parent/'setp4/base'):
        report=read_report(path/'report.json')
        if report.get('schema')=='rgbd_scene_base_v1':results.setdefault(result_kind(report),path)
    return results


def bases_for(dataset):return results_for_stage(Path(dataset)/'rgbd_scene/setp3')


def preferred_kind(stage_root,available):
    saved=read_report(Path(stage_root)/'active_result.json').get('kind')
    return saved if isinstance(saved,str) and saved in available else next((kind for kind in ('mesh','gaussian') if kind in available),None)


def base_for(dataset):
    stage_root=Path(dataset)/'rgbd_scene/setp3';available=bases_for(dataset)
    kind=preferred_kind(stage_root,available)
    return available[kind] if kind else result_target(stage_root,'mesh')


def segmentation_for(dataset):
    root=Path(dataset)/'rgbd_scene'
    for path in (root/'setp4/segmentation',root/'setp3'):
        if (path/'labels.npz').is_file():return path
    return root/'setp4/segmentation'


def current_base(dataset,source_fingerprint):
    path=base_for(dataset);report=read_report(path/'report.json')
    return path if report.get('complete') and report.get('source_fingerprint')==source_fingerprint else None


def geometry_model(folder,report):
    """Prefer the original geometry in legacy colored datasets as well."""
    from brush_reconstruction import is_splats
    if is_splats(report):raise ValueError('该结果是 3DGS 高斯模型，不是三角网格')
    from scene_reconstruction import result_path
    for relative in (report.get('geometry_relative','geometry/mesh.ply'),report.get('model_relative','geometry/mesh.ply')):
        path=result_path(folder,relative)
        if path.is_file():return path
    raise FileNotFoundError('该数据集缺少几何网格文件')


def base_model(folder,report):
    from brush_reconstruction import is_splats,validate_result
    return validate_result(folder,report) if is_splats(report) else geometry_model(folder,report)


def object_models(dataset,scene):
    """Local completed GLBs only; never query the cloud while opening showcase."""
    root=Path(dataset)/'rgbd_scene/setp4/fal';models=[]
    if not root.is_dir():return models
    names={o['id']:o['name'] for o in scene.objects()}
    for state_path in sorted(root.glob('object_*/*/run_state.json'),reverse=True):
        state=read_report(state_path);path=state_path.parent/'generated.glb'
        if (state.get('schema')!='fal_object_run_v1' or state.get('status')!='downloaded'
            or state.get('source_fingerprint')!=scene.report['source_fingerprint'] or not path.is_file()
            or root.resolve() not in path.resolve().parents):continue
        ident=state.get('object_id');name=state.get('object_name') or names.get(ident,f'物体 {ident}')
        models.append(dict(name=f'#{ident} {name} · {state_path.parent.name}',path=path,state=state))
    return models
