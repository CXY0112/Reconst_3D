"""Stage-three Brush Gaussian splats; separate from the triangle-mesh contract."""
from pathlib import Path
import json,os,shutil,subprocess,time
import cv2
import numpy as np
from reconstruct import ROOT
from preprocess import _check

METHOD='brush_3dgs'
LABEL='Brush 3DGS'
DEFAULTS=dict(steps=8000,resolution=320,max_splats=30000,sh_degree=1,eval_every=250,eval_split_every=5,seed=42)


def executable():
    path=ROOT/'vendor/brush/v0.3.0/bin/brush_app.exe'
    if not path.is_file():raise FileNotFoundError('缺少 Brush，请将 v0.3.0 放到 vendor/brush/v0.3.0/bin/brush_app.exe')
    return path


def is_splats(report):return report.get('representation')=='gaussian_splats' or report.get('method')==METHOD


def session_for(source):
    source=Path(source).resolve()
    if (source/'geometry_poses.json').is_file():return source
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    relative=report.get('source_session_relative')
    if relative:
        candidate=(source/relative.replace('\\','/')).resolve()
        if (candidate/'geometry_poses.json').is_file():return candidate
    from saved_sources import reference
    candidate=reference(report.get('source_session'),source)
    if candidate and (candidate/'geometry_poses.json').is_file():return candidate
    raise ValueError('Brush 3DGS 找不到关联的一阶段 RGB-D 与相机位姿；请导入完整数据集')


def splat_count(path):
    """Reject accidentally passing a point cloud/mesh as a Gaussian model."""
    properties=set();count=None;binary=False
    with Path(path).open('rb') as stream:
        if stream.readline()!=b'ply\n':raise ValueError('无效的高斯 PLY 文件')
        for _ in range(200):
            line=stream.readline(1024).decode('ascii').strip()
            if line=='format binary_little_endian 1.0':binary=True
            if line.startswith('element vertex '):count=int(line.split()[-1])
            if line.startswith('property float '):properties.add(line.split()[-1])
            if line=='end_header':break
        else:raise ValueError('无效的高斯 PLY 头部')
        required={'x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]}
        if not count or not binary or not required<=properties:raise ValueError('PLY 不包含有效 Brush 高斯模型')
        if Path(path).stat().st_size-stream.tell()<count*4*len(properties):raise ValueError('高斯模型数据不完整')
    return count


def validate_result(folder,report):
    from scene_reconstruction import result_path,digest
    if not report.get('complete') or not is_splats(report):raise ValueError('3DGS 结果不完整')
    model=result_path(folder,report['model_relative'])
    if digest(model)!=report['model_sha256']:raise ValueError('3DGS 模型校验失败')
    if splat_count(model)!=report['gaussians']:raise ValueError('3DGS 高斯数量不一致')
    for view in report.get('evaluation_views',[]):
        for name in ('render_relative','input_relative','comparison_relative'):
            if not result_path(folder,view[name]).is_file():raise FileNotFoundError('3DGS 缺少保存的预览图')
    if not report.get('evaluation_views'):raise ValueError('3DGS 缺少结果预览')
    return model


def consumed_files(session):
    """Hash precisely the saved observations used by the exporter."""
    graph=json.loads((session/'geometry_poses.json').read_text())
    rows={r['capture']:r for r in map(json.loads,(session/'frames.jsonl').read_text().splitlines())}
    captures={r['node']:r['capture'] for r in map(json.loads,(session/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted')}
    names=['geometry_poses.json','sdk_calibration.json','frames.jsonl','registrations.jsonl']
    from rgbd_geometry import recorded_coordinates
    for node in graph['node_ids']:
        row=rows[captures[node]];recorded_coordinates(row)
        stamp=f'{float(row["stamp"]):.6f}.png'
        names.extend([f'rgb/{stamp}',f'depth/{stamp}',f'map_frames/{int(node)}.npy'])
    from scene_reconstruction import digest
    return {name:digest(session/name) for name in names}


def train(dataset,output,cancel=None,progress=None,settings=None,timeout=600):
    settings={**DEFAULTS,**(settings or {})};output=Path(output);output.mkdir()
    steps=settings['steps'];eval_every=min(settings['eval_every'],steps)
    if steps<1:raise ValueError('训练步数必须为正整数')
    transforms=json.loads((Path(dataset)/'transforms.json').read_text())
    eval_count=sum(i%settings['eval_split_every']==0 for i in range(len(transforms['frames'])))
    command=[str(executable()),str(Path(dataset).resolve()),'--total-steps',str(steps),
        '--max-resolution',str(settings['resolution']),'--max-splats',str(settings['max_splats']),
        '--sh-degree',str(settings['sh_degree']),'--seed',str(settings['seed']),
        '--eval-split-every',str(settings['eval_split_every']),'--eval-every',str(eval_every),
        '--eval-save-to-disk','--export-every',str(steps),'--export-path',str(output.resolve())]
    _check(cancel);start=time.monotonic();last=-1;process=None
    with (output/'train.log').open('w',encoding='utf-8') as log:
        try:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while process.poll() is None:
                _check(cancel)
                if time.monotonic()-start>timeout:raise TimeoutError('Brush 训练超时，原有结果已保留')
                completed=max([0]+[int(p.name[5:]) for p in output.glob('eval_*') if p.is_dir()
                    and p.name[5:].isdigit() and len(list(p.glob('*.png')))==eval_count])
                if progress and completed!=last:
                    progress(dict(text=f'Brush 训练 · 已保存评估 {completed:,}/{steps:,} 步',percent=15+70*completed/steps));last=completed
                if cancel is not None:cancel.wait(.1)
                else:time.sleep(.1)
            _check(cancel)
            if process.returncode:
                log.flush();detail=(output/'train.log').read_text(encoding='utf-8',errors='replace')[-2000:]
                raise RuntimeError('Brush 训练失败：'+(detail or f'退出码 {process.returncode}'))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    model=output/f'export_{steps}.ply';count=splat_count(model)
    return dict(parameters=settings,gaussians=count,elapsed_seconds=time.monotonic()-start)


def reconstruct(scene,pending,session,cancel=None,progress=None,settings=None):
    from tools.export_brush import export
    from scene_reconstruction import write_json,digest
    session=Path(session).resolve() if session else None
    if session is None:raise ValueError('Brush 3DGS 需要关联一阶段的 RGB-D、内参和相机位姿；单独点云不能训练')
    executable();_check(cancel);settings={**DEFAULTS,**(settings or {})};start=time.perf_counter()
    before=consumed_files(session)
    if progress:progress(dict(text='正在导出 3DGS 图像、相机位姿与初始点…',percent=3))
    data=pending/'dataset'
    exported=export(session,data,eval_every=settings['eval_split_every'],cancel=cancel)
    if exported['train_views']<2 or exported['eval_views']<1:raise ValueError('3DGS 至少需要 2 个训练视角和 1 个评估视角，请增加采集视角')
    run=train(data,pending/'training',cancel,progress,settings)
    _check(cancel)
    if progress:progress(dict(text='正在保存 3DGS 模型与留出视角预览…',percent=90))
    model=pending/'gaussians.ply';shutil.copy2(pending/'training'/f'export_{settings["steps"]}.ply',model)
    views=[];preview=pending/'previews';preview.mkdir()
    transforms=json.loads((data/'transforms.json').read_text())
    for i,frame in enumerate(transforms['frames']):
        _check(cancel)
        if i%settings['eval_split_every']:continue
        name=Path(frame['file_path']).name
        rendered=pending/'training'/f'eval_{settings["steps"]}'/name;image=data/'images'/name
        prediction=cv2.imread(str(rendered));original=cv2.imread(str(image))
        if prediction is None or original is None:raise ValueError('Brush 未生成完整的留出视角预览')
        original=cv2.resize(original,(prediction.shape[1],prediction.shape[0]),interpolation=cv2.INTER_AREA)
        mse=float(np.mean((prediction.astype(float)-original.astype(float))**2));psnr=float(10*np.log10(255**2/max(mse,1e-10)))
        comparison=preview/name
        if not cv2.imwrite(str(comparison),np.hstack((original,prediction))):raise OSError('无法保存 3DGS 对照预览')
        views.append(dict(name=name,psnr_db=psnr,render_relative=rendered.relative_to(pending).as_posix(),
                          input_relative=image.relative_to(pending).as_posix(),comparison_relative=comparison.relative_to(pending).as_posix()))
    if before!=consumed_files(session):raise ValueError('训练期间 RGB-D、位姿或初始点发生变化，原有结果已保留')
    report=dict(schema='rgbd_scene_base_v1',complete=True,stage=3,method=METHOD,representation='gaussian_splats',
        source_relative=os.path.relpath(scene.source,pending),source_session_relative=os.path.relpath(session,pending),
        source_fingerprint=scene.report['source_fingerprint'],input_sha256=before,
        model_relative='gaussians.ply',model_sha256=digest(model),geometry_only=False,
        coordinates='world X forward Y left Z up',units='metres',parameters=run['parameters'],
        gaussians=run['gaussians'],training_seconds=run['elapsed_seconds'],elapsed_seconds=time.perf_counter()-start,
        train_views=exported['train_views'],eval_views=len(views),mean_psnr_db=float(np.mean([v['psnr_db'] for v in views])),
        evaluation_views=views,poses_refined=False,initial_points=exported['initial_points'],
        quality_note='Held-out RGB fit, not geometric accuracy; saved RGB camera poses are used unchanged.')
    write_json(pending/'report.json',report);validate_result(pending,report)
    # Keep final model, input cameras and final evaluation images; release intermediate checkpoints.
    for directory in (pending/'training').glob('eval_*'):
        if directory.is_dir() and directory.name!=f'eval_{settings["steps"]}':shutil.rmtree(directory)
    if progress:progress(dict(text='Brush 3DGS 已完成',percent=100))
    return report


