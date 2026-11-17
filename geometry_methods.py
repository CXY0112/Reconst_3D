"""Stage-three method defaults and the shared saved-mesh contract."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time

METHODS = ('Poisson', 'NKSR', 'TSDF')
PIPELINES = {'poisson': 'poisson_supported_smooth_qem_planar',
             'nksr': 'nksr_default', 'noksr': 'noksr_default', 'tsdf':'sdk_depth_tsdf'}
NOKSR_DEFAULTS = dict(k_neighbors=8, last_n_layers=4, mask_threshold_mm=30.,
                      voxel_mm=20., meshing_voxel_mm=15.)
DESCRIPTIONS = {
    'poisson': 'Poisson → 15mm支持裁剪 → 平滑 → 减面 → 平面优化；面数从40%起逐次检查',
    'nksr': 'NKSR：detail_level=0.4，mise_iter=1；保留模型原生表面，不追加Poisson后处理',
    'noksr': 'NoKSR：解码20mm、提取15mm、8邻居、4尺度、内部掩码30mm；不追加外部裁剪或平滑',
    'tsdf': 'SDK原始深度直接TSDF：5mm体素、30mm截断；使用一阶段内参/位姿，绕过二阶段点云，不追加平滑或减面',
}


def is_geometry_report(report):
    return bool(report.get('complete') and report.get('pipeline') in PIPELINES.values())


def run_neural_pipeline(source, output, method, preview_packet=None):
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    from geometry_comparison import load_source, measure
    from reconstruction_pipeline import file_hash
    from nksr_runtime import default_nksr_python, inference_command
    root=Path(__file__).resolve().parent
    source,output=Path(source).resolve(),Path(output).resolve()
    if not (source/'processed_world.npy').is_file():raise ValueError('NKSR需要二阶段预处理结果；跳过二阶段后请选择Poisson或TSDF')
    if method not in ('nksr','noksr'): raise ValueError('Unknown neural geometry method')
    if output==source or source in output.parents: raise ValueError('Output must be outside source')
    stamp=time.perf_counter()
    points,normals,hashes=load_source(source)
    output.mkdir(parents=True,exist_ok=False)
    parameters=(dict(detail_level=.4,mise_iter=1) if method=='nksr' else dict(NOKSR_DEFAULTS))
    report=dict(schema_version=1,stage=3,pipeline=PIPELINES[method],method=method,input=str(source),
                input_points=len(points),input_sha256=hashes,complete=False,parameters=parameters,
                units='metres',coordinates='world X forward Y left Z up',
                stages={},stage_seconds={},decimation_attempts=[],
                notes='Native neural mesh with model-internal support mask; no external support trim, smoothing, decimation or planar correction. Agreement with observations is not ground-truth accuracy.')
    def save(): (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        print('1/6 Loading preprocessed cloud and verified model runtime...',flush=True)
        with tempfile.TemporaryDirectory(prefix='.inference_',dir=output) as temporary:
            packet=Path(temporary)/'mesh.npz'
            arguments=['--source',str(source),'--packet',str(packet)]
            if method=='nksr':
                runtime=default_nksr_python()
                command=inference_command(runtime,root/'tools/compare_geometry.py',
                    ['--infer',*arguments,'--detail-level','.4','--device',os.environ.get('RECONST_NKSR_DEVICE','auto')])
            else:
                runtime=os.environ.get('RECONST_NOKSR_PYTHON','wsl:Ubuntu-22.04' if os.name=='nt' else sys.executable)
                command=inference_command(runtime,root/'tools/noksr_infer.py',
                    [*arguments,'--device',os.environ.get('RECONST_NOKSR_DEVICE','cpu')])
                if runtime.startswith('wsl:'):
                    index=next(i for i,s in enumerate(command) if s.endswith('/bin/python'))
                    command[index]='/opt/reconst_3d/noksr/env/bin/python'
            print('2/6 Neural reconstruction and native mesh extraction...',flush=True)
            with (output/'inference.log').open('w',encoding='utf-8') as log:
                status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
            if status:
                detail=(output/'inference.log').read_text(encoding='utf-8',errors='replace')[-2500:]
                raise RuntimeError(f'{method.upper()} inference failed; no fallback.\n{detail}')
            with np.load(packet,allow_pickle=False) as data:
                mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),o3d.utility.Vector3iVector(data['triangles']))
                report['stage_seconds']['model_inference']=float(data['seconds'])
            if method=='noksr':
                metadata=json.loads(packet.with_suffix('.json').read_text(encoding='utf-8'))
                if metadata['input_sha256']!=hashes or not metadata['input_unchanged'] or metadata['smoke']:
                    raise ValueError('NoKSR source validation failed')
                report['inference']=metadata
        print('3/6 Validating mesh geometry...',flush=True)
        mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
        if not len(mesh.triangles) or not np.isfinite(np.asarray(mesh.vertices)).all(): raise ValueError('Invalid reconstructed mesh')
        mesh.compute_vertex_normals()
        print('4/6 Measuring agreement with the unchanged source...',flush=True)
        metrics=measure(mesh,points,cKDTree(points))
        report['metrics']=metrics
        print('5/6 Checking source fingerprints...',flush=True)
        if hashes!={name:file_hash(source/name) for name in hashes}:raise RuntimeError('Input data changed')
        print('6/6 Saving stage-three mesh and preview...',flush=True)
        if not o3d.io.write_triangle_mesh(str(output/'mesh.ply'),mesh,write_vertex_colors=False):raise OSError('Mesh export failed')
        if preview_packet is not None:
            np.savez(preview_packet,vertices=np.asarray(mesh.vertices,dtype=np.float32),
                     triangles=np.asarray(mesh.triangles,dtype=np.uint32),normals=np.asarray(mesh.vertex_normals,dtype=np.float32))
        report['final']=dict(vertices=len(mesh.vertices),triangles=len(mesh.triangles),mesh_sha256=file_hash(output/'mesh.ply'))
        report.update(complete=True,source_unchanged=True,elapsed_seconds=time.perf_counter()-stamp)
        save();return report
    except Exception as exc:
        report.update(error=str(exc),elapsed_seconds=time.perf_counter()-stamp);save();raise
