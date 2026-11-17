"""Compare the active workbench defaults on one unchanged stage-two input."""
from pathlib import Path
from datetime import datetime
import sys,json,os,argparse,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    from result_paths import latest_preprocessed
    from runtime_paths import geometry_python,activate_geometry_packages
    source=(args.source or latest_preprocessed()).resolve()
    output=(args.output or source.parent/('geometry_comparison_defaults_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not args.worker:
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
        return subprocess.call([str(geometry_python()),'-u',str(Path(__file__).resolve()),'--worker','--source',str(source),'--output',str(output)],env=env)
    activate_geometry_packages()
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    from geometry_comparison import load_source,measure
    from reconstruction_pipeline import run_pipeline,file_hash
    from geometry_methods import METHODS
    if output==source or source in output.parents:raise ValueError('Output must be outside source')
    points,_,hashes=load_source(source);tree=cKDTree(points)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,experiment='workbench_defaults',input=str(source),input_sha256=hashes,input_points=len(points),
                complete=False,parameters=dict(nksr_detail_level=.4),methods={},
                metric_note='Actual default final outputs. Poisson includes its production refinement; neural methods retain native model surfaces. These are not matched-postprocessing model-only results. Observation agreement is not ground-truth accuracy.')
    def save():
        (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for method in (name.lower() for name in METHODS):
        print('Running workbench default:',method,flush=True)
        stamp=time.perf_counter()
        try:
            result=run_pipeline(source,output/method,method=method)
            mesh=o3d.io.read_triangle_mesh(str(output/method/'mesh.ply'));mesh.compute_vertex_normals()
            metrics=result.get('metrics') or measure(mesh,points,tree)
            def display(a):return np.ascontiguousarray(np.column_stack((-a[:,1],-a[:,2],a[:,0])),dtype=np.float32)
            np.savez(output/method/'final.npz',vertices=display(np.asarray(mesh.vertices)),
                     triangles=np.asarray(mesh.triangles,dtype=np.uint32),normals=display(np.asarray(mesh.vertex_normals)))
            row=dict(status='complete',reconstruction_seconds=result['elapsed_seconds'],
                     wall_seconds=time.perf_counter()-stamp,variants=dict(final=metrics),parameters=result['parameters'],
                     pipeline=result['pipeline'],stage_seconds=result['stage_seconds'])
            if 'inference' in result:row['inference']=result['inference']
            report['methods'][method]=row
        except Exception as exc:report['methods'][method]=dict(status='failed',error=str(exc))
        save()
    report['input_unchanged']=hashes=={name:file_hash(source/name) for name in hashes}
    report['complete']=report['input_unchanged'] and all(r.get('status')=='complete' for r in report['methods'].values())
    save()
    lines=['# 默认几何模式对照（独立开发工具）','',report['metric_note'],'',
           '| 方法 | 顶点 | 三角面 | 10mm覆盖 | P95距离(mm) | 连通分量 | 核心流程秒 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,row in report['methods'].items():
        if row['status']!='complete':lines+=['',name+': '+row['error']];continue
        m=row['variants']['final']
        lines.append(f'| {name} | {m["vertices"]:,} | {m["triangles"]:,} | {m["input_coverage"]["10"]:.2%} | {m["input_to_mesh_mm"]["p95"]:.2f} | {m["components"]:,} | {row["reconstruction_seconds"]:.2f} |')
    lines+=['','Poisson保留当前正式的裁剪、平滑、减面和平面优化；NKSR默认detail_level=0.4、mise_iter=1。神经模型不追加外部裁剪或Poisson后处理。',
            '同一完整预处理来源；NoKSR按模型输入要求保留1cm网格唯一索引。CPU耗时是不同完整默认流程，不是纯模型公平速度基准，不代表CUDA速度。',
            '每个方法子目录含mesh.ply/report.json；本对照不写入正式setp3、setp4或dataset.json。工作台已移除对照入口，本工具仅供独立开发核验。']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Comparison saved:',output,flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':sys.exit(main())
