"""Compare NoKSR against Poisson and the accepted NKSR detail_level=0.4 baseline."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--packet',type=Path,help='Reuse an already verified full-scene inference packet')
    parser.add_argument('--device',default=os.environ.get('RECONST_NOKSR_DEVICE','cpu'),choices=('cpu','cuda:0'))
    parser.add_argument('--input-points',type=int,default=-1,help='Default full input; 10000 tests the official ScanNet sampling count')
    parser.add_argument('--noksr-python',default=os.environ.get('RECONST_NOKSR_PYTHON','wsl:Ubuntu-22.04' if os.name=='nt' else sys.executable))
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    from result_paths import latest_preprocessed
    from runtime_paths import geometry_python,activate_geometry_packages
    source=(args.source or latest_preprocessed()).resolve()
    output=(args.output or source.parent/('geometry_comparison_noksr_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not args.worker:
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        return subprocess.call([str(geometry_python()),'-u',str(Path(__file__).resolve()),'--worker',
            '--source',str(source),'--output',str(output),'--device',args.device,'--noksr-python',args.noksr_python,
            '--input-points',str(args.input_points),
            *(['--packet',str(args.packet.resolve())] if args.packet else [])],env=env)
    activate_geometry_packages()
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    from geometry_comparison import load_source,save_method,run_comparison
    from reconstruction_pipeline import file_hash
    from nksr_runtime import inference_command,default_nksr_python
    if output==source or source in output.parents: raise ValueError('Output must be outside source')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'comparison.json').exists(): raise FileExistsError('Comparison already exists')
    points,normals,hashes=load_source(source)
    baseline=None
    for path in sorted(source.parent.glob('geometry_comparison_*/comparison.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('complete') and r.get('input_sha256')==hashes and r.get('parameters',{}).get('nksr_detail_level')==.4:
            baseline=(path.parent,r); break
    if baseline is None:
        folder=source.parent/('geometry_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        baseline=(folder,run_comparison(source,folder,default_nksr_python()))
        if not baseline[1]['complete']: raise RuntimeError('Poisson/NKSR baseline is incomplete')
    report={'schema_version':1,'experiment':'poisson_nksr_noksr','input':str(source),'input_points':len(points),
            'input_sha256':hashes,'complete':False,'parameters':dict(baseline[1]['parameters'],noksr_config='ScanNet_Serial',
                noksr_voxel_mm=20,noksr_intrinsic_mise_iter=0), 'methods':{},
            'metric_note':'Agreement with noisy observations, not ground-truth accuracy. Intrinsic extraction differs by model.'}
    def save():
        (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        lines=['# Poisson / NKSR 0.4 / NoKSR 对照','','完整对照：'+str(report['complete']),'',
               '| 方法 | 重建/提取 | 裁剪后顶点 | 裁剪后三角面 | 输入P95(mm) | 10mm覆盖 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,m in report['methods'].items():
            v=m.get('variants',{}).get('supported')
            if v: lines.append(f'| {name} | {m["reconstruction_seconds"]:.2f}s | {v["vertices"]:,} | {v["triangles"]:,} | {v["input_to_mesh_mm"]["p95"]:.2f} | {v["input_coverage"]["10"]:.2%} |')
        device=report['methods'].get('noksr',{}).get('inference',{}).get('device','尚未推理')
        lines+=['',f'NoKSR实际设备：{device}。Poisson与NKSR复用同一输入SHA256的既有成功结果，未重新计时；不要将混合设备时间视为公平GPU性能对照。',
                'NoKSR计时从权重载入开始，含输入准备、编码、解码与网格提取，排除网络构造和依赖导入；NKSR原对照包含模型构造。',
                '统一15mm外部观测支持裁剪；无平滑、减面或平面优化。各模型保留各自内置表面/掩码机制，不是容量完全相同的对照。',
                '输入距离为与噪声观测的一致性，不是绝对精度。CPU移植差异和来源校验见NoKSR推理元数据。','',
                '原始数据与正式setp3/setp4未覆盖。']
        (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save()
    try:
        for name in ('poisson','nksr'):
            destination=output/name; destination.mkdir()
            for variant in ('raw','supported'):
                for suffix in ('.ply','.npz'):
                    src=baseline[0]/name/(variant+suffix); dst=destination/src.name
                    shutil.copyfile(src,dst)
                    if file_hash(src)!=file_hash(dst): raise RuntimeError('Baseline copy mismatch')
            report['methods'][name]=dict(baseline[1]['methods'][name],reused_from=str(baseline[0])); save()
        packet=args.packet.resolve() if args.packet else output/'noksr_inference.npz'
        if args.packet is None:
            command=inference_command(args.noksr_python,ROOT/'tools/noksr_infer.py',
                ['--source',str(source),'--packet',str(packet),'--device',args.device,'--input-points',str(args.input_points)])
            if args.noksr_python.startswith('wsl:'):
                # This isolated venv shares the verified NKSR Torch/CUDA runtime.
                index=next(i for i,s in enumerate(command) if s.endswith('/bin/python'))
                command[index]='/opt/reconst_3d/noksr/env/bin/python'
            with (output/'noksr_inference.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('NoKSR inference failed; see inference log')
        metadata=json.loads(packet.with_suffix('.json').read_text(encoding='utf-8'))
        report['parameters']['noksr_voxel_mm']=float(metadata['voxel_m'])*1000
        if 'parameters' in metadata:
            report['parameters']['noksr_inference_parameters']=metadata['parameters']
        if metadata['smoke'] or metadata.get('source_points',metadata['input_points'])!=len(points) or metadata['input_sha256']!=hashes or not metadata['input_unchanged']:
            raise ValueError('Packet is not a verified scene result for this input')
        with np.load(packet,allow_pickle=False) as data:
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),o3d.utility.Vector3iVector(data['triangles']))
            seconds=float(data['seconds'])
        report['methods']['noksr']=save_method('noksr',mesh,points,cKDTree(points),output,seconds)
        report['methods']['noksr']['inference']=metadata
        report['methods']['noksr']['source']=json.loads((ROOT/'vendor/noksr_linux/source.json').read_text())
        report['methods']['noksr']['checkpoint']=json.loads((ROOT/'vendor/noksr_linux/checkpoint.json').read_text())
    except Exception as exc:
        report['error']=str(exc); raise
    finally:
        report['input_unchanged']=hashes=={name:file_hash(source/name) for name in hashes}
        report['complete']=report['input_unchanged'] and all(report['methods'].get(n,{}).get('status')=='complete' for n in ('poisson','nksr','noksr'))
        save()
    print('Comparison saved:',output,flush=True)
    return 0 if report['complete'] else 2

if __name__=='__main__': sys.exit(main())
