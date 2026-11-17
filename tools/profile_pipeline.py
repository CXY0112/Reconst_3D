"""Sample real process trees; uses existing vendored psutil, no installation needed."""
from pathlib import Path
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
import os
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'vendor/rtabmap/bin/python.exe'


def measure(command,folder,psutil,interval=.1):
    folder.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    started=time.monotonic();cached={};rows=[]
    with (folder/'stdout.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        root=psutil.Process(process.pid)
        while True:
            try:children=root.children(recursive=True)
            except psutil.Error:children=[]
            live=[]
            for proc in [root]+children:
                try:
                    with proc.oneshot():
                        cpu=proc.cpu_times();mem=proc.memory_info();io=proc.io_counters()
                        row={'pid':proc.pid,'name':proc.name(),'cpu':cpu.user+cpu.system,
                            'rss':mem.rss,'private':getattr(mem,'private',0),'threads':proc.num_threads(),
                            'read':io.read_bytes,'write':io.write_bytes}
                    cached[proc.pid]=row;live.append(row)
                except psutil.Error:pass
            rows.append({'time':time.monotonic(),'processes':len(live),
                'cpu_seconds':sum(x['cpu'] for x in cached.values()),'rss_bytes':sum(x['rss'] for x in live),
                'private_commit_bytes':sum(x['private'] for x in live),'threads':sum(x['threads'] for x in live),
                'read_bytes':sum(x['read'] for x in cached.values()),'write_bytes':sum(x['write'] for x in cached.values()),
                'system_available_bytes':psutil.virtual_memory().available})
            if process.poll() is not None:break
            time.sleep(interval)
        status=process.wait()
    with (folder/'resources.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    elapsed=time.monotonic()-started
    summary={'status':status,'wall_seconds_including_startup':elapsed,'sample_interval_seconds':interval,
        'cpu_seconds':max(x['cpu_seconds'] for x in rows),
        'cpu_average_core_equivalents':max(x['cpu_seconds'] for x in rows)/elapsed,
        'peak_aggregate_working_set_bytes':max(x['rss_bytes'] for x in rows),
        'peak_aggregate_private_commit_bytes':max(x['private_commit_bytes'] for x in rows),
        'peak_threads':max(x['threads'] for x in rows),'peak_processes':max(x['processes'] for x in rows),
        'logical_read_bytes':max(x['read_bytes'] for x in rows),'logical_write_bytes':max(x['write_bytes'] for x in rows),
        'output_file_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
        'seen_processes':[{'pid':x['pid'],'name':x['name'],'cpu_seconds':x['cpu']} for x in cached.values()]}
    events=folder/'events.jsonl'
    if events.exists():
        summary['events']=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines() if line.strip()]
    if (folder/'workload.json').exists():summary['workload']=json.loads((folder/'workload.json').read_text(encoding='utf-8'))
    (folder/'resource_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',action='store_true');parser.add_argument('--host-python',default=sys.executable)
    parser.add_argument('--output',type=Path);parser.add_argument('--gui-only',action='store_true')
    parser.add_argument('--stages',nargs='+',choices=['stage1','stage2','stage3','stage4','gui'])
    parser.add_argument('--stage1-core',action='store_true')
    args=parser.parse_args()
    if not args.controller:
        raise SystemExit(subprocess.call([str(RUNTIME),str(Path(__file__).resolve()),'--controller',
            '--host-python',sys.executable,*sys.argv[1:]]))
    sys.path.insert(0,str(ROOT/'vendor/geometry_packages'))
    import psutil
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=(args.output or ROOT/'results'/f'pipeline_performance_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=True)
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    hardware={'platform':platform.platform(),'cpu_name':cpu_name,'physical_cores':psutil.cpu_count(logical=False),
        'logical_processors':psutil.cpu_count(),'total_ram_bytes':psutil.virtual_memory().total,
        'available_ram_before_bytes':psutil.virtual_memory().available,'time_local':stamp,
        'target_python':args.host_python,'geometry_python':str(RUNTIME),'sampling':'process tree every 100ms; RSS sum includes shared pages',
        'io_note':'Windows process transfer bytes include cached I/O and pipes, not physical SSD bandwidth',
        'source':str(ROOT/'results/scan_20261015_102420_653218')}
    if not (output/'hardware.json').exists():
        (output/'hardware.json').write_text(json.dumps(hardware,ensure_ascii=False,indent=2),encoding='utf-8')
    source=Path(hardware['source']);script=ROOT/'tools/performance_worker.py'
    summaries=json.loads((output/'summary.json').read_text(encoding='utf-8')) if (output/'summary.json').exists() else {}
    if args.gui_only:jobs=[('gui',args.host_python,source,'gui')]
    else:jobs=[('stage1',args.host_python,source,'stage1')]+[
        ('stage2',args.host_python,source,f'stage2_run{i}') for i in (1,2)]+[
        ('stage3',str(RUNTIME),ROOT/'results/setp2_20261015_170233_921076',f'stage3_run{i}') for i in (1,2)]+[
        ('stage4',str(RUNTIME),ROOT/'results/setp3_20261015_170235_174939',f'stage4_run{i}') for i in (1,2)]
    if args.stages:jobs=[job for job in jobs if job[0] in args.stages]
    for stage,python,input_path,name in jobs:
        if stage=='stage1' and args.stage1_core:name='stage1_core'
        original_name=name;attempt=2
        while (output/name).exists():name=f'{original_name}_attempt{attempt}';attempt+=1
        folder=output/name;print('Measuring',name,flush=True)
        command=[python,'-u',str(script),stage,'--source',str(input_path),'--folder',str(folder)]
        if stage=='stage1' and args.stage1_core:command.append('--core-only')
        summary=measure(command,folder,psutil)
        summaries[name]=summary
        (output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,'status',summary['status'],'seconds',round(summary['wall_seconds_including_startup'],2),
              'peakMiB',round(summary['peak_aggregate_working_set_bytes']/2**20),flush=True)
        if summary['status']:print('See',folder/'stdout.log',flush=True)
    print('Report folder:',output,flush=True)


if __name__=='__main__':main()
