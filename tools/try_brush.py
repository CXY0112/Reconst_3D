"""Bounded headless Brush experiment; does not modify the stage-three pipeline."""
import argparse,ctypes as ct,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


class Memory(ct.Structure):
    _fields_=[('cb',ct.c_ulong),('PageFaultCount',ct.c_ulong)]+[(k,ct.c_size_t) for k in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8000,help='Training steps (default: 8000)');p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--resolution',type=int,default=320);p.add_argument('--max-splats',type=int,default=30000)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);output=args.output.resolve()
    command=[str(ROOT/'vendor/brush/v0.3.0/bin/brush_app.exe'),str(args.dataset.resolve()),'--total-steps',str(args.steps),
             '--max-resolution',str(args.resolution),'--max-splats',str(args.max_splats),'--sh-degree','1','--eval-split-every','5',
             '--eval-every',str(min(250,args.steps)),'--eval-save-to-disk','--export-every',str(args.steps),'--export-path',str(output)]
    start=time.monotonic();peak=0;cancelled=False
    with (output/'train.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (output/'process.json').write_text(json.dumps(dict(pid=process.pid,command=command)))
        try:
            while process.poll() is None:
                counters=Memory();counters.cb=ct.sizeof(counters)
                if ct.windll.psapi.GetProcessMemoryInfo(ct.c_void_p(int(process._handle)),ct.byref(counters),ct.sizeof(counters)):
                    peak=max(peak,counters.PeakWorkingSetSize)
                if time.monotonic()-start>args.timeout:cancelled=True;process.kill();break
                time.sleep(.2)
            code=process.wait()
        finally:
            if process.poll() is None:process.kill();process.wait()
    report=dict(command=command,exit_code=code,timeout=cancelled,wall_s=time.monotonic()-start,
                process_peak_working_set_MiB=peak/2**20,notes='Working set is CPU process memory, not total GPU/shared-memory usage.')
    (output/'run.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if code or cancelled or not (output/f'export_{args.steps}.ply').is_file():raise RuntimeError('Brush experiment incomplete')


if __name__=='__main__':main()
