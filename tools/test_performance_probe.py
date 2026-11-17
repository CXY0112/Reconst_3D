"""Smoke-test that the sampler counts child CPU and child private memory."""
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/geometry_packages'))
from profile_pipeline import measure
import psutil

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='.probe_test_',dir=ROOT/'results') as temporary:
        folder=Path(temporary)/'run'
        child='import time; block=bytearray(64*1024*1024); end=time.monotonic()+1.0\nwhile time.monotonic()<end: sum(range(1000))\ntime.sleep(.3)'
        parent='import subprocess,sys;subprocess.run([sys.executable,"-c",'+repr(child)+'],check=True)'
        summary=measure([sys.executable,'-c',parent],folder,psutil,interval=.05)
        assert summary['status']==0
        assert summary['peak_processes']>=2
        assert summary['peak_aggregate_private_commit_bytes']>=64*1024*1024
        assert summary['cpu_seconds']>.5
        print('Child-process CPU/private-memory sampling smoke test passed')
