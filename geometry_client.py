from runtime_paths import geometry_python
"""Persistent geometry worker, isolated from the main Python/RTAB-Map DLL ABI."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


class GeometryClient:
    def __init__(self, folder, asynchronous=False):
        self.folder = Path(folder)
        self.log = (self.folder/'geometry_worker.log').open('w',encoding='utf-8')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
        self.process = subprocess.Popen([str(geometry_python()),'-u',str(ROOT/'geometry_worker.py'),str(self.folder)],
                                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                        text=True,encoding='utf-8',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.matcher = None
        self.asynchronous=asynchronous;self.background=None;self.pending_loop=None

    def request(self, command, **payload):
        self.process.stdin.write(json.dumps({'command':command,**payload})+'\n')
        self.process.stdin.flush()
        for line in self.process.stdout:
            if not line.startswith('@@GEOMETRY '):
                self.log.write(line); self.log.flush()
                continue
            result = json.loads(line[len('@@GEOMETRY '):])
            if 'error' in result:
                raise ValueError(result['error'])
            if command=='commit' and self.pending_loop is not None:
                job=self.pending_loop;self.pending_loop=None
                try:
                    if self.background is None:
                        from async_loop import AsyncLoop
                        self.background=AsyncLoop(self.folder)
                    self.background.submit(job)
                except Exception as exc:
                    Path(job['packet']).unlink(missing_ok=True)
                    self.log.write('Async startup: '+str(exc)+'\n');self.log.flush()
            return result
        raise RuntimeError(f'几何工作进程退出，详情：{self.folder / "geometry_worker.log"}')

    def register(self, points):
        if self.pending_loop is not None:
            Path(self.pending_loop['packet']).unlink(missing_ok=True);self.pending_loop=None
        path = self.folder/'geometry_input.npy'
        np.save(path,points)
        try:
            prepared = self.request('prepare',path=str(path))
        finally:
            # The worker has consumed this single-frame transfer file; no raw XYZ archive.
            path.unlink(missing_ok=True)
        local=prepared.get('local_candidates',prepared['candidates'])
        historical=prepared.get('historical_candidates',[])
        chosen=local if self.asynchronous else prepared['candidates']
        with np.load(prepared['packet']) as packet:
            arrays={key:packet[key].copy() for key in packet.files}
        result=self.request('finish',proposals=self._match(chosen,arrays))
        # Independent viewpoints may need historical relocalization to establish any pose.
        if self.asynchronous and not result['accepted'] and historical:
            result=self.request('finish',proposals=self._match(prepared['candidates'],arrays))
            historical=[]
            result['synchronous_relocalization']=True
        if self.asynchronous and result['accepted'] and historical:
            directory=self.folder/'loop_jobs';directory.mkdir(exist_ok=True)
            path=directory/f"{prepared['version']}_{prepared['source']}.npz"
            np.savez(path,**{key:value for key,value in arrays.items() if int(key.rsplit('_',1)[1]) in historical})
            self.pending_loop=dict(packet=str(path),source=prepared['source'],version=prepared['version'],references=historical,archives=str(self.folder/'registration_frames'))
        result['feature_points']=prepared['feature_points']
        result['async_loop']=bool(self.asynchronous)
        return result

    def discard_capture(self):
        if self.pending_loop is not None:
            Path(self.pending_loop['packet']).unlink(missing_ok=True);self.pending_loop=None
        return self.request('discard_capture')

    def _match(self,references,arrays):
        if references and self.matcher is None:
            from pointdsc_matcher import PointDSCMatcher
            self.matcher=PointDSCMatcher()
        proposals=[]
        for reference in references:
            try:
                pose=self.matcher.estimate(arrays[f'source_{reference}'],arrays[f'target_{reference}'])
                proposals.append(dict(reference=reference,pose=pose.tolist()))
            except (ValueError,RuntimeError) as exc:proposals.append(dict(reference=reference,error=str(exc)))
        return proposals

    def poll_loop(self):
        if self.background is None:return None
        response=self.background.poll()
        if response is None:return None
        if 'error' in response or not response.get('links'):
            return dict(accepted=False,reason=response.get('error','No valid historical matches'),background=response)
        try:
            result=self.request('propose_loop',source=response['source'],version=response['version'],links=response['links'])
            result['background']=response;return result
        except ValueError as exc:
            self.request('discard_loop')
            return dict(accepted=False,reason=str(exc),background=response)

    def background_busy(self):
        return self.background is not None and bool(self.background.running or self.background.pending)

    def stop_background(self):
        if self.background is not None:self.background.close()
        if self.pending_loop is not None:
            Path(self.pending_loop['packet']).unlink(missing_ok=True);self.pending_loop=None

    def close(self):
        self.stop_background()
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n'); self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self.process.kill(); self.process.wait()
        self.process.stdin.close(); self.process.stdout.close(); self.log.close()
