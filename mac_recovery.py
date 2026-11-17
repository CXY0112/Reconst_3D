"""Single bounded background recovery job; no camera/graph access in worker."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from runtime_paths import geometry_python

ROOT=Path(__file__).resolve().parent
DEFAULTS=dict(enabled=False,voxel_m=.03,max_input_points=12000,max_correspondences=256,
              graph_sigma_m=.3,graph_threshold=.99,max_cliques=10000,clique_timeout_s=2.,correspondence_spacing_m=.06,
              correspondence_distance_m=.06,min_correspondence_inliers=12,min_inlier_ratio=.1,
              timeout_s=12.,retry_interval_s=3.,max_seed_age_s=15.)


class MACRecovery:
    def __init__(self,folder,settings):
        self.folder=Path(folder);self.config={**DEFAULTS,**settings.get('mac',{})};self.validation=settings
        c=self.config
        if (not isinstance(c['enabled'],bool) or not 32<=c['max_correspondences']<=1024
                or any(type(c[k]) is not int for k in ('max_correspondences','max_input_points','max_cliques','min_correspondence_inliers'))
                or not 100<=c['max_input_points']<=60000 or not .005<=c['voxel_m']<=.15
                or not 0<c['graph_sigma_m']<=1 or not 0<c['graph_threshold']<1 or not 0<c['correspondence_spacing_m']<=.5
                or not 0<c['min_inlier_ratio']<=1 or not 3<=c['min_correspondence_inliers']<=c['max_correspondences']
                or not 1<=c['max_cliques']<=100000 or not 0<c['clique_timeout_s']<=10
                or not 1<=c['timeout_s']<=60 or not 0<c['retry_interval_s']<=60
                or not 1<=c['max_seed_age_s']<=60 or not 0<c['correspondence_distance_m']<=.2):
            raise ValueError('Invalid MAC recovery configuration')
        if c['enabled'] and not geometry_python().is_file():raise RuntimeError('MAC requires the project Open3D geometry Python runtime')
        self.process=None;self.log=None;self.seed=None;self.reference=None;self.job=0;self.next_try=0.
        self.status='idle';self.error=None;self.seed_time=0.;self.started=0.
        (self.folder/'mac_runtime.json').write_text(json.dumps(c,indent=2),encoding='utf-8')

    def _finish(self,kill=False):
        if self.process is not None:
            if kill and self.process.poll() is None:self.process.kill()
            self.process.wait();self.process=None
        if self.log is not None:self.log.close();self.log=None

    def reset(self):
        self._finish(kill=True);self.seed=None;self.reference=None;self.status='idle';self.next_try=0.

    def poll(self,reference,source,target,stamp):
        if not self.config['enabled']:return None
        now=time.monotonic()
        if self.reference is not None and self.reference!=reference:self.reset()
        if self.process is not None:
            if now-self.started>self.config['timeout_s']:
                self._finish(kill=True);self.status='failed';self.error='MAC worker timeout';self.next_try=now+self.config['retry_interval_s']
            elif self.process.poll() is not None:
                code=self.process.returncode;self._finish();self.next_try=now+self.config['retry_interval_s']
                try:
                    result=json.loads((self.folder/'mac_output.json').read_text(encoding='utf-8'))
                    if code!=0 or result['job']!=self.job or result['reference']!=reference or not result['ok']:
                        raise ValueError(result.get('error','MAC worker failed or stale output'))
                    self.seed=result['result'];self.seed_time=self.started;self.status='seed_ready';self.error=None
                except (OSError,ValueError,KeyError) as exc:self.status='failed';self.error=str(exc)
                with (self.folder/'mac_metrics.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(stamp=stamp,job=self.job,reference=reference,status=self.status,error=self.error,result=self.seed),ensure_ascii=False)+'\n')
        if self.seed is not None:
            if now-self.seed_time<=self.config['max_seed_age_s']:return np.asarray(self.seed['pose'],float)
            self.seed=None;self.status='expired'
        if self.process is None and now>=self.next_try:
            self.reference=reference;self.job+=1;self.started=now
            def bounded(points):
                a=np.asarray(points,float);a=a[np.isfinite(a).all(1)]
                return a[::max(1,int(np.ceil(len(a)/self.config['max_input_points'])))].copy()
            meta=dict(settings=self.config,validation=self.validation,reference=reference,job=self.job,source_stamp=stamp)
            np.savez_compressed(self.folder/'mac_input.npz',source=bounded(source),target=bounded(target),metadata=json.dumps(meta))
            (self.folder/'mac_output.json').unlink(missing_ok=True)
            self.log=(self.folder/'mac_worker.log').open('ab')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            try:
                self.process=subprocess.Popen([str(geometry_python()),str(ROOT/'mac_worker.py'),str(self.folder/'mac_input.npz'),str(self.folder/'mac_output.json')],
                    stdout=self.log,stderr=self.log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.status='running';self.error=None
            except OSError:
                self.log.close();self.log=None;raise
        return None

    def reject_seed(self,reason):
        self.seed=None;self.status='rejected';self.error=reason;self.next_try=time.monotonic()+self.config['retry_interval_s']

    def snapshot(self):return dict(enabled=self.config['enabled'],status=self.status,job=self.job,reference=self.reference,error=self.error)
    def close(self):self._finish(kill=True)
