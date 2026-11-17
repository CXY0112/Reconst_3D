"""Deterministic confidence gate and local recovery hysteresis for G-CVO.

Scores measure configurable quality margins, not correctness probabilities.
Recovery probes retain the last trusted reference/initial guess; no global
relocalization or MAC is performed here.
"""
import math


DEFAULTS=dict(enabled=True,low_threshold=.55,recovery_threshold=.75,
              low_frames=3,recovery_good_frames=3,good_overlap=.75,good_rmse_m=.015)


class GCVOTracking:
    def __init__(self,settings):
        self.settings=settings
        self.config={**DEFAULTS,**settings.get('tracking',{})}
        c=self.config
        if (not isinstance(c['enabled'],bool)
                or not 0<c['low_threshold']<c['recovery_threshold']<=1
                or any(not isinstance(c[k],int) or isinstance(c[k],bool) or not 1<=c[k]<=100
                       for k in ('low_frames','recovery_good_frames'))
                or not settings.get('min_overlap',.35)<c['good_overlap']<=1
                or not 0<c['good_rmse_m']<settings.get('max_rmse_m',.035)):
            raise ValueError('Invalid G-CVO tracking thresholds')
        self.state='NORMAL';self.initialized=False;self.low_count=0;self.good_count=0
        self.confidence=None;self.components={};self.reason='等待首帧';self.allowed=False
        self.events=[];self.retried=False

    def _transition(self,state,stamp,reason):
        if state==self.state:return
        self.events.append(dict(stamp=stamp,previous=self.state,state=state,reason=reason))
        self.state=state

    def failure(self,stamp,reason):
        self.confidence=0.;self.components={};self.reason=reason;self.allowed=False;self.retried=False
        self.good_count=0
        if not self.initialized:return
        self.low_count+=1
        if self.config['enabled'] and self.low_count>=self.config['low_frames']:
            self._transition('RECOVERY',stamp,'连续配准失败或低置信度')

    def resume(self,stamp):
        if not self.initialized:return
        self.low_count=0;self.good_count=0;self.confidence=None;self.allowed=False
        self.reason='暂停后续接，重新确认最后可信参考'
        if self.config['enabled']:self._transition('RECOVERY',stamp,self.reason)

    def assess(self,details,metrics,stamp,initial=False):
        self.retried=bool(details.get('retried'))
        if initial:
            self.confidence=None;self.components={};self.reason='建立首帧参考';self.allowed=True
            return True
        c=self.config;s=self.settings
        overlap=metrics.get('overlap');rmse=metrics.get('rmse_m')
        translation=metrics.get('translation_m');rotation=metrics.get('rotation_deg')
        values=(overlap,rmse,translation,rotation)
        if (any(v is None or not math.isfinite(v) for v in values)
                or not 0<=overlap<=1 or min(rmse,translation,rotation)<0
                or details.get('solver_status',0)!=0):
            self.failure(stamp,'置信度所需几何或收敛数据无效');return False
        # The hard geometry/SE3 gates always run before this margin assessment.
        clip=lambda v:max(0.,min(1.,v))
        floor=s.get('min_overlap',.35);ceiling=s.get('max_rmse_m',.035)
        support=clip((overlap-floor)/(c['good_overlap']-floor))
        residual=clip((ceiling-rmse)/(ceiling-c['good_rmse_m']))
        motion=clip(1-max(translation/s.get('max_translation_m',.75),rotation/s.get('max_rotation_deg',60)))
        self.components=dict(overlap_margin=support,residual_margin=residual,motion_margin=motion,converged=True)
        self.confidence=.45*support+.35*residual+.20*motion
        threshold=c['recovery_threshold'] if self.state=='RECOVERY' else c['low_threshold']
        good=self.confidence>=threshold
        if not good:
            self.good_count=0;self.low_count+=1;self.reason='几何质量余量不足'
            if c['enabled'] and self.low_count>=c['low_frames']:
                self._transition('RECOVERY',stamp,'连续配准失败或低置信度')
        elif self.state=='RECOVERY':
            self.good_count+=1;self.reason='对最后可信参考帧的局部配准通过，等待连续确认'
        else:
            # Clear failures only after native acceptance, not on a candidate.
            self.good_count=0;self.reason='配准可信'
        self.allowed=not c['enabled'] or (good and (self.state=='NORMAL' or self.good_count>=c['recovery_good_frames']))
        return self.allowed

    def submitted(self,status,stamp):
        if status<=0:
            self.failure(stamp,'原生后端未接受候选位姿');return
        self.initialized=True
        if self.state=='RECOVERY':
            self._transition('NORMAL',stamp,'连续高置信度局部配准且原生后端接受')
            self.reason='已恢复局部跟踪'
        self.low_count=0;self.good_count=0

    def snapshot(self):
        return dict(state=self.state,initialized=self.initialized,confidence=self.confidence,
                    confidence_kind='heuristic_quality_margin_not_probability',components=self.components,
                    low_count=self.low_count,recovery_good_count=self.good_count,
                    recovery_good_required=self.config['recovery_good_frames'],
                    allowed=self.allowed,reason=self.reason,retried=self.retried,
                    policy_enabled=self.config['enabled'],recovery_method='gcvo_local_probe',mac_enabled=False)
